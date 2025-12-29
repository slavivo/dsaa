import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

# from torch.profiler import ProfilerActivity, profile

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders
from moshi.utils.sampling import sample_token
import sphn
import numpy as np

from transformers import AutoModelForCausalLM, AutoConfig


logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    if args.text_llm.enable:
        config = AutoConfig.from_pretrained(args.text_llm.hf_repo_id, trust_remote_code=True)
        config.sliding_window = None
        config.output_hidden_states = True
        config.output_attentions = False
        config.attn_implementation = "flash_attention_2"
        config.return_dict_in_generate = True
        text_llm = AutoModelForCausalLM.from_pretrained(
            args.text_llm.hf_repo_id,
            config=config,
            device_map="cuda",
            torch_dtype=param_dtype,
            trust_remote_code=True,  # Needed for some Text-llm-specific code
        )
        checkpoint_info.text_llm = text_llm
    else:
        checkpoint_info.text_llm = None

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    tokenizer_repo_id = None if not args.text_llm.enable else args.text_llm.hf_repo_id
    text_tokenizer = checkpoint_info.get_text_tokenizer(tokenizer_repo_id)
    pad_token_id, epad_token_id, bos_token_id = None, None, None
    if args.text_llm.enable:
        pad_token_id = text_tokenizer.encode(" ")[0]
        epad_token_id = pad_token_id
        bos_token_id = text_tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token = '<|im_start|>'
            bos_token_id = text_tokenizer.encode(bos_token)[0]

    model = get_fsdp_model(args, checkpoint_info, pad_token_id, epad_token_id, bos_token_id)

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    args.data.ssc_data = 'eval_test'

    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=1,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        mode="ssc",
    )

    # 11. train!
    # text_tokenizer = checkpoint_info.get_text_tokenizer()

    model.eval()

    # load the first sample
    sample = next(iter(data_loader))
    codes = sample.codes.to("cuda")
    len_codes = codes.shape[2]

    pad_tensor = torch.tensor([3, 1031, 243, 1178, 546, 1736, 1572, 1978, 2008, 1031, 243, 1178, 546, 1736, 1572, 1978, 2008], device='cuda', dtype=torch.int32)
    pad_tensor = pad_tensor.unsqueeze(0).unsqueeze(-1)

    input_tensor = codes[:, :, :1].clone()

    for i in range(1, 30, 1):
        # print(f"Input first half: {input_tensor}")
        if i >= len_codes:
            user_tokens = pad_tensor
        else:
            user_tokens = codes[:, :, i:i+1]

        output = model(codes=input_tensor)
        last_text_logit = output.text_logits[:, :, -1, :].unsqueeze(2)
        last_semantic_logit = output.logits[:, 0, -1, :].unsqueeze(1).unsqueeze(2)
        # for i in range(last_audio_logits.shape[1]):
        #     print(f"Ith tensor: {last_audio_logits[0, i, 0, :]}")
        # print(f"Text shape: {last_text_logit.shape}")
        # print(f"Audio shape: {last_audio_logits.shape}")
        text_token = sample_token(last_text_logit.float(), True, 0.5, 25)
        semantic_token = sample_token(last_semantic_logit.float(), True, 0.8, 250)
        # print(f"Text token shape: {text_token.shape}")
        # print(f"Audio token shape: {audio_tokens.shape}")
        # print(f"Generated first half, text: {text_token}, audio: {audio_tokens}")

        next_step = torch.empty(17)
        next_step[0] = text_token[0, 0, 0]
        next_step[1] = semantic_token[0, 0, 0]
        # next_step[2:] = input_tensor[0, 2:, -1]
        next_step[2:] = user_tokens[0, 2:, 0]

        input_tensor[0, :, -1] = next_step
        input_tensor = torch.cat([input_tensor, pad_tensor], dim=2)

        # print(f"Input second half: {input_tensor}")
        output = model(codes=torch.cat([input_tensor, pad_tensor], dim=2))
        last_audio_logits = output.logits[:, :, -2, :].unsqueeze(2)
        audio_tokens = sample_token(last_audio_logits.float(), True, 0.8, 250)
        # print(f"Generated second half, text: {text_token}, audio: {audio_tokens}")

        next_step[0] = input_tensor[0, 0, -2]
        next_step[1] = input_tensor[0, 1, -2]
        next_step[2:9] = audio_tokens[0, 1:, 0]
        next_step[9:] = input_tensor[0, 9:, -2]

        input_tensor[0, :, -2] = next_step

        # input_tensor = torch.cat([input_tensor, pad_tensor], dim=2)

    # text = [tokenizer.id_to_piece(t.item()) for t in input_tensor[0, 0, 1:]]
    text = text_tokenizer.decode(input_tensor[0, 0, 1:].tolist())
    print(f"###Main text:\n {''.join(text)}")

    print(f"Generated text: {input_tensor[0, 0, :]}")
    print(f"Generated audio: {input_tensor[:, 1:9, :]}")
    audio = mimi.decode(input_tensor[:, 1:9, :])[0]
    sphn.write_wav(
        "gen_main.wav",
        audio[0].cpu().numpy().astype(np.float32),
        mimi.sample_rate,
    )

if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
