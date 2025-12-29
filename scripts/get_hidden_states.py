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
# from finetune.eval import evaluate
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

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

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

    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.eval_duration_sec
    )

    args.data.shuffle = False
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=1,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        mode="eval_loss",
        is_finite=True
    )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)

    # --- Extraction Logic Starts Here ---
    from moshi.models.lm import _delay_sequence
    from tqdm import tqdm

    # Create output directory
    save_dir = run_dir / "extracted_states"
    save_dir.mkdir(exist_ok=True, parents=True)
    main_logger_info(f"Saving extracted states to {save_dir}")

    model.eval()
    
    # Unwrap model if necessary (handle FSDP/DDP wrappers to access custom methods)
    inner_model = model
    if hasattr(model, "module"):
        inner_model = model.module
    elif hasattr(model, "_fsdp_wrapped_module"):
        inner_model = model._fsdp_wrapped_module

    main_logger_info("Starting extraction loop...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Extracting")):
            # Move to device
            codes = batch.codes.to("cuda") # [B, K, T]
            
            # 1. Prepare Inputs (Apply Delays)
            # We replicate the input preparation logic from LMModel.forward
            B, K, T = codes.shape
            initial = inner_model._get_initial_token().expand(B, -1, -1)
            delayed_codes = _delay_sequence(inner_model.delays, codes, initial)
            # Insert empty tokens for the first time step
            delayed_codes = torch.cat([initial, delayed_codes], dim=2)
            # Remove last step to match target shape (standard Moshi practice)
            input_codes = delayed_codes[:, :, :-1] # [B, K, T]
            
            batch_results = {}
            
            # 2. Extract Hidden States
            # Check if it is TextLMModel (Modified) or standard LMModel (Moshi)
            if hasattr(inner_model, 'text_llm') and inner_model.text_llm is not None:
                # --- Modified Model Case ---
                
                # A. Audio Branch (Temporal Audio)
                # This is the output of the Moshi-part transformer processing only audio codes
                audio_out = inner_model.forward_temporal_audio(input_codes)
                batch_results['modified_audio'] = audio_out.cpu()
                
                # B. Text Branch (Fused Temporal Text)
                # This is the output of the Qwen-part AFTER fusing with audio_out
                temporal_out, _ = inner_model.forward_temporal_text(input_codes, audio_out)
                batch_results['modified_text_fused'] = temporal_out.cpu()
                
                # C. Vanilla Qwen (Raw Text LLM output)
                # We run the text codes through the internal LLM without any audio fusion
                text_codes = input_codes[:, 0, :]
                qwen_out = inner_model.text_llm(
                    text_codes, 
                    output_hidden_states=True, 
                    output_attentions=False, 
                    output_logits=False
                )
                # Extract the last hidden state
                batch_results['vanilla_qwen'] = qwen_out.hidden_states[-1].cpu()
                
            else:
                # --- Baseline Moshi Case ---
                # Standard forward pass returning the transformer output
                transformer_out, _ = inner_model.forward_text(input_codes)
                batch_results['moshi_baseline'] = transformer_out.cpu()
            
            # 3. Save Batch
            save_path = save_dir / f"batch_{i:04d}.pt"
            torch.save(batch_results, save_path)
            
            # Optional: Limit number of batches for quick testing
            if i >= 50: break

    main_logger_info("Extraction done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
