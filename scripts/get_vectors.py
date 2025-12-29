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

from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path

logger = logging.getLogger("train")

def pick_params_by_keywords(model, keywords):
    keywords = [k.lower() for k in keywords]
    params = []
    names = set()
    for n, p in model.named_parameters():
        ln = n.lower()
        if any(k in ln for k in keywords):
            params.append(p)
            names.add(n)
    return params, names


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

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

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
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    train_data = args.data.train_data
    if type(train_data) == str:
        train_data = [train_data]

    train_data_loaders = []
    for single_data in train_data:
        args.data.train_data = single_data
        train_data_loaders.append(
            build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=args.seed,
                rank=get_rank(),  # DDP rank
                world_size=get_world_size(),  # DDP world_size
            )
        )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    
    state = TrainState(sum(args.max_steps))

    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.eval()
    torch.cuda.empty_cache()
    save_dir = Path("saved_vectors")
    save_dir.mkdir(parents=True, exist_ok=True)


    dataset_idx = 0
    total_max_steps = sum(args.max_steps)
    for dataset_steps, train_data_loader in zip(args.max_steps, train_data_loaders):
        main_logger_info(f"Starting dataset {dataset_idx} for {dataset_steps} steps")
        dataset_step_end = state.step + dataset_steps

        while state.step < dataset_step_end:
            state.start_step()

            for i in range(args.num_microbatches):
                batch = next(train_data_loader)
                codes = batch.codes

                condition_tensors = None
                if batch.condition_attributes is not None:
                    condition_tensors = model.condition_provider.prepare(
                        batch.condition_attributes
                    )

                # forward / backward
                vectors = model.get_vectors(codes=codes, condition_tensors=condition_tensors)
                torch.save(vectors, save_dir / f"batch_{state.step:05d}.pt")
                
        dataset_idx += 1

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
