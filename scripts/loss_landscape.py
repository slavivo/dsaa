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

    # ...existing code...
    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)

    # --- Loss Landscape Analysis Logic Starts Here ---
    from tqdm import tqdm
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    # Create output directory for plots
    plot_dir = run_dir / "loss_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    main_logger_info(f"Saving analysis plots to {plot_dir}")

    model.eval()
    
    # Unwrap model if necessary
    inner_model = model
    if hasattr(model, "module"):
        inner_model = model.module
    elif hasattr(model, "_fsdp_wrapped_module"):
        inner_model = model._fsdp_wrapped_module

    main_logger_info("Starting loss analysis loop...")

    # We need the padding token IDs to identify them in the analysis
    text_pad_id = inner_model.text_padding_token_id
    
    # Loss function without reduction to get per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    # Accumulators for global stats
    stats = {
        "pad_loss_sum": 0.0, "pad_count": 0,
        "speech_loss_sum": 0.0, "speech_count": 0,
        "transition_loss_sum": 0.0, "transition_count": 0,
        "audio_loss_sum": 0.0, "audio_count": 0
    }
    
    samples_to_plot = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Analyzing Loss")):
            # Move to device
            codes = batch.codes.to("cuda") # [B, K, T]
            
            # Prepare conditions if available (same as train.py)
            condition_tensors = None
            if batch.condition_attributes is not None:
                 condition_tensors = inner_model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # Forward pass
            output = inner_model(codes=codes, condition_tensors=condition_tensors)
            
            # --- 1. Text Loss Landscape ---
            token_losses = None
            is_pad = None
            is_transition = None
            
            if output.text_logits is not None:
                text_logits = output.text_logits.squeeze(1) # [B, T, V]
                text_targets = codes[:, 0, :] # [B, T]
                
                # Flatten for loss computation then reshape
                B, T, V = text_logits.shape
                flat_logits = text_logits.reshape(-1, V)
                flat_targets = text_targets.reshape(-1)
                
                # Compute raw loss per token
                token_losses = loss_fct(flat_logits, flat_targets)
                token_losses = token_losses.reshape(B, T)
                
                # Mask out invalid positions
                text_mask = output.text_mask.squeeze(1)
                token_losses = token_losses * text_mask.float()
                
                # Identify Regions
                is_pad = (text_targets == text_pad_id)
                is_pad_valid = is_pad & text_mask.bool()
                is_speech = (~is_pad) & text_mask.bool()
                
                # Identify Transitions (PAD -> Speech)
                # Shift is_pad to find where previous was pad and current is speech
                prev_is_pad = torch.roll(is_pad, shifts=1, dims=1)
                prev_is_pad[:, 0] = False # Boundary condition
                is_transition = prev_is_pad & is_speech

                # Accumulate Stats
                stats["pad_loss_sum"] += token_losses[is_pad_valid].sum().item()
                stats["pad_count"] += is_pad_valid.sum().item()
                
                stats["speech_loss_sum"] += token_losses[is_speech].sum().item()
                stats["speech_count"] += is_speech.sum().item()

                if is_transition.any():
                    stats["transition_loss_sum"] += token_losses[is_transition].sum().item()
                    stats["transition_count"] += is_transition.sum().item()
            
            # --- 2. Audio Loss Landscape ---
            audio_loss_tensor = None
            if output.logits is not None:
                audio_logits = output.logits # [B, 8, T, V]
                audio_targets = codes[:, inner_model.audio_offset : inner_model.audio_offset + inner_model.dep_q, :] # [B, 8, T]
                
                # Compute average audio loss per timestep across codebooks
                audio_losses_per_step = []
                for k in range(audio_logits.shape[1]):
                    k_logits = audio_logits[:, k, :, :]
                    k_targets = audio_targets[:, k, :]
                    k_loss = loss_fct(k_logits.reshape(-1, k_logits.shape[-1]), k_targets.reshape(-1))
                    k_loss = k_loss.reshape(B, T)
                    audio_losses_per_step.append(k_loss)
                
                audio_loss_tensor = torch.stack(audio_losses_per_step, dim=1).mean(dim=1)
                valid_audio_mask = output.mask[:, 0, :] # [B, T]
                audio_loss_tensor = audio_loss_tensor * valid_audio_mask.float()

                stats["audio_loss_sum"] += audio_loss_tensor.sum().item()
                stats["audio_count"] += valid_audio_mask.sum().item()

            # --- 3. Store Sample for Plotting ---
            if i < 5 and token_losses is not None:
                samples_to_plot.append({
                    "batch_idx": i,
                    "text_loss": token_losses[0].cpu().numpy(), # Take first sequence
                    "audio_loss": audio_loss_tensor[0].cpu().numpy() if audio_loss_tensor is not None else None,
                    "is_pad": is_pad[0].cpu().numpy(),
                    "is_transition": is_transition[0].cpu().numpy()
                })
            
            # Limit number of batches for analysis
            if i >= 50: break

    # --- Final Reporting ---
    avg_pad_loss = stats["pad_loss_sum"] / max(1, stats["pad_count"])
    avg_speech_loss = stats["speech_loss_sum"] / max(1, stats["speech_count"])
    avg_trans_loss = stats["transition_loss_sum"] / max(1, stats["transition_count"])
    avg_audio_loss = stats["audio_loss_sum"] / max(1, stats["audio_count"])

    main_logger_info("\n" + "="*30)
    main_logger_info("LOSS LANDSCAPE ANALYSIS RESULTS")
    main_logger_info("="*30)
    main_logger_info(f"Avg Text Loss (PAD):    {avg_pad_loss:.4f}")
    main_logger_info(f"Avg Text Loss (SPEECH): {avg_speech_loss:.4f}")
    main_logger_info(f"Avg Text Loss (START):  {avg_trans_loss:.4f} (First token after PAD)")
    main_logger_info(f"Avg Audio Loss:         {avg_audio_loss:.4f}")
    main_logger_info("="*30)

    # --- Plotting ---
    for sample in samples_to_plot:
        idx = sample["batch_idx"]
        T = len(sample["text_loss"])
        x = np.arange(T)
        
        plt.figure(figsize=(15, 6))
        
        # Plot Text Loss
        plt.plot(x, sample["text_loss"], label="Text Loss", color="blue", alpha=0.7, linewidth=1)
        
        # Highlight PAD regions
        plt.fill_between(x, 0, sample["text_loss"].max(), where=sample["is_pad"], 
                         color="gray", alpha=0.2, label="PAD Tokens")
        
        # Highlight Transitions
        trans_indices = np.where(sample["is_transition"])[0]
        if len(trans_indices) > 0:
            plt.scatter(trans_indices, sample["text_loss"][trans_indices], color="red", zorder=5, label="Turn Start")

        # Plot Audio Loss if available
        if sample["audio_loss"] is not None:
            plt.plot(x, sample["audio_loss"], label="Audio Loss (Avg)", color="green", alpha=0.5, linewidth=1)

        plt.title(f"Loss Landscape of Moshi")
        plt.xlabel("Time Step")
        plt.ylabel("Cross Entropy Loss")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        out_file = plot_dir / f"loss_landscape_batch_{idx}.pdf"
        plt.savefig(out_file)
        plt.close()

if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
