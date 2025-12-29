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

logger = logging.getLogger("train")

def merge_dataloaders(dataloaders, max_steps_list):
    import numpy as np
    
    # Calculate sampling probabilities based on max_steps
    total_steps = sum(max_steps_list)
    probabilities = [steps / total_steps for steps in max_steps_list]
    
    # Create iterators that cycle when exhausted
    from itertools import cycle
    iterators = [cycle(dl) for dl in dataloaders]
    
    while True:
        # Sample a dataloader based on probabilities
        idx = np.random.choice(len(iterators), p=probabilities)
        yield next(iterators[idx])

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

    if args.data.merge_group:
        merged_loaders = []
        merged_steps = []
        merged = set()
        to_merge_loaders = []
        to_merge_steps = []
        for i, single_data in enumerate(train_data):
            if single_data in args.data.merge_group and single_data not in merged:
                to_merge_loaders.append(train_data_loaders[i])
                to_merge_steps.append(args.max_steps[i])
                merged.add(single_data)
            else:
                merged_loaders.append(train_data_loaders[i])
                merged_steps.append(args.max_steps[i])

        merged_loaders.insert(0, merge_dataloaders(to_merge_loaders, to_merge_steps))
        merged_steps.insert(0, sum(to_merge_steps))

        train_data_loaders = merged_loaders
        args.max_steps = merged_steps

    interleaved_tokenizer.duration_sec = args.eval_duration_sec

    eval_data_loaders = {}
    if args.do_eval_loss:
        eval_data_loaders['eval_loss'] = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            mode="eval_loss",
        )

    if args.do_mmlu:
        eval_data_loaders['mmlu'] = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=1,
            seed=args.seed,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            mode="mmlu",
        )

    if args.do_slm:
        eval_data_loaders['swuggy'] = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=1,
            seed=args.seed,
            rank=0,
            world_size=1,  # DDP world_size
            mode="swuggy",
        )
        
        eval_data_loaders['sblimp'] = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=1,
            seed=args.seed,
            rank=0,
            world_size=1,  # DDP world_size
            mode="sblimp"
        )

        eval_data_loaders['ssc'] = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=1,
            seed=args.seed,
            rank=0,
            world_size=1,  # DDP world_size
            mode="ssc"
        )

    if args.do_cd:
        for cd_dataset in args.data.cd_data:
            eval_data_loaders[cd_dataset] = build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=1,
                seed=args.seed,
                rank=0,
                world_size=1,  # DDP world_size
                mode="cd",
                pretrain_data=cd_dataset
            )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    if args.text_llm.enable:
        if not args.only_text:
            projector_keys = ["text_projector", "depformer_projector"]
            finetune_keys  = ["text_head", "depformer_text_emb", "depformer_text_proj"
                              "depformer.layers", "depformer_emb", "linears."]
        else:
            projector_keys = ["text_projector"]
            finetune_keys = ["text_head"]

        projectors, projector_names = pick_params_by_keywords(model, projector_keys)
        finetune,   finetune_names  = pick_params_by_keywords(model, finetune_keys)

        keep = projector_names | finetune_names
        for n, p in model.named_parameters():
            p.requires_grad = (n in keep)

        lr_projectors = args.optim.proj_lr
        lr_finetune = args.optim.finetune_lr

        wd_projectors = getattr(args.optim, "wd_projectors", args.optim.proj_wd)
        wd_finetune   = getattr(args.optim, "wd_finetune",   args.optim.finetune_wd)

        param_groups = [
            {"params": projectors, "lr": lr_projectors, "weight_decay": wd_projectors},
            {"params": finetune,   "lr": lr_finetune,   "weight_decay": wd_finetune},
        ]    

        optimizer = AdamW(
            param_groups,
            lr=args.optim.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.optim.weight_decay,
        )

        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr_projectors, lr_finetune],
            total_steps=sum(args.max_steps),
            pct_start=args.optim.pct_start,
        )
    else:        
        optimizer = AdamW(
            model.parameters(),
            lr=args.optim.lr,
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=args.optim.weight_decay,
        )

        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.optim.lr,
            total_steps=sum(args.max_steps),
            pct_start=args.optim.pct_start,
        )

    state = TrainState(sum(args.max_steps))

    # 8. Initialize checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.train()
    torch.cuda.empty_cache()

    dataset_idx = 0
    total_max_steps = sum(args.max_steps)
    for dataset_steps, train_data_loader in zip(args.max_steps, train_data_loaders):
        main_logger_info(f"Starting dataset {dataset_idx} for {dataset_steps} steps")
        dataset_step_end = state.step + dataset_steps

        while state.step < dataset_step_end:
            state.start_step()
            is_last_step = (state.step + 1) == total_max_steps

            optimizer.zero_grad()

            loss = torch.tensor([0.0], device="cuda")
            n_batch_tokens: int = 0
            n_real_tokens: int = 0

            for i in range(args.num_microbatches):
                batch = next(train_data_loader)
                codes = batch.codes

                condition_tensors = None
                if batch.condition_attributes is not None:
                    condition_tensors = model.condition_provider.prepare(
                        batch.condition_attributes
                    )

                # forward / backward
                output = model(codes=codes, condition_tensors=condition_tensors)
                text_loss = compute_loss_with_mask(
                    output.text_logits,
                    codes[:, : model.audio_offset],
                    output.text_mask,
                    mode="text",
                    text_padding_weight=args.text_padding_weight,
                    text_padding_ids={
                        model.text_padding_token_id,
                        model.end_of_text_padding_id,
                    },
                )
                mb_loss = text_loss
                if not args.only_text and state.step % 4 != 0:
                    audio_loss = compute_loss_with_mask(
                        output.logits,
                        codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                        output.mask,
                        mode="audio",
                        first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                    )
                    mb_loss += audio_loss

                mb_loss.backward()

                loss += mb_loss.detach()
                n_batch_tokens += output.text_mask.numel() + output.mask.numel()
                n_real_tokens += (
                    torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
                )

                if i < args.num_microbatches - 1:
                    # synchronize CUDA to re-run backward
                    assert args.num_microbatches > 1  # should not happen
                    torch.cuda.synchronize()

            if args.num_microbatches > 1:
                loss /= args.num_microbatches
                for p in model.parameters():
                    if p.requires_grad:
                        assert p.grad is not None
                        p.grad.div_(args.num_microbatches)

            # upcast params for optimizer update
            upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

            # clip grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            # optimizer step
            optimizer.step()

            # downcast params for forward & backward
            downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

            last_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # Host sync
            loss_item = loss.item()
            avg_loss = avg_aggregate(loss_item)

            if args.do_eval and (
                (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
            ):
                # write perplexity to state
                evaluate(model, text_tokenizer, eval_data_loaders, state, args)

                eval_logs = get_eval_logs(
                    state.step,
                    avg_loss,
                    state.this_eval_perplexity,
                    state.this_eval_loss,
                    state.this_text_loss,
                    state.this_audio_loss,
                    state.this_mmlu_acc,
                    state.this_swuggy_acc,
                    state.this_sblimp_acc,
                    state.this_ssc_acc,
                    state.this_cd_acc
                )

                main_logger_info(eval_log_msg(eval_logs))
                eval_logger.log(eval_logs, step=state.step)

            # Timing
            state.end_step(n_batch_tokens)

            if state.step % args.log_freq == 0:
                train_logs = get_train_logs(
                    state,
                    avg_loss,
                    n_real_tokens,
                    last_lr,
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.memory_allocated(),
                    args,
                )
                main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
                metrics_logger.log(train_logs, step=state.step)

            if args.do_ckpt and (
                (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
            ):
                checkpointer.save_checkpoint(
                    save_only_lora=not args.full_finetuning and args.save_adapters,
                    dtype=param_dtype,
                )

        dataset_idx += 1

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
