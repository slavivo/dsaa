import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.distributed as dist
from finetune.args import TrainArgs
from finetune.distributed import (
    BACKEND,
    get_rank,
    is_torchrun,
    set_device,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger
)
from finetune.monitoring.utils import set_logger
from finetune.utils import logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

from transformers import AutoModelForCausalLM, AutoConfig

from torch import nn
from safetensors.torch import save_model

class MaskedEmbedding(nn.Module):
    def __init__(self, original_emb, zero_idx):
        super().__init__()
        self.zero_idx = zero_idx

        self.emb = nn.Embedding(
            num_embeddings=original_emb.num_embeddings,
            embedding_dim=original_emb.embedding_dim,
            device=original_emb.weight.device,
            dtype=original_emb.weight.dtype,
        )
        self.emb.weight.data.copy_(original_emb.weight.data)

    def forward(self, x):
        is_zero = (x == self.zero_idx)
        x_safe = x.clamp_min(0)
        y = self.emb(x_safe)

        y = torch.where(
            is_zero[..., None],
            torch.zeros(1, device=y.device, dtype=y.dtype),
            y,
        )
        return y

class TextPreservingProjector(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        
        # Only process the concatenated features for refinement
        input_dim = text_dim + audio_dim
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize to near-zero output
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fusion[-1].bias)
        
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable mixing

    def forward(self, x):
        text_part = x[..., :self.text_dim]
        # Text passes through unchanged, fusion adds audio-aware refinement
        fusion_out = self.fusion(x)
        return text_part + self.alpha * fusion_out

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
    model = get_fsdp_model(args, checkpoint_info)

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

    model.text_projector = TextPreservingProjector(2048, 4096, 4096, 2048).to(model.device)
    model.text_head = nn.Linear(
        in_features=text_llm.get_output_embeddings().in_features,
        out_features=text_llm.get_output_embeddings().out_features,
        bias=False,
    )
    model.text_head.weight.data = text_llm.get_output_embeddings().weight.data.clone()
    model.text_head.to(dtype=text_llm.get_output_embeddings().weight.dtype, device=model.device)
    model.depformer_projectors = nn.ModuleList(
        [ResidualProjector(4096+2048+1024, 4096, 1024).to(model.device) for _ in range(model.dep_q)]
    )
    original_emb = text_llm.get_input_embeddings()
    embedding_matrix = original_emb.weight.data.float()
    model.depformer_text_emb = MaskedEmbedding(text_llm.get_input_embeddings(), -1)
    masked_emb = MaskedEmbedding(original_emb, -1)
    model.text_llm.hf_model.set_input_embeddings(masked_emb)

    model.depformer_text_proj = nn.Linear(text_llm.config.hidden_size, model.depformer_dim, bias=True).to(model.device)
    # # Perform PCA for initialization
    mean_embedding = embedding_matrix.mean(dim=0, keepdim=True)
    centered_embeddings = embedding_matrix - mean_embedding
    U, S, Vt = torch.linalg.svd(centered_embeddings, full_matrices=False)
    truncated_vt = Vt[:model.depformer_dim, :]
    model.depformer_text_proj.weight.data.copy_(truncated_vt.to(model.device))
    mean_projection = torch.matmul(mean_embedding, truncated_vt.t())
    model.depformer_text_proj.bias.data.copy_(-mean_projection.squeeze(0))

    save_model(model, 'weights/model.safetensors')
                                              
    main_logger_info("done!")

if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
