import torch
from torch.nn import functional as F
from typing import Literal

def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    reduction: Literal["mean", "none"] = "mean"
):
    B, K, T = target.shape
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)

    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))

    if reduction == "mean":
        return  torch.sum(mb_loss) / torch.sum(weights)
    elif reduction == "none":
        mb_loss = mb_loss.view(B, K, T)
        weights = weights.view(B, K, T)
        return mb_loss.sum(dim=(1, 2)) / (weights.sum(dim=(1, 2)))
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")