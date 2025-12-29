import torch
from torch import nn

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

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class TextResidualProjector(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        input_dim = text_dim + audio_dim
        
        # Main transformation path
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable mixing

    def forward(self, x):
        text_part = x[..., :self.text_dim]
        fusion_out = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return text_part + self.alpha * fusion_out


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

class ResidualProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)  # dim-align skip

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x):
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return self.skip(x) + h

class GatedProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.core  = MLPProjector(input_dim, hidden_dim, output_dim, dropout)
        self.gate  = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.Sigmoid())
    def forward(self, x):
        out = self.core(x)
        g   = self.gate(x)
        return out * g

def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)

class CompresSAEEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, k: int):
        super().__init__()
        self.k = k
        self.encoder_w = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))

    @staticmethod
    def topk_mask(e: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        e_topk = torch.topk(torch.abs(e), k, dim)
        return torch.zeros_like(e).scatter(dim, e_topk.indices, e_topk.values) * torch.sign(e)

    def encode(self, x: torch.Tensor, apply_activation: bool = True) -> torch.Tensor:
        e_pre = l2_normalize(x) @ self.encoder_w + self.encoder_b
        return self.topk_mask(e_pre, self.k) if apply_activation else e_pre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
