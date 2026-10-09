"""Two interchangeable sequence encoders.

Both expose the same surface: forward(batch_dict) -> Tensor (B, embed_dim).
This lets PrototypicalNetwork swap encoders via a single CLI flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import VOCAB_SIZE, PAD_IDX


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        z = x.mean(dim=-1)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.unsqueeze(-1)


class ScratchEncoder(nn.Module):
    """AA-Embedding -> multi-scale CNN (k=3,5,7) -> SE -> BiGRU -> mean+max pool."""

    OUT_DIM = 512

    def __init__(self, embed_dim: int = 128, cnn_channels: int = 128, gru_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, cnn_channels, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
        )
        merged = cnn_channels * 3
        self.se = _SEBlock(merged)
        self.gru = nn.GRU(
            input_size=merged,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # mean + max pool over BiGRU output (2*gru_hidden each).
        self._out_dim = 4 * gru_hidden
        assert self._out_dim == self.OUT_DIM, f"OUT_DIM mismatch: {self._out_dim}"

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, batch: dict) -> torch.Tensor:
        ids = batch["input_ids"]  # (B, L)
        mask = (ids != PAD_IDX).float()  # (B, L)
        x = self.embedding(ids)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L)
        feats = [F.relu(c(x)) for c in self.convs]
        h = torch.cat(feats, dim=1)  # (B, 3C, L)
        h = self.se(h)
        h = h.transpose(1, 2)  # (B, L, 3C)
        h, _ = self.gru(h)  # (B, L, 2*gru_hidden)
        h = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = h.sum(dim=1) / denom
        # masked max-pool: set padded positions to a large negative value
        h_masked = h.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        max_pool, _ = h_masked.max(dim=1)
        z = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 4*gru_hidden)
        return self.dropout(z)


class GenericPLMFrozenEncoder(nn.Module):
    """Operates on cached residue-level PLM embeddings.

    The PLM weights themselves are NOT loaded here; they are baked into an
    offline cache produced by `plm_embeddings.py`. This head is intentionally
    small enough to train on desktop GPUs.
    """

    OUT_DIM = 512

    def __init__(
        self,
        in_dim: int = 1024,
        proj_dim: int = 256,
        cnn_channels: int = 128,
        dropout: float = 0.3,
        use_multiscale: bool = True,
        use_se: bool = True,
        pooling: str = "attn_mean_max",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.use_multiscale = use_multiscale
        self.use_se = use_se
        self.pooling = pooling
        kernels = (3, 5, 7) if use_multiscale else (3,)
        self.proj_in = nn.Linear(in_dim, proj_dim)
        self.convs = nn.ModuleList([nn.Conv1d(proj_dim, cnn_channels, kernel_size=k, padding=k // 2) for k in kernels])
        merged = cnn_channels * len(kernels)
        self.se = _SEBlock(merged) if use_se else nn.Identity()
        self.attn = nn.Linear(merged, 1)
        self.dropout = nn.Dropout(dropout)
        pool_parts = 1 if pooling == "mean" else 3
        self.head = nn.Linear(merged * pool_parts, self.OUT_DIM)

    @property
    def out_dim(self) -> int:
        return self.OUT_DIM

    def forward(self, batch: dict) -> torch.Tensor:
        emb = batch["embedding"]  # (B, L, D)
        mask = batch["mask"].float()  # (B, L)
        x = F.gelu(self.proj_in(emb))
        x = x.transpose(1, 2)  # (B, 256, L)
        feats = [F.relu(c(x)) for c in self.convs]
        h = torch.cat(feats, dim=1)
        h = self.se(h)
        h = h.transpose(1, 2)  # (B, L, 384)
        h = h * mask.unsqueeze(-1)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = h.sum(dim=1) / denom
        if self.pooling == "mean":
            z = mean_pool
            return self.dropout(F.gelu(self.head(z)))

        scores = self.attn(h).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(mask == 0, -1e4)
        w = torch.softmax(scores, dim=-1)
        attn_pool = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        max_pool, _ = h.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1)

        z = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        z = self.dropout(F.gelu(self.head(z)))
        return z


class ProtT5FrozenEncoder(GenericPLMFrozenEncoder):
    """Backward-compatible 1024-dim ProtT5 head."""

    IN_DIM = 1024

    def __init__(self, cnn_channels: int = 128, dropout: float = 0.3):
        super().__init__(in_dim=self.IN_DIM, cnn_channels=cnn_channels, dropout=dropout)


def build_encoder(name: str) -> nn.Module:
    name = name.lower()
    if name == "scratch":
        return ScratchEncoder()
    if name in {"prott5", "plm"}:
        return ProtT5FrozenEncoder()
    raise ValueError(f"Unknown encoder: {name}")
