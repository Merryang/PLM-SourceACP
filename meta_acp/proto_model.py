"""Prototype and supervised heads for PLM-SourceACP.

The module keeps one small interface around any encoder that returns a
fixed-size sequence embedding. It supports both standard binary prediction
and episodic prototype logits used by the source-adaptation experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """Wrap an encoder with a metric projection and an auxiliary BCE head."""

    def __init__(self, encoder: nn.Module, proj_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(encoder.out_dim, encoder.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.out_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
        )
        self.log_tau = nn.Parameter(torch.zeros(()))

    def embed(self, batch: dict) -> torch.Tensor:
        z = self.encoder(batch)
        z = self.proj(z)
        return F.normalize(z, p=2, dim=-1)

    def aux_logit(self, batch: dict) -> torch.Tensor:
        z = self.embed(batch)
        return self.classifier(z).squeeze(-1)

    def predict_proba(self, batch: dict) -> torch.Tensor:
        return torch.sigmoid(self.aux_logit(batch))

    def compute_prototypes(self, support_pos: dict, support_neg: dict) -> torch.Tensor:
        z_pos = self.embed(support_pos).mean(dim=0)
        z_neg = self.embed(support_neg).mean(dim=0)
        protos = torch.stack([z_neg, z_pos], dim=0)
        return F.normalize(protos, p=2, dim=-1)

    def prototype_logits(self, query: dict, prototypes: torch.Tensor) -> torch.Tensor:
        z = self.embed(query)
        tau = self.log_tau.exp().clamp(min=1e-3, max=100.0)
        return tau * z @ prototypes.t()

    def forward_episode(
        self,
        support_pos: dict,
        support_neg: dict,
        query_pos: dict,
        query_neg: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        protos = self.compute_prototypes(support_pos, support_neg)
        q_batch = _cat_batches(query_pos, query_neg)
        logits = self.prototype_logits(q_batch, protos)
        labels = torch.cat(
            [
                torch.ones(_batch_size(query_pos), dtype=torch.long, device=logits.device),
                torch.zeros(_batch_size(query_neg), dtype=torch.long, device=logits.device),
            ],
            dim=0,
        )
        return logits, labels


def _batch_size(batch: dict) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("batch has no tensor values")


def _cat_batches(left: dict, right: dict) -> dict:
    out = {}
    for key, value in left.items():
        if torch.is_tensor(value) and key in right and torch.is_tensor(right[key]):
            out[key] = torch.cat([value, right[key]], dim=0)
    return out
