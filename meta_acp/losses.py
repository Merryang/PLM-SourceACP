"""Loss functions for the meta_acp pipeline.

- `proto_ce_loss`: standard cross-entropy on prototype-distance logits.
- `focal_proto_loss`: focal-weighted variant for imbalanced episodes.
- `FocalBinaryLoss`: re-exported from the parent acp_pipeline so the aux
  BCE head shares the validated implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def proto_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on (N, 2) prototype logits with hard labels in {0, 1}."""
    return F.cross_entropy(logits, labels.long())


def focal_proto_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Focal-weighted CE on (N, 2) prototype logits.

    Reduces to standard CE when gamma=0 and alpha=0.5.
    """
    labels = labels.long()
    log_prob = F.log_softmax(logits, dim=-1)  # (N, 2)
    log_pt = log_prob.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)
    pt = log_pt.exp().clamp(min=1e-8, max=1.0)
    # Per-class alpha, normalized so alpha=0.5 means "no class weighting"
    # (each class gets weight 1.0 → reduces cleanly to CE when gamma=0).
    alpha_pos = 2.0 * alpha
    alpha_neg = 2.0 * (1.0 - alpha)
    alpha_t = torch.where(
        labels == 1,
        torch.tensor(alpha_pos, device=logits.device),
        torch.tensor(alpha_neg, device=logits.device),
    )
    loss = -alpha_t * (1.0 - pt) ** gamma * log_pt
    return loss.mean()


class FocalBinaryLoss(torch.nn.Module):
    """Mirror of acp_pipeline.losses.FocalBinaryLoss (kept local to avoid
    cross-package import friction)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def class_balanced_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """neg/pos ratio for `BCEWithLogitsLoss(pos_weight=...)`."""
    pos = labels.sum().clamp(min=1.0)
    neg = (labels.numel() - labels.sum()).clamp(min=1.0)
    return neg / pos
