"""TDD: losses module — proto-CE, focal-proto, gradient flow."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from losses import FocalBinaryLoss, focal_proto_loss, proto_ce_loss


# ---------------------------------------------------------------------------
# proto_ce_loss
# ---------------------------------------------------------------------------


def test_proto_ce_loss_zero_when_perfect():
    """Logits hugely favoring the correct class → loss ≈ 0."""
    # 4 queries, 2 classes
    logits = torch.tensor(
        [
            [10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = proto_ce_loss(logits, labels)
    assert loss.item() < 1e-3


def test_proto_ce_loss_high_when_wrong():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    labels = torch.tensor([1, 0], dtype=torch.long)  # both wrong
    loss = proto_ce_loss(logits, labels)
    assert loss.item() > 5.0


def test_proto_ce_loss_matches_torch_cross_entropy():
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,), dtype=torch.long)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(proto_ce_loss(logits, labels), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# focal_proto_loss
# ---------------------------------------------------------------------------


def test_focal_proto_with_gamma_zero_alpha_half_equals_ce():
    """γ=0 and α=0.5 (equal weighting) → reduces to standard CE."""
    logits = torch.randn(16, 2)
    labels = torch.randint(0, 2, (16,), dtype=torch.long)
    fp = focal_proto_loss(logits, labels, gamma=0.0, alpha=0.5)
    ce = F.cross_entropy(logits, labels)
    assert torch.allclose(fp, ce, atol=1e-5)


def test_focal_proto_downweights_easy_examples():
    """Mostly correct predictions → focal loss must be smaller than CE."""
    logits = torch.tensor([[5.0, -5.0]] * 10)  # confident class-0 predictions
    labels = torch.zeros(10, dtype=torch.long)
    ce = F.cross_entropy(logits, labels).item()
    fp = focal_proto_loss(logits, labels, gamma=2.0, alpha=0.5).item()
    assert fp < ce, f"focal {fp} should be < ce {ce} for easy examples"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_proto_ce_loss_is_differentiable():
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = proto_ce_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_proto_is_differentiable():
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = focal_proto_loss(logits, labels, gamma=2.0, alpha=0.25)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Re-exported FocalBinaryLoss (for aux BCE head)
# ---------------------------------------------------------------------------


def test_focal_binary_loss_smaller_than_bce_on_easy_negatives():
    logits = torch.tensor([-5.0, -5.0, -5.0, -5.0])  # confident negatives
    targets = torch.zeros(4)
    bce = F.binary_cross_entropy_with_logits(logits, targets).item()
    fl = FocalBinaryLoss(alpha=0.25, gamma=2.0)(logits, targets).item()
    assert fl < bce
