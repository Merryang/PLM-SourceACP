"""TDD: PrototypicalNetwork — prototype shape, episode forward, temperature, L2-norm, aux head."""

from __future__ import annotations

import pytest
import torch

from encoders import ScratchEncoder
from proto_model import PrototypicalNetwork


def _toy_batch(B: int = 4, L: int = 16) -> dict:
    return {"input_ids": torch.randint(1, 21, (B, L), dtype=torch.long)}


# ---------------------------------------------------------------------------
# Construction & temperature
# ---------------------------------------------------------------------------


def test_temperature_is_learnable_parameter():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=128)
    assert isinstance(net.log_tau, torch.nn.Parameter)
    assert net.log_tau.requires_grad


def test_proj_dim_controls_metric_space_dimension():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64)
    z = net.embed(_toy_batch())
    assert z.shape[-1] == 64


def test_embeddings_are_l2_normalized():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64).eval()
    with torch.no_grad():
        z = net.embed(_toy_batch(B=6))
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# Episode forward
# ---------------------------------------------------------------------------


def test_compute_prototypes_shape_is_2_by_d():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64).eval()
    s_pos = _toy_batch(B=3)
    s_neg = _toy_batch(B=3)
    with torch.no_grad():
        protos = net.compute_prototypes(s_pos, s_neg)
    assert protos.shape == (2, 64)


def test_forward_episode_returns_query_logits():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64).eval()
    s_pos, s_neg = _toy_batch(B=2), _toy_batch(B=2)
    q_pos, q_neg = _toy_batch(B=5), _toy_batch(B=5)
    with torch.no_grad():
        logits, labels = net.forward_episode(s_pos, s_neg, q_pos, q_neg)
    assert logits.shape == (10, 2)
    assert labels.shape == (10,)
    # Convention: first half query labels = 1 (positives), second half = 0.
    assert labels[:5].tolist() == [1] * 5
    assert labels[5:].tolist() == [0] * 5


def test_forward_episode_finite_and_differentiable():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64)
    s_pos, s_neg = _toy_batch(B=2), _toy_batch(B=2)
    q_pos, q_neg = _toy_batch(B=3), _toy_batch(B=3)
    logits, labels = net.forward_episode(s_pos, s_neg, q_pos, q_neg)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()  # gradient must flow through the encoder
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert has_grad


# ---------------------------------------------------------------------------
# Aux BCE head
# ---------------------------------------------------------------------------


def test_aux_logit_returns_one_logit_per_sample():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64).eval()
    with torch.no_grad():
        logit = net.aux_logit(_toy_batch(B=7))
    assert logit.shape == (7,)
    assert torch.isfinite(logit).all()


# ---------------------------------------------------------------------------
# Inference helper (predict prob of positive class via aux head)
# ---------------------------------------------------------------------------


def test_predict_proba_is_in_unit_interval():
    net = PrototypicalNetwork(encoder=ScratchEncoder(), proj_dim=64).eval()
    with torch.no_grad():
        p = net.predict_proba(_toy_batch(B=4))
    assert p.shape == (4,)
    assert ((p >= 0.0) & (p <= 1.0)).all()
