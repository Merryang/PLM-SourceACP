"""TDD: encoder modules — shape, NaN-free, masking, registry."""

from __future__ import annotations

import pytest
import torch

from dataset import PAD_IDX
from encoders import ProtT5FrozenEncoder, ScratchEncoder, build_encoder


# ---------------------------------------------------------------------------
# ScratchEncoder
# ---------------------------------------------------------------------------


def test_scratch_encoder_out_dim_is_512():
    enc = ScratchEncoder()
    assert enc.out_dim == 512
    assert ScratchEncoder.OUT_DIM == 512


def test_scratch_encoder_forward_shape_and_finite():
    enc = ScratchEncoder().eval()
    ids = torch.randint(1, 21, (4, 32), dtype=torch.long)
    out = enc({"input_ids": ids})
    assert out.shape == (4, 512)
    assert torch.isfinite(out).all()


def test_scratch_encoder_handles_full_padding_without_nan():
    """All-PAD batch: mean/max pool denominators must not divide by zero."""
    enc = ScratchEncoder().eval()
    ids = torch.full((2, 16), PAD_IDX, dtype=torch.long)
    out = enc({"input_ids": ids})
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all(), "encoder produced NaN/Inf on fully padded input"


def test_scratch_encoder_padding_does_not_affect_real_tokens():
    """Adding PAD positions to a sequence should not change its embedding much."""
    enc = ScratchEncoder().eval()
    short = torch.randint(1, 21, (1, 10), dtype=torch.long)
    pad_tail = torch.full((1, 22), PAD_IDX, dtype=torch.long)
    long = torch.cat([short, pad_tail], dim=1)
    with torch.no_grad():
        # Pad short to same length so we can call encoder once each.
        short_padded = torch.cat([short, torch.full((1, 22), PAD_IDX, dtype=torch.long)], dim=1)
        z1 = enc({"input_ids": short_padded})
        z2 = enc({"input_ids": long})
    assert torch.allclose(z1, z2), "extra PAD changed the embedding"


# ---------------------------------------------------------------------------
# ProtT5FrozenEncoder
# ---------------------------------------------------------------------------


def test_prott5_encoder_out_dim_is_512():
    enc = ProtT5FrozenEncoder()
    assert enc.out_dim == 512


def test_prott5_encoder_forward_shape_and_finite():
    enc = ProtT5FrozenEncoder().eval()
    B, L, D = 3, 12, 1024
    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 8:] = False  # variable length
    out = enc({"embedding": emb, "mask": mask})
    assert out.shape == (B, 512)
    assert torch.isfinite(out).all()


def test_prott5_encoder_masking_changes_output():
    """Different masks over the same embedding must produce different outputs
    (else masking is silently ignored)."""
    enc = ProtT5FrozenEncoder().eval()
    emb = torch.randn(1, 16, 1024)
    m_full = torch.ones(1, 16, dtype=torch.bool)
    m_half = torch.ones(1, 16, dtype=torch.bool)
    m_half[0, 8:] = False
    with torch.no_grad():
        z_full = enc({"embedding": emb, "mask": m_full})
        z_half = enc({"embedding": emb, "mask": m_half})
    assert not torch.allclose(z_full, z_half, atol=1e-4)


# ---------------------------------------------------------------------------
# build_encoder registry
# ---------------------------------------------------------------------------


def test_build_encoder_dispatches_by_name():
    assert isinstance(build_encoder("scratch"), ScratchEncoder)
    assert isinstance(build_encoder("prott5"), ProtT5FrozenEncoder)
    # Case-insensitive
    assert isinstance(build_encoder("SCRATCH"), ScratchEncoder)


def test_build_encoder_unknown_raises():
    with pytest.raises(ValueError):
        build_encoder("nonexistent")
