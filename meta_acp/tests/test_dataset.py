"""TDD: dataset module — covers known pitfalls (truncation, unknown AA, sampler balance, hash, collate)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dataset import (
    AA_VOCAB,
    BalancedEpisodicSampler,
    EmbeddingDataset,
    PAD_IDX,
    SequenceDataset,
    VOCAB_SIZE,
    collate_embedding_batch,
    collate_token_batch,
    encode_sequence,
    seq_hash,
)


# ---------------------------------------------------------------------------
# encode_sequence
# ---------------------------------------------------------------------------


def test_encode_truncates_to_max_len():
    arr = encode_sequence("A" * 500, max_len=64)
    assert arr.shape == (64,)
    assert (arr == AA_VOCAB["A"]).sum() == 64


def test_encode_pads_short_sequences_with_pad_idx():
    arr = encode_sequence("ACD", max_len=10)
    assert arr.shape == (10,)
    assert arr[0] == AA_VOCAB["A"]
    assert arr[1] == AA_VOCAB["C"]
    assert arr[2] == AA_VOCAB["D"]
    assert (arr[3:] == PAD_IDX).all()


def test_encode_unknown_amino_acid_maps_to_pad():
    """Selenocysteine 'U' and gap '-' are not in the vocab; must map to 0."""
    arr = encode_sequence("AUX-B", max_len=8)
    assert arr[0] == AA_VOCAB["A"]
    assert arr[1] == 0  # U -> unknown
    assert arr[2] == 0  # X -> unknown
    assert arr[3] == 0  # - -> unknown
    assert arr[4] == 0  # B -> unknown


def test_vocab_size_constant_matches_definition():
    assert VOCAB_SIZE == len(AA_VOCAB) + 1 == 21


# ---------------------------------------------------------------------------
# seq_hash
# ---------------------------------------------------------------------------


def test_seq_hash_is_deterministic():
    assert seq_hash("ACDEFG") == seq_hash("ACDEFG")


def test_seq_hash_distinguishes_different_sequences():
    assert seq_hash("ACDEFG") != seq_hash("ACDEFH")


# ---------------------------------------------------------------------------
# SequenceDataset
# ---------------------------------------------------------------------------


def test_sequence_dataset_returns_expected_keys(tiny_csv: Path):
    ds = SequenceDataset(str(tiny_csv), max_len=64)
    item = ds[0]
    assert set(item.keys()) >= {"input_ids", "label", "length", "seq_hash"}
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].shape == (64,)
    assert item["label"].dtype == torch.float32
    assert item["label"].item() in (0.0, 1.0)


def test_sequence_dataset_length_matches_csv(tiny_csv: Path):
    df = pd.read_csv(tiny_csv)
    ds = SequenceDataset(str(tiny_csv), max_len=64)
    assert len(ds) == len(df)


def test_doubao_csv_schema_intact(doubao_csv_paths):
    """Real-data smoke: required columns must exist."""
    required = {"binary_label", "sequence", "length"}
    for name, p in doubao_csv_paths.items():
        if not p.exists():
            pytest.skip(f"{name} CSV not present at {p}")
        df = pd.read_csv(p, nrows=5)
        assert required.issubset(df.columns), f"missing cols in {name}: {required - set(df.columns)}"


# ---------------------------------------------------------------------------
# BalancedEpisodicSampler
# ---------------------------------------------------------------------------


def test_balanced_sampler_yields_equal_class_counts(tiny_csv: Path):
    df = pd.read_csv(tiny_csv)
    sampler = BalancedEpisodicSampler(df, k_shot=2, q_query=3, seed=0)
    s_pos, s_neg, q_pos, q_neg = sampler.sample_episode()
    assert len(s_pos) == len(s_neg) == 2
    assert len(q_pos) == len(q_neg) == 3
    # All s_pos indices must reference positives, etc.
    assert (df.iloc[s_pos]["binary_label"] == 1).all()
    assert (df.iloc[s_neg]["binary_label"] == 0).all()
    assert (df.iloc[q_pos]["binary_label"] == 1).all()
    assert (df.iloc[q_neg]["binary_label"] == 0).all()


def test_balanced_sampler_episodes_are_independent(tiny_csv: Path):
    df = pd.read_csv(tiny_csv)
    sampler = BalancedEpisodicSampler(df, k_shot=2, q_query=2, seed=0)
    e1 = sampler.sample_episode()
    e2 = sampler.sample_episode()
    # Two consecutive episodes should differ (with overwhelming probability for our tiny set).
    assert not all(np.array_equal(a, b) for a, b in zip(e1, e2))


def test_balanced_sampler_warns_when_class_too_small(tiny_csv: Path, capsys):
    """8 positives in tiny_csv; ask for K+Q=12 → must warn and use replacement."""
    df = pd.read_csv(tiny_csv)
    sampler = BalancedEpisodicSampler(df, k_shot=4, q_query=8, seed=0)
    captured = capsys.readouterr()
    assert "WARN" in captured.out
    assert sampler.replace_pos is True
    s_pos, s_neg, q_pos, q_neg = sampler.sample_episode()
    assert len(s_pos) + len(q_pos) == 12  # still returns the requested count


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def test_collate_token_batch_stacks_correctly(tiny_csv: Path):
    ds = SequenceDataset(str(tiny_csv), max_len=32)
    batch = collate_token_batch([ds[0], ds[1], ds[2]])
    assert batch["input_ids"].shape == (3, 32)
    assert batch["label"].shape == (3,)


def test_collate_embedding_batch_pads_to_max_length():
    items = [
        {"embedding": torch.randn(5, 1024), "label": torch.tensor(1.0), "length": torch.tensor(5)},
        {"embedding": torch.randn(8, 1024), "label": torch.tensor(0.0), "length": torch.tensor(8)},
        {"embedding": torch.randn(3, 1024), "label": torch.tensor(1.0), "length": torch.tensor(3)},
    ]
    batch = collate_embedding_batch(items)
    assert batch["embedding"].shape == (3, 8, 1024)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].sum().item() == 5 + 8 + 3
    # Padded positions must remain zero (we wrote into pad-zeros tensor).
    assert (batch["embedding"][0, 5:] == 0).all()
    assert (batch["embedding"][2, 3:] == 0).all()


# ---------------------------------------------------------------------------
# EmbeddingDataset
# ---------------------------------------------------------------------------


def test_embedding_dataset_returns_correct_residue_tensor(tiny_csv: Path, tiny_embedding_cache: Path):
    ds = EmbeddingDataset(str(tiny_csv), str(tiny_embedding_cache), max_len=64)
    item = ds[0]
    assert "embedding" in item and "label" in item and "seq_hash" in item
    assert item["embedding"].dim() == 2 and item["embedding"].shape[1] == 1024
    # length matches min(seq_len, max_len)
    df = pd.read_csv(tiny_csv)
    expected_len = min(len(str(df.iloc[0]["sequence"])), 64)
    assert item["embedding"].shape[0] == expected_len
