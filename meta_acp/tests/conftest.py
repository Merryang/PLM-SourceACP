"""Shared pytest fixtures for the meta_acp test suite."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# Make the meta_acp package importable when running pytest from anywhere.
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed every RNG before each test for reproducibility."""
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """Build a minimal Doubao-schema CSV with controlled class imbalance.

    20 sequences total, 8 positives + 12 negatives, varied lengths.
    Schema matches: source_file, source_id, raw_label, binary_label,
                    sequence, length, split_group.
    """
    rows = []
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    rng = random.Random(7)

    def _seq(L: int) -> str:
        return "".join(rng.choices(aas, k=L))

    for i in range(8):
        s = _seq(rng.randint(8, 25))
        rows.append(
            ("Train.fasta", f"POS{i}", "1", 1, s, len(s), f"g{i}")
        )
    for i in range(12):
        s = _seq(rng.randint(8, 40))
        rows.append(
            ("Train.fasta", f"NEG{i}", "0", 0, s, len(s), f"g{i+100}")
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "source_file",
            "source_id",
            "raw_label",
            "binary_label",
            "sequence",
            "length",
            "split_group",
        ],
    )
    out = tmp_path / "tiny.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture
def doubao_csv_paths() -> dict:
    """Real Doubao CSV paths; tests can xfail if files are absent."""
    base = Path(__file__).resolve().parents[2]
    return {
        "train": base / "train.csv",
        "val": base / "validation.csv",
        "test": base / "test.csv",
    }


@pytest.fixture
def tiny_embedding_cache(tiny_csv: Path, tmp_path: Path) -> Path:
    """Build a fake ProtT5 embedding cache for the tiny CSV (1024-dim per residue)."""
    import hashlib

    df = pd.read_csv(tiny_csv)
    cache = {}
    for s in df["sequence"].astype(str).tolist():
        L = len(s)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        # deterministic pseudo-embeddings
        g = torch.Generator().manual_seed(int(h[:8], 16) % (2**31))
        cache[h] = torch.randn(L, 1024, generator=g)
    cache_path = tmp_path / "embeddings.pt"
    torch.save(cache, cache_path)
    return cache_path
