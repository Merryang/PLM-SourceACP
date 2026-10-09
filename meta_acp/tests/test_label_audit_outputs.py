"""Regression tests for audited label/provenance rebuild outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def require_file(name: str) -> Path:
    path_map = {
        'raw_records_label_audit.csv': 'data/audit/raw_records_label_audit.csv',
        'label_conflicts.csv': 'data/audit/label_conflicts.csv',
        'merged_all.csv': 'data/processed/merged_all.csv',
        'merged_all_cdhit08.csv': 'data/processed/merged_all_cdhit08.csv',
        'provenance_cdhit08_long.csv': 'data/processed/provenance_cdhit08_long.csv',
    }
    path = ROOT / path_map.get(name, name)
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return path


def test_label_audit_raw_counts_and_conflict_sequence():
    raw = pd.read_csv(require_file("raw_records_label_audit.csv"))
    assert len(raw) == 3418
    expected = {
        "training.fasta": {0: 120, 1: 443},
        "testing.fasta": {0: 30, 1: 108},
        "Train.fasta": {0: 1479, 1: 487},
        "Test1.fasta": {0: 360, 1: 135},
        "Test2.fasta": {0: 157, 1: 99},
    }
    counts = raw.groupby(["source_file", "binary_label"]).size().unstack(fill_value=0)
    for source_file, label_counts in expected.items():
        for label, n in label_counts.items():
            assert int(counts.loc[source_file, label]) == n

    conflicts = pd.read_csv(require_file("label_conflicts.csv"))
    assert set(conflicts["sequence"]) == {"ALWKTIIKGAGKMIGSLAKNLLGSQAQPES"}
    assert conflicts["binary_label"].nunique() == 2


def test_modeling_and_cdhit_tables_have_unique_nonconflicting_sequences():
    for name in ["merged_all.csv", "merged_all_cdhit08.csv"]:
        df = pd.read_csv(require_file(name))
        assert df["sequence"].is_unique
        assert df.groupby("sequence")["binary_label"].nunique().max() == 1
        assert "seq_id" in df.columns
        assert "all_source_files" in df.columns
    cdhit = pd.read_csv(require_file("merged_all_cdhit08.csv"))
    assert len(cdhit) == 2882


def test_splits_are_disjoint_and_cover_cdhit_sequences():
    cdhit = pd.read_csv(require_file("merged_all_cdhit08.csv"))
    splits = {name: pd.read_csv(require_file(f"{name}.csv")) for name in ["train", "validation", "test"]}
    seq_sets = {name: set(df["sequence"]) for name, df in splits.items()}
    assert not (seq_sets["train"] & seq_sets["validation"])
    assert not (seq_sets["train"] & seq_sets["test"])
    assert not (seq_sets["validation"] & seq_sets["test"])
    assert set.union(*seq_sets.values()) == set(cdhit["sequence"])
    assert [len(splits["train"]), len(splits["validation"]), len(splits["test"])] == [1728, 577, 577]


def test_provenance_restores_aegis_negative_membership_after_cdhit():
    provenance = pd.read_csv(require_file("provenance_cdhit08_long.csv"))
    summary = provenance.groupby(["source_file", "binary_label"]).size().unstack(fill_value=0)
    assert int(summary.loc["training.fasta", 0]) == 120
    assert int(summary.loc["testing.fasta", 0]) == 29
    assert int(summary.loc["training.fasta", 1]) == 430
    assert int(summary.loc["testing.fasta", 1]) == 99
    assert provenance.drop_duplicates(["sequence", "source_file"]).shape[0] <= len(provenance)
