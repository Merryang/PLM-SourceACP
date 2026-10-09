"""Prepare official AntiAngioPred and PepBenchmark CPP splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_contract_columns(df: pd.DataFrame, task: str, split: str, source: str) -> pd.DataFrame:
    out = df[["sequence", "binary_label"]].copy().reset_index(drop=True)
    out["seq_id"] = [
        "ext_" + hashlib.sha1(f"{task}|{sequence}".encode("utf-8")).hexdigest()[:16]
        for sequence in out["sequence"]
    ]
    out["length"] = out["sequence"].str.len()
    out["source_file"] = source
    out["source_group"] = task
    out["official_split"] = split
    out["source_id"] = [f"{source}:{split}:{i + 1}" for i in range(len(out))]
    return out[
        [
            "seq_id",
            "sequence",
            "binary_label",
            "length",
            "source_file",
            "source_group",
            "official_split",
            "source_id",
        ]
    ]


def write_split(out_dir: Path, name: str, df: pd.DataFrame) -> None:
    df.to_csv(out_dir / f"{name}_source.csv", index=False)
    df[["sequence", "binary_label"]].to_csv(out_dir / f"{name}.csv", index=False)


def stratified_train_validation(
    df: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for _, part in df.groupby("binary_label"):
        indices = part.index.to_numpy().copy()
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])
    return (
        df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True),
        df.loc[validation_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def extract_antiangiopred_pdf(path: Path) -> tuple[list[str], list[str]]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    positive: list[str] = []
    negative: list[str] = []
    mode = None
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", "", raw_line).upper()
        if not line:
            continue
        if line.startswith("POSITIVE"):
            mode = "positive"
            continue
        if line.startswith("NEGATIVE"):
            mode = "negative"
            continue
        if not re.fullmatch(r"[A-Z]+", line):
            continue
        if mode == "positive":
            positive.append(line)
        elif mode == "negative":
            negative.append(line)
    return positive, negative


def prepare_antiangiopred(source_root: Path, out_dir: Path, seed: int) -> dict:
    cv_pdf = source_root / "antiangiopred_80percent_cross_validation.txt.pdf"
    test_pdf = source_root / "antiangiopred_20percent.txt.pdf"
    cv_pos, cv_neg = extract_antiangiopred_pdf(cv_pdf)
    test_pos, test_neg = extract_antiangiopred_pdf(test_pdf)
    if (len(cv_pos), len(cv_neg), len(test_pos), len(test_neg)) != (107, 107, 28, 28):
        raise ValueError("AntiAngioPred PDF extraction counts do not match the official archive")

    rows = []
    excluded = []
    for split, label, sequences in [
        ("development", 1, cv_pos),
        ("development", 0, cv_neg),
        ("test", 1, test_pos),
        ("test", 0, test_neg),
    ]:
        for sequence in sequences:
            noncanonical = sorted(set(sequence) - CANONICAL)
            record = {"sequence": sequence, "binary_label": label, "official_split": split}
            if noncanonical:
                record["noncanonical_residues"] = "".join(noncanonical)
                excluded.append(record)
            else:
                rows.append(record)
    data = pd.DataFrame(rows)
    if data.duplicated("sequence").any():
        raise ValueError("duplicate AntiAngioPred sequences after extraction")

    development = data[data["official_split"] == "development"][["sequence", "binary_label"]]
    test = data[data["official_split"] == "test"][["sequence", "binary_label"]]
    train, validation = stratified_train_validation(development, 0.2, seed)
    train = add_contract_columns(train, "antiangiogenic", "train", "AntiAngioPred_official_80percent")
    validation = add_contract_columns(
        validation, "antiangiogenic", "validation", "AntiAngioPred_official_80percent"
    )
    test = add_contract_columns(test, "antiangiogenic", "test", "AntiAngioPred_official_20percent")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(out_dir, "train", train)
    write_split(out_dir, "validation", validation)
    write_split(out_dir, "test", test)
    pd.concat([train, validation, test], ignore_index=True).to_csv(out_dir / "all_source.csv", index=False)
    pd.DataFrame(excluded).to_csv(out_dir / "excluded_noncanonical.csv", index=False)

    summary = {
        "task_name": "antiangiogenic",
        "protocol": "Official AntiAngioPred 80% development / 20% independent split; development set split 80/20 for train/validation.",
        "official_counts_before_canonical_filter": {
            "development_positive": len(cv_pos),
            "development_negative": len(cv_neg),
            "test_positive": len(test_pos),
            "test_negative": len(test_neg),
        },
        "excluded_noncanonical": len(excluded),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "class_counts": {
            name: {str(k): int(v) for k, v in frame["binary_label"].value_counts().sort_index().items()}
            for name, frame in [("train", train), ("validation", validation), ("test", test)]
        },
        "exact_overlap": {
            "train_validation": len(set(train["sequence"]) & set(validation["sequence"])),
            "train_test": len(set(train["sequence"]) & set(test["sequence"])),
            "validation_test": len(set(validation["sequence"]) & set(test["sequence"])),
        },
        "sources": {
            "repository": "https://github.com/sachini-tech/AntiAngioPred/tree/V1",
            "archive": "https://doi.org/10.5281/zenodo.20080041",
            "paper": "https://doi.org/10.1371/journal.pone.0136990",
            "development_pdf_sha256": sha256(cv_pdf),
            "test_pdf_sha256": sha256(test_pdf),
        },
    }
    (out_dir / "external_task_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def prepare_cpp(source_root: Path, out_dir: Path) -> dict:
    all_path = source_root / "all.csv"
    split_path = source_root / "cdhit_split.json"
    raw = pd.read_csv(all_path).rename(columns={"label": "binary_label"})
    if raw.duplicated("sequence").any() or raw.groupby("sequence")["binary_label"].nunique().gt(1).any():
        raise ValueError("PepBenchmark CPP contains duplicate/conflicting sequences")
    if raw["sequence"].map(lambda s: bool(set(s) - CANONICAL)).any():
        raise ValueError("PepBenchmark CPP contains noncanonical residues")
    split = json.loads(split_path.read_text(encoding="utf-8"))["seed_0"]

    frames = {}
    for target_name, source_name in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        frame = raw.iloc[split[source_name]][["sequence", "binary_label"]]
        frames[target_name] = add_contract_columns(
            frame, "cell_penetrating_peptide", target_name, "PepBenchmark_CPP_CDHIT_seed0"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        write_split(out_dir, name, frame)
    pd.concat(list(frames.values()), ignore_index=True).to_csv(out_dir / "all_source.csv", index=False)

    summary = {
        "task_name": "cell_penetrating_peptide",
        "protocol": "PepBenchmark CPP cdhit_split.json seed_0, preserving official train/valid/test indices.",
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "test_rows": len(frames["test"]),
        "class_counts": {
            name: {str(k): int(v) for k, v in frame["binary_label"].value_counts().sort_index().items()}
            for name, frame in frames.items()
        },
        "length_range": {
            name: [int(frame["length"].min()), int(frame["length"].max())]
            for name, frame in frames.items()
        },
        "exact_overlap": {
            "train_validation": len(set(frames["train"]["sequence"]) & set(frames["validation"]["sequence"])),
            "train_test": len(set(frames["train"]["sequence"]) & set(frames["test"]["sequence"])),
            "validation_test": len(set(frames["validation"]["sequence"]) & set(frames["test"]["sequence"])),
        },
        "sources": {
            "repository": "https://github.com/ZGCI-AI4S-Pep/PepBenchmark",
            "dataset": "https://huggingface.co/datasets/jiahuizhang/PepBenchData/tree/main/PepBenchData-50/cpp",
            "all_csv_sha256": sha256(all_path),
            "cdhit_split_sha256": sha256(split_path),
        },
    }
    (out_dir / "external_task_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="meta_acp/external_sources")
    parser.add_argument("--out-root", default="meta_acp/external_tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)

    anti = prepare_antiangiopred(
        source_root / "antiangiopred" / "dataset" / "AntiAngioPred",
        out_root / "antiangiopred_official",
        args.seed,
    )
    cpp = prepare_cpp(
        source_root / "pepbenchmark_cpp",
        out_root / "pepbenchmark_cpp_cdhit_seed0",
    )
    print(json.dumps({"antiangiopred": anti, "cpp": cpp}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
