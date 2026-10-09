"""Prepare external peptide/protein binary tasks for PLM-SourceACP transfer.

This script standardizes public datasets such as CPPsite, AntiAngioPred,
TumorHoPe, or PepBenchmark into the same CSV contract used by the ACP pipeline:

sequence,binary_label for modeling splits, plus provenance-rich companion
tables for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha_id(seq: str) -> str:
    return "ext_" + hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def normalize_sequence(seq: str, keep_noncanonical: bool = False) -> str:
    seq = "".join(str(seq).upper().split())
    if keep_noncanonical:
        return seq
    return "".join(ch for ch in seq if ch in CANONICAL_AA)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def load_labeled_csv(path: Path, task_name: str, source_name: str, keep_noncanonical: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    seq_col = "sequence" if "sequence" in df.columns else "seq" if "seq" in df.columns else None
    label_col = "binary_label" if "binary_label" in df.columns else "label" if "label" in df.columns else None
    if seq_col is None or label_col is None:
        raise ValueError(f"{path} must contain sequence/seq and binary_label/label columns")
    out = df[[seq_col, label_col]].rename(columns={seq_col: "sequence", label_col: "binary_label"}).copy()
    out["sequence"] = out["sequence"].map(lambda x: normalize_sequence(x, keep_noncanonical))
    out["binary_label"] = out["binary_label"].astype(int)
    out["task_name"] = task_name
    out["source_file"] = source_name or path.name
    out["source_group"] = task_name
    out["source_id"] = [f"{path.name}:{i+1}" for i in range(len(out))]
    out["raw_label"] = out["binary_label"].astype(str)
    return out


def load_fasta_pair(
    positive: Path | None,
    negative: Path | None,
    task_name: str,
    keep_noncanonical: bool,
) -> pd.DataFrame:
    rows = []
    for path, label, raw_label in [(positive, 1, "positive"), (negative, 0, "negative")]:
        if path is None:
            continue
        for i, (header, seq) in enumerate(read_fasta(path), start=1):
            rows.append(
                {
                    "task_name": task_name,
                    "source_file": path.name,
                    "source_group": task_name,
                    "source_id": f"{path.name}:{i}",
                    "raw_label": raw_label,
                    "header": header,
                    "sequence": normalize_sequence(seq, keep_noncanonical),
                    "binary_label": label,
                }
            )
    if not rows:
        raise ValueError("provide --input-csv or at least one of --positive-fasta/--negative-fasta")
    return pd.DataFrame(rows)


def audit_and_collapse(raw: pd.DataFrame, min_len: int, max_len: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["length"] = raw["sequence"].astype(str).str.len()
    raw = raw[(raw["length"] >= min_len) & (raw["length"] <= max_len)].copy()
    raw = raw[raw["sequence"].astype(str).str.len() > 0].copy()
    conflicts = raw.groupby("sequence")["binary_label"].nunique()
    conflict_sequences = set(conflicts[conflicts > 1].index)
    conflict_df = raw[raw["sequence"].isin(conflict_sequences)].copy()
    raw = raw[~raw["sequence"].isin(conflict_sequences)].copy()

    records = []
    provenance = []
    for seq, part in raw.groupby("sequence", sort=True):
        label = int(part["binary_label"].iloc[0])
        seq_id = sha_id(seq)
        source_files = sorted(part["source_file"].astype(str).unique())
        source_ids = sorted(part["source_id"].astype(str).unique())
        raw_labels = sorted(part["raw_label"].astype(str).unique())
        records.append(
            {
                "seq_id": seq_id,
                "sequence": seq,
                "binary_label": label,
                "length": len(seq),
                "source_file": source_files[0],
                "source_group": str(part["source_group"].iloc[0]),
                "all_source_files": ";".join(source_files),
                "all_source_ids": ";".join(source_ids),
                "all_raw_labels": ";".join(raw_labels),
                "n_raw_records": int(len(part)),
            }
        )
        for _, row in part.iterrows():
            provenance.append(
                {
                    "seq_id": seq_id,
                    "sequence": seq,
                    "binary_label": label,
                    "source_file": row["source_file"],
                    "source_group": row["source_group"],
                    "source_id": row["source_id"],
                    "raw_label": row["raw_label"],
                }
            )
    return pd.DataFrame(records), pd.DataFrame(provenance), conflict_df


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            fh.write(f">{row['seq_id']}|label={int(row['binary_label'])}\n{row['sequence']}\n")


def parse_cdhit_clstr(path: Path) -> list[str]:
    reps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().endswith("*"):
            token = line.split(">", 1)[1].split("...", 1)[0]
            reps.append(token)
    return reps


def apply_cdhit(df: pd.DataFrame, out_dir: Path, identity: float, threads: int, skip_cdhit: bool) -> tuple[pd.DataFrame, dict]:
    fasta = out_dir / "sequences.fasta"
    prefix = out_dir / "cdhit_result_08"
    write_fasta(df, fasta)
    status = {"identity": identity, "mode": "exact_unique_only", "representatives": int(len(df))}
    if skip_cdhit:
        status["note"] = "CD-HIT skipped by user; exact duplicate collapse only."
        return df.copy(), status
    if shutil.which("docker") is None:
        status["note"] = "Docker not found; CD-HIT not executed. Exact duplicate collapse only."
        return df.copy(), status

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{out_dir.resolve()}:/work",
        "-w",
        "/work",
        "quay.io/biocontainers/cd-hit:v4.8.1-2-deb_cv1",
        "cd-hit",
        "-i",
        fasta.name,
        "-o",
        prefix.name,
        "-c",
        str(identity),
        "-n",
        "5",
        "-T",
        str(threads),
        "-M",
        "0",
    ]
    subprocess.run(cmd, check=True)
    reps = parse_cdhit_clstr(prefix.with_suffix(".clstr"))
    rep_set = set(reps)
    reduced = df[df["seq_id"].isin(rep_set)].copy()
    order = {seq_id: i for i, seq_id in enumerate(reps)}
    reduced["cdhit_rank"] = reduced["seq_id"].map(order)
    reduced = reduced.sort_values("cdhit_rank").drop(columns=["cdhit_rank"]).reset_index(drop=True)
    status.update({"mode": "cd-hit", "representatives": int(len(reduced)), "note": "CD-HIT completed."})
    return reduced, status


def split_dataset(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df["binary_label"].nunique() < 2:
        raise ValueError("binary split requires both classes")
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df["binary_label"], random_state=seed)
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_fraction, stratify=train_val["binary_label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external binary peptide/protein task splits.")
    parser.add_argument("--task-name", required=True, help="e.g. cpp, antiangiogenic, tumor_homing")
    parser.add_argument("--input-csv", default=None, help="CSV with sequence/seq and binary_label/label columns")
    parser.add_argument("--positive-fasta", default=None)
    parser.add_argument("--negative-fasta", default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-cdhit", action="store_true")
    parser.add_argument("--keep-noncanonical", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--negative-construction", default="", help="Free-text audit note for negative-set origin.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.input_csv:
        raw = load_labeled_csv(Path(args.input_csv), args.task_name, args.source_name or Path(args.input_csv).name, args.keep_noncanonical)
    else:
        raw = load_fasta_pair(
            Path(args.positive_fasta) if args.positive_fasta else None,
            Path(args.negative_fasta) if args.negative_fasta else None,
            args.task_name,
            args.keep_noncanonical,
        )

    modeling, provenance, conflicts = audit_and_collapse(raw, args.min_len, args.max_len)
    reduced, cdhit_status = apply_cdhit(modeling, out_dir, args.identity, args.threads, args.skip_cdhit)
    rep_ids = set(reduced["seq_id"])
    prov_reduced = provenance[provenance["seq_id"].isin(rep_ids)].copy()
    train, val, test = split_dataset(reduced, args.seed, args.train_ratio, args.val_ratio)

    raw.to_csv(out_dir / "raw_records.csv", index=False)
    modeling.to_csv(out_dir / "merged_all_pre_cdhit.csv", index=False)
    provenance.to_csv(out_dir / "provenance_pre_cdhit_long.csv", index=False)
    conflicts.to_csv(out_dir / "label_conflicts.csv", index=False)
    reduced.to_csv(out_dir / "merged_all_cdhit08.csv", index=False)
    prov_reduced.to_csv(out_dir / "provenance_cdhit08_long.csv", index=False)
    for name, split in [("train", train), ("validation", val), ("test", test)]:
        split.to_csv(out_dir / f"{name}_source.csv", index=False)
        split[["sequence", "binary_label"]].to_csv(out_dir / f"{name}.csv", index=False)

    summary = {
        "task_name": args.task_name,
        "raw_records": int(len(raw)),
        "pre_cdhit_unique_nonconflicting": int(len(modeling)),
        "cdhit08_sequences": int(len(reduced)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "label_conflict_rows": int(len(conflicts)),
        "class_counts_cdhit08": {str(k): int(v) for k, v in reduced["binary_label"].value_counts().sort_index().items()},
        "negative_construction": args.negative_construction,
        "cdhit": cdhit_status,
    }
    (out_dir / "external_task_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
