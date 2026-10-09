"""Build comparison dataset branches for CD-HIT threshold sensitivity.

This script intentionally writes branch-specific files under
``meta_acp/dataset_branches`` and does not overwrite the primary post-audit
CD-HIT 0.8 dataset in the project root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from rebuild_data_audit import (
        AEGIS_FILES,
        MANUSCRIPT_FILES,
        aggregate_modeling,
        parse_raw_records,
        seq_hash,
        stratified_split,
        validate_raw,
        write_fasta,
        write_source_summaries,
    )
except ImportError:  # pragma: no cover
    from .rebuild_data_audit import (
        AEGIS_FILES,
        MANUSCRIPT_FILES,
        aggregate_modeling,
        parse_raw_records,
        seq_hash,
        stratified_split,
        validate_raw,
        write_fasta,
        write_source_summaries,
    )


def write_csv_pair(df: pd.DataFrame, prefix: Path) -> None:
    df[["sequence", "binary_label"]].to_csv(prefix.with_suffix(".csv"), index=False)
    df.to_csv(prefix.with_name(prefix.name + "_source.csv"), index=False)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["binary_label"].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def run_cdhit_in_dir(out_dir: Path, identity: float, threads: int, image: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for CD-HIT but was not found on PATH.")
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{out_dir.resolve()}:/data",
        "-w",
        "/data",
        image,
        "cd-hit",
        "-i",
        "sequences.fasta",
        "-o",
        "cdhit_result_09",
        "-c",
        str(identity),
        "-n",
        "5",
        "-M",
        "4000",
        "-T",
        str(threads),
    ]
    subprocess.run(cmd, check=True)


def parse_branch_cdhit_representatives(clstr_path: Path, seq_ids: set[str]) -> list[str]:
    """Parse CD-HIT representatives, resolving truncated IDs by unique prefix."""
    representatives = []
    with clstr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "*" not in line:
                continue
            token = line.split(">")[1].split("...")[0]
            if token in seq_ids:
                representatives.append(token)
                continue
            matches = [seq_id for seq_id in seq_ids if seq_id.startswith(token)]
            if len(matches) != 1:
                raise KeyError(f"cannot resolve CD-HIT representative token {token}; matches={matches[:5]}")
            representatives.append(matches[0])
    return representatives


def parse_branch_cdhit_clustered_ids(clstr_path: Path, seq_ids: set[str]) -> set[str]:
    """Parse all sequence IDs present in a CD-HIT .clstr file."""
    clustered = set()
    with clstr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ">" not in line or line.startswith(">Cluster"):
                continue
            token = line.split(">")[1].split("...")[0]
            if token in seq_ids:
                clustered.add(token)
                continue
            matches = [seq_id for seq_id in seq_ids if seq_id.startswith(token)]
            if len(matches) != 1:
                raise KeyError(f"cannot resolve CD-HIT cluster token {token}; matches={matches[:5]}")
            clustered.add(matches[0])
    return clustered


def manuscript_record_table(raw: pd.DataFrame) -> pd.DataFrame:
    manuscript = raw[raw["source_file"].isin(MANUSCRIPT_FILES)].copy()
    rows = []
    for _, row in manuscript.iterrows():
        short_hash = seq_hash(row["sequence"])[:12]
        source_idx = str(row["source_id"]).split(":", 1)[-1]
        rows.append(
            {
                "seq_id": f"mscrnet_{row['source_file'].replace('.fasta', '').lower()}_{source_idx}_{short_hash}",
                "sequence": row["sequence"],
                "binary_label": int(row["binary_label"]),
                "length": int(row["length"]),
                "source_file": row["source_file"],
                "source_group": "ProtT5-MSCRNet",
                "all_source_files": row["source_file"],
                "all_source_groups": "ProtT5-MSCRNet",
                "all_raw_labels": str(row["raw_label"]),
                "all_source_ids": row["source_id"],
                "n_raw_records": 1,
                "split_group": short_hash,
            }
        )
    return pd.DataFrame(rows)


def write_overlap_audit(df: pd.DataFrame, out_dir: Path) -> dict:
    rows = []
    detail = []
    source_sets = {name: set(df.loc[df["source_file"] == name, "sequence"]) for name in MANUSCRIPT_FILES}
    for left in MANUSCRIPT_FILES:
        for right in MANUSCRIPT_FILES:
            if left >= right:
                continue
            overlap = sorted(source_sets[left] & source_sets[right])
            rows.append({"left_source": left, "right_source": right, "exact_overlap_sequences": len(overlap)})
            for seq in overlap:
                labels = sorted(df.loc[df["sequence"] == seq, "binary_label"].astype(int).unique())
                detail.append(
                    {
                        "left_source": left,
                        "right_source": right,
                        "sequence": seq,
                        "binary_labels": ";".join(map(str, labels)),
                        "length": len(seq),
                    }
                )
    summary = pd.DataFrame(rows)
    details = pd.DataFrame(detail)
    summary.to_csv(out_dir / "exact_overlap_summary.csv", index=False)
    details.to_csv(out_dir / "exact_overlap_sequences.csv", index=False)
    return {f"{r.left_source}__{r.right_source}": int(r.exact_overlap_sequences) for r in summary.itertuples()}


def build_prott5_mscrnet_branch(raw: pd.DataFrame, out_root: Path, seed: int) -> dict:
    out_dir = out_root / "benchmark_prott5_mscrnet_cdhit09"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = manuscript_record_table(raw)
    df.to_csv(out_dir / "raw_records_prott5_mscrnet.csv", index=False)
    overlaps = write_overlap_audit(df, out_dir)

    source_counts = (
        df.groupby(["source_file", "binary_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non_acp", 1: "acp"})
        .reset_index()
    )
    for col in ["non_acp", "acp"]:
        if col not in source_counts.columns:
            source_counts[col] = 0
    source_counts["total"] = source_counts["non_acp"] + source_counts["acp"]
    source_counts.to_csv(out_dir / "source_summary.csv", index=False)

    train_full = df[df["source_file"] == "Train.fasta"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test1 = df[df["source_file"] == "Test1.fasta"].reset_index(drop=True)
    test2 = df[df["source_file"] == "Test2.fasta"].reset_index(drop=True)
    train, validation = train_test_split(
        train_full,
        test_size=0.2,
        stratify=train_full["binary_label"],
        random_state=seed,
    )
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation = validation.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    write_csv_pair(train_full, out_dir / "train_full")
    write_csv_pair(train, out_dir / "train")
    write_csv_pair(validation, out_dir / "validation")
    write_csv_pair(test1, out_dir / "test1")
    write_csv_pair(test2, out_dir / "test2")
    write_csv_pair(test1, out_dir / "test")

    manifest = {
        "branch": "benchmark_prott5_mscrnet_cdhit09",
        "purpose": "Literature-compatible ProtT5-MSCRNet Train/Test1/Test2 benchmark.",
        "cdhit_identity": 0.9,
        "cdhit_handling": (
            "The official Train/Test1/Test2 partitions are preserved. The source article reports "
            "CD-HIT 0.9 preprocessing; this branch does not re-cluster and remove official test rows."
        ),
        "train_full_rows": int(len(train_full)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test1_rows": int(len(test1)),
        "test2_rows": int(len(test2)),
        "train_full_label_counts": label_counts(train_full),
        "train_label_counts": label_counts(train),
        "validation_label_counts": label_counts(validation),
        "test1_label_counts": label_counts(test1),
        "test2_label_counts": label_counts(test2),
        "exact_overlap_counts": overlaps,
        "default_test_alias": "test.csv and test_source.csv point to Test1.fasta for script compatibility.",
    }
    (out_dir / "branch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def aegis_record_table(raw: pd.DataFrame) -> pd.DataFrame:
    aegis = raw[raw["source_file"].isin(AEGIS_FILES)].copy()
    rows = []
    for _, row in aegis.iterrows():
        short_hash = seq_hash(row["sequence"])[:12]
        source_idx = str(row["source_id"]).split(":", 1)[-1]
        rows.append(
            {
                "seq_id": f"aegis_{row['source_file'].replace('.fasta', '').lower()}_{source_idx}_{short_hash}",
                "sequence": row["sequence"],
                "binary_label": int(row["binary_label"]),
                "length": int(row["length"]),
                "source_file": row["source_file"],
                "source_group": "Aegis",
                "all_source_files": row["source_file"],
                "all_source_groups": "Aegis",
                "all_raw_labels": str(row["raw_label"]),
                "all_source_ids": row["source_id"],
                "n_raw_records": 1,
                "split_group": short_hash,
            }
        )
    return pd.DataFrame(rows)


def build_aegis_branch(raw: pd.DataFrame, out_root: Path, seed: int) -> dict:
    out_dir = out_root / "benchmark_aegis_cdhit08"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = aegis_record_table(raw)
    df.to_csv(out_dir / "raw_records_aegis.csv", index=False)

    train_full = df[df["source_file"] == "training.fasta"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = df[df["source_file"] == "testing.fasta"].reset_index(drop=True)
    train, validation = train_test_split(
        train_full,
        test_size=0.2,
        stratify=train_full["binary_label"],
        random_state=seed,
    )
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation = validation.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    write_csv_pair(train_full, out_dir / "train_full")
    write_csv_pair(train, out_dir / "train")
    write_csv_pair(validation, out_dir / "validation")
    write_csv_pair(test, out_dir / "test")

    overlap = sorted(set(train_full["sequence"]) & set(test["sequence"]))
    pd.DataFrame({"sequence": overlap, "length": [len(seq) for seq in overlap]}).to_csv(
        out_dir / "training_testing_exact_overlap.csv", index=False
    )

    source_counts = (
        df.groupby(["source_file", "binary_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non_acp", 1: "acp"})
        .reset_index()
    )
    for col in ["non_acp", "acp"]:
        if col not in source_counts.columns:
            source_counts[col] = 0
    source_counts["total"] = source_counts["non_acp"] + source_counts["acp"]
    source_counts.to_csv(out_dir / "source_summary.csv", index=False)

    manifest = {
        "branch": "benchmark_aegis_cdhit08",
        "purpose": "Aegis-compatible comparison branch using original training.fasta/testing.fasta.",
        "cdhit_identity": 0.8,
        "cdhit_handling": (
            "The Aegis files are already literature-processed with CD-HIT 0.8. The official testing.fasta "
            "is preserved as the independent test set; training.fasta is split 80:20 into train/validation."
        ),
        "train_full_rows": int(len(train_full)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_full_label_counts": label_counts(train_full),
        "train_label_counts": label_counts(train),
        "validation_label_counts": label_counts(validation),
        "test_label_counts": label_counts(test),
        "training_testing_exact_overlap": int(len(overlap)),
    }
    (out_dir / "branch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_integrated_cdhit09_branch(
    raw: pd.DataFrame,
    out_root: Path,
    seed: int,
    threads: int,
    skip_cdhit: bool,
    cdhit_image: str,
) -> dict:
    out_dir = out_root / "integrated_cdhit09"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = validate_raw(raw)
    modeling, provenance, conflicts = aggregate_modeling(raw)
    modeling.to_csv(out_dir / "merged_all_pre_cdhit.csv", index=False)
    provenance.to_csv(out_dir / "provenance_pre_cdhit_long.csv", index=False)
    conflicts.to_csv(out_dir / "label_conflicts.csv", index=False)
    write_source_summaries(provenance, out_dir, "pre_cdhit")
    write_fasta(modeling, out_dir / "sequences.fasta")

    if skip_cdhit:
        representatives = modeling["seq_id"].tolist()
        clustered_ids = set(modeling["seq_id"])
        cdhit_mode = "skipped"
    else:
        run_cdhit_in_dir(out_dir, identity=0.9, threads=threads, image=cdhit_image)
        seq_ids = set(modeling["seq_id"])
        representatives = parse_branch_cdhit_representatives(out_dir / "cdhit_result_09.clstr", seq_ids)
        clustered_ids = parse_branch_cdhit_clustered_ids(out_dir / "cdhit_result_09.clstr", seq_ids)
        cdhit_mode = "cd-hit"

    rep_set = set(representatives)
    if len(rep_set) != len(representatives):
        raise AssertionError("CD-HIT representative IDs are not unique")
    cdhit = modeling[modeling["seq_id"].isin(rep_set)].copy()
    removed_by_cdhit = modeling[modeling["seq_id"].isin(clustered_ids - rep_set)].copy()
    not_clustered_by_cdhit = modeling[~modeling["seq_id"].isin(clustered_ids)].copy()
    order = {seq_id: i for i, seq_id in enumerate(representatives)}
    cdhit["cdhit_rank"] = cdhit["seq_id"].map(order)
    cdhit = cdhit.sort_values("cdhit_rank").drop(columns=["cdhit_rank"]).reset_index(drop=True)
    cdhit_provenance = provenance[provenance["seq_id"].isin(rep_set)].copy().reset_index(drop=True)
    cdhit.to_csv(out_dir / "merged_all_cdhit09.csv", index=False)
    removed_by_cdhit.to_csv(out_dir / "removed_by_cdhit09.csv", index=False)
    not_clustered_by_cdhit.to_csv(out_dir / "not_clustered_by_cdhit09.csv", index=False)
    cdhit_provenance.to_csv(out_dir / "provenance_cdhit09_long.csv", index=False)
    write_fasta(cdhit, out_dir / "cdhit_result_09_representatives.fasta")
    write_source_summaries(cdhit_provenance, out_dir, "cdhit09")

    train, validation, test = stratified_split(cdhit, seed=seed)
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        write_csv_pair(split, out_dir / name)

    manifest = {
        "branch": "integrated_cdhit09",
        "purpose": "Integrated Aegis plus ProtT5-MSCRNet sensitivity branch with global CD-HIT 0.9.",
        "cdhit_identity": 0.9,
        "cdhit_mode": cdhit_mode,
        "cdhit_image": None if skip_cdhit else cdhit_image,
        "raw_records": int(audit["raw_records"]),
        "pre_cdhit_unique_nonconflicting": int(len(modeling)),
        "cdhit09_clustered_sequences": int(len(clustered_ids)),
        "cdhit09_sequences": int(len(cdhit)),
        "removed_by_cdhit09_redundancy": int(len(removed_by_cdhit)),
        "not_clustered_by_cdhit09": int(len(not_clustered_by_cdhit)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "pre_cdhit_label_counts": label_counts(modeling),
        "cdhit09_label_counts": label_counts(cdhit),
        "train_label_counts": label_counts(train),
        "validation_label_counts": label_counts(validation),
        "test_label_counts": label_counts(test),
        "removed_by_cdhit09_label_counts": label_counts(removed_by_cdhit) if len(removed_by_cdhit) else {},
        "not_clustered_by_cdhit09_label_counts": label_counts(not_clustered_by_cdhit) if len(not_clustered_by_cdhit) else {},
    }
    (out_dir / "branch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ProtT5-MSCRNet-compatible and integrated CD-HIT 0.9 data branches.")
    parser.add_argument("--raw-dir", default="../..")
    parser.add_argument("--out-root", default="meta_acp/dataset_branches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cdhit-image", default="biocontainers/cd-hit:v4.8.1-2-deb_cv1")
    parser.add_argument("--skip-cdhit", action="store_true", help="Only for debugging; integrated branch will keep pre-CD-HIT records.")
    args = parser.parse_args()

    raw = parse_raw_records(Path(args.raw_dir).resolve())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "benchmark_aegis_cdhit08": build_aegis_branch(raw, out_root, args.seed),
        "benchmark_prott5_mscrnet_cdhit09": build_prott5_mscrnet_branch(raw, out_root, args.seed),
        "integrated_cdhit09": build_integrated_cdhit09_branch(
            raw,
            out_root,
            args.seed,
            args.threads,
            args.skip_cdhit,
            args.cdhit_image,
        ),
    }
    (out_root / "branch_summary.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
