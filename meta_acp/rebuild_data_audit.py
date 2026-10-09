"""Rebuild ACP data with label audit and multi-source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


AEGIS_FILES = ["training.fasta", "testing.fasta"]
MANUSCRIPT_FILES = ["Train.fasta", "Test1.fasta", "Test2.fasta"]
SOURCE_GROUPS = {
    "training.fasta": "Aegis",
    "testing.fasta": "Aegis",
    "Train.fasta": "Manuscript1856815",
    "Test1.fasta": "Manuscript1856815",
    "Test2.fasta": "Manuscript1856815",
}
EXPECTED_SOURCE_COUNTS = {
    "training.fasta": {0: 120, 1: 443},
    "testing.fasta": {0: 30, 1: 108},
    "Train.fasta": {0: 1479, 1: 487},
    "Test1.fasta": {0: 360, 1: 135},
    "Test2.fasta": {0: 157, 1: 99},
}
CONFLICT_SEQUENCE = "ALWKTIIKGAGKMIGSLAKNLLGSQAQPES"


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def normalize_sequence(seq: str) -> str:
    return str(seq).strip().upper()


def read_aegis_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def read_manuscript_csvlike(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"{path} line {line_no} is not label,sequence format: {line}")
            records.append((parts[0].strip(), parts[1].strip()))
    return records


def aegis_label(header: str) -> int:
    token = header.split("|", 1)[0].upper()
    if token.startswith("POS") or token.startswith("OTH"):
        return 1
    if token.startswith("NEG"):
        return 0
    raise ValueError(f"cannot parse Aegis label from header: {header}")


def manuscript_label(label: str) -> int:
    if label not in {"0", "1"}:
        raise ValueError(f"cannot parse manuscript label: {label}")
    return int(label)


def parse_raw_records(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for source_file in AEGIS_FILES:
        for idx, (header, seq) in enumerate(read_aegis_fasta(raw_dir / source_file), start=1):
            rows.append(
                {
                    "source_file": source_file,
                    "source_group": SOURCE_GROUPS[source_file],
                    "source_id": f"{source_file}:{idx}",
                    "raw_label": header.split("|", 1)[0],
                    "header": header,
                    "sequence": normalize_sequence(seq),
                    "binary_label": aegis_label(header),
                }
            )
    for source_file in MANUSCRIPT_FILES:
        for idx, (label, seq) in enumerate(read_manuscript_csvlike(raw_dir / source_file), start=1):
            rows.append(
                {
                    "source_file": source_file,
                    "source_group": SOURCE_GROUPS[source_file],
                    "source_id": f"{source_file}:{idx}",
                    "raw_label": label,
                    "header": label,
                    "sequence": normalize_sequence(seq),
                    "binary_label": manuscript_label(label),
                }
            )
    df = pd.DataFrame(rows)
    df["length"] = df["sequence"].str.len()
    return df


def validate_raw(raw: pd.DataFrame) -> dict:
    audit: dict = {
        "raw_records": int(len(raw)),
        "unique_sequences": int(raw["sequence"].nunique()),
        "unparseable_labels": 0,
    }
    if len(raw) != 3418:
        raise AssertionError(f"expected 3418 raw records, got {len(raw)}")
    counts = raw.groupby(["source_file", "binary_label"]).size().unstack(fill_value=0)
    for source, expected in EXPECTED_SOURCE_COUNTS.items():
        for label, n in expected.items():
            got = int(counts.loc[source, label]) if source in counts.index and label in counts.columns else 0
            if got != n:
                raise AssertionError(f"{source} label {label}: expected {n}, got {got}")
    conflict_labels = raw.groupby("sequence")["binary_label"].nunique()
    conflicts = conflict_labels[conflict_labels > 1]
    audit["duplicate_sequence_groups"] = int((raw.groupby("sequence").size() > 1).sum())
    audit["duplicate_records_beyond_first"] = int(len(raw) - raw["sequence"].nunique())
    audit["conflicting_label_sequences"] = int(len(conflicts))
    if len(conflicts) != 1 or conflicts.index[0] != CONFLICT_SEQUENCE:
        raise AssertionError(f"expected one known conflict sequence, found {list(conflicts.index)}")
    return audit


def aggregate_modeling(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_nunique = raw.groupby("sequence")["binary_label"].nunique()
    conflict_sequences = set(label_nunique[label_nunique > 1].index)
    conflicts = raw[raw["sequence"].isin(conflict_sequences)].sort_values(["sequence", "source_file", "source_id"])
    clean = raw[~raw["sequence"].isin(conflict_sequences)].copy()
    records = []
    provenance = []
    for sequence, part in clean.groupby("sequence", sort=False):
        label_values = sorted(part["binary_label"].unique())
        if len(label_values) != 1:
            raise AssertionError(f"sequence has conflicting labels after filtering: {sequence}")
        seq_id = f"seq_{seq_hash(sequence)[:16]}"
        part = part.sort_values(["source_group", "source_file", "source_id"]).copy()
        source_files = sorted(part["source_file"].unique())
        source_groups = sorted(part["source_group"].unique())
        raw_labels = sorted(part["raw_label"].astype(str).unique())
        source_ids = sorted(part["source_id"].astype(str).unique())
        records.append(
            {
                "seq_id": seq_id,
                "sequence": sequence,
                "binary_label": int(label_values[0]),
                "length": len(sequence),
                "source_file": source_files[0],
                "source_group": source_groups[0],
                "all_source_files": ";".join(source_files),
                "all_source_groups": ";".join(source_groups),
                "all_raw_labels": ";".join(raw_labels),
                "all_source_ids": ";".join(source_ids),
                "n_raw_records": int(len(part)),
                "split_group": seq_hash(sequence)[:12],
            }
        )
        for _, row in part.iterrows():
            provenance.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "binary_label": int(row["binary_label"]),
                    "source_file": row["source_file"],
                    "source_group": row["source_group"],
                    "source_id": row["source_id"],
                    "raw_label": row["raw_label"],
                    "header": row["header"],
                    "length": len(sequence),
                }
            )
    modeling = pd.DataFrame(records).sort_values(["length", "seq_id"], ascending=[False, True]).reset_index(drop=True)
    provenance_df = pd.DataFrame(provenance).sort_values(["seq_id", "source_file", "source_id"]).reset_index(drop=True)
    if modeling.duplicated("sequence").any():
        raise AssertionError("modeling table contains duplicate sequences")
    if modeling.groupby("sequence")["binary_label"].nunique().max() != 1:
        raise AssertionError("modeling table contains conflicting labels")
    return modeling, provenance_df, conflicts


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            handle.write(f">{row['seq_id']}\n{row['sequence']}\n")


def read_fasta_map(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header = None
    seq: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = normalize_sequence("".join(seq))
                header = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if header is not None:
        records[header] = normalize_sequence("".join(seq))
    return records


def run_cdhit(input_fasta: Path, output_prefix: Path, identity: float = 0.8, threads: int = 2) -> None:
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:/data",
        "-w",
        "/data",
        "biocontainers/cd-hit:v4.8.1-2-deb_cv1",
        "cd-hit",
        "-i",
        input_fasta.name,
        "-o",
        output_prefix.name,
        "-c",
        str(identity),
        "-n",
        "5",
        "-M",
        "1000",
        "-T",
        str(threads),
    ]
    subprocess.run(docker_cmd, check=True)


def parse_cdhit_representatives(clstr_path: Path) -> list[str]:
    representatives = []
    with clstr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "*" not in line:
                continue
            token = line.split(">")[1].split("...")[0]
            representatives.append(token)
    return representatives


def reused_representatives_from_legacy(legacy_dir: Path, modeling: pd.DataFrame) -> list[str]:
    legacy_fasta = legacy_dir / "sequences.fasta"
    legacy_clstr = legacy_dir / "cdhit_result_08.clstr"
    if not legacy_fasta.exists() or not legacy_clstr.exists():
        raise FileNotFoundError(f"{legacy_dir} must contain sequences.fasta and cdhit_result_08.clstr")
    legacy_records = read_fasta_map(legacy_fasta)
    current_by_sequence = dict(zip(modeling["sequence"], modeling["seq_id"]))
    legacy_sequences = set(legacy_records.values())
    current_sequences = set(current_by_sequence)
    if legacy_sequences != current_sequences:
        missing_in_current = sorted(legacy_sequences - current_sequences)[:5]
        missing_in_legacy = sorted(current_sequences - legacy_sequences)[:5]
        raise AssertionError(
            "legacy CD-HIT input sequence set differs from rebuilt modeling set; "
            f"missing_in_current={missing_in_current}; missing_in_legacy={missing_in_legacy}"
        )
    legacy_rep_ids = parse_cdhit_representatives(legacy_clstr)
    representatives = []
    for legacy_id in legacy_rep_ids:
        if legacy_id not in legacy_records:
            raise KeyError(f"legacy representative {legacy_id} absent from {legacy_fasta}")
        representatives.append(current_by_sequence[legacy_records[legacy_id]])
    return representatives


def stratified_split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=seed)
    train, validation = train_test_split(train_val, test_size=0.25, stratify=train_val["binary_label"], random_state=seed)
    return (
        train.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        validation.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        test.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def backup_outputs(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        dest = backup_dir / path.name
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)


def write_source_summaries(provenance: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = provenance.groupby(["source_file", "source_group", "binary_label"]).size().unstack(fill_value=0)
    for label in [0, 1]:
        if label not in source.columns:
            source[label] = 0
    source = source.rename(columns={0: "non_acp", 1: "acp"})
    source["total"] = source["non_acp"] + source["acp"]
    source.reset_index()[["source_file", "source_group", "non_acp", "acp", "total"]].to_csv(
        out_dir / f"{prefix}_source_summary.csv", index=False
    )
    group = provenance.groupby(["source_group", "binary_label"]).size().unstack(fill_value=0)
    for label in [0, 1]:
        if label not in group.columns:
            group[label] = 0
    group = group.rename(columns={0: "non_acp", 1: "acp"})
    group["total"] = group["non_acp"] + group["acp"]
    group.reset_index()[["source_group", "non_acp", "acp", "total"]].to_csv(out_dir / f"{prefix}_source_group_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ACP data with audited labels and provenance.")
    parser.add_argument("--raw-dir", default="../..")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--skip-cdhit", action="store_true")
    parser.add_argument("--reuse-cdhit-dir", default=None, help="Reuse pre-audit CD-HIT 0.8 representatives from this backup directory.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_pre_label_audit_{timestamp}")
    audit_dir = Path("meta_acp/runs_post_label_audit/data_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_backup:
        backup_outputs(
            [
                Path("merged_all.csv"),
                Path("merged_all_cdhit08.csv"),
                Path("train.csv"),
                Path("validation.csv"),
                Path("test.csv"),
                Path("sequences.fasta"),
                Path("cdhit_result_08"),
                Path("cdhit_result_08.clstr"),
                Path("meta_acp/runs"),
            ],
            backup_dir,
        )

    raw = parse_raw_records(raw_dir)
    audit = validate_raw(raw)
    modeling, provenance, conflicts = aggregate_modeling(raw)
    raw.to_csv("raw_records_label_audit.csv", index=False)
    raw.to_csv(audit_dir / "raw_records_label_audit.csv", index=False)
    conflicts.to_csv("label_conflicts.csv", index=False)
    conflicts.to_csv(audit_dir / "label_conflicts.csv", index=False)
    modeling.to_csv("merged_all.csv", index=False)
    provenance.to_csv("provenance_long.csv", index=False)
    modeling.to_csv(audit_dir / "merged_all_pre_cdhit.csv", index=False)
    provenance.to_csv(audit_dir / "provenance_long_pre_cdhit.csv", index=False)
    write_source_summaries(provenance, audit_dir, "pre_cdhit")

    write_fasta(modeling, Path("sequences.fasta"))
    cdhit_reuse = False
    if args.reuse_cdhit_dir:
        representatives = reused_representatives_from_legacy(Path(args.reuse_cdhit_dir), modeling)
        cdhit_reuse = True
    else:
        if not args.skip_cdhit:
            run_cdhit(Path("sequences.fasta"), Path("cdhit_result_08"), identity=args.identity, threads=args.threads)
        representatives = parse_cdhit_representatives(Path("cdhit_result_08.clstr"))
    rep_set = set(representatives)
    if len(rep_set) != len(representatives):
        raise AssertionError("CD-HIT representative IDs are not unique")
    cdhit = modeling[modeling["seq_id"].isin(rep_set)].copy()
    order = {seq_id: i for i, seq_id in enumerate(representatives)}
    cdhit["cdhit_rank"] = cdhit["seq_id"].map(order)
    cdhit = cdhit.sort_values("cdhit_rank").drop(columns=["cdhit_rank"]).reset_index(drop=True)
    cdhit_provenance = provenance[provenance["seq_id"].isin(rep_set)].copy().reset_index(drop=True)
    cdhit.to_csv("merged_all_cdhit08.csv", index=False)
    cdhit_provenance.to_csv("provenance_cdhit08_long.csv", index=False)
    write_fasta(cdhit, Path("cdhit_result_08_representatives.fasta"))
    write_source_summaries(cdhit_provenance, audit_dir, "cdhit08")

    train, validation, test = stratified_split(cdhit, seed=args.seed)
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        split[["sequence", "binary_label"]].to_csv(f"{name}.csv", index=False)
        split.to_csv(f"{name}_source.csv", index=False)
        split.to_csv(audit_dir / f"{name}_source.csv", index=False)
    audit.update(
        {
            "modeling_sequences_after_conflict_removal": int(len(modeling)),
            "cdhit08_sequences": int(len(cdhit)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "backup_dir": str(backup_dir) if not args.skip_backup else None,
            "cdhit_reused_from": str(args.reuse_cdhit_dir) if cdhit_reuse else None,
        }
    )
    Path("label_audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (audit_dir / "label_audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
