"""Source-aware data preparation for PLM-SourceACP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


FULL_BINARY_SOURCES = ["Train.fasta", "Test1.fasta", "Test2.fasta", "training.fasta", "testing.fasta"]
ACP_RICH_SOURCES: list[str] = []
SOURCE_GROUPS = {
    "Manuscript1856815": ["Train.fasta", "Test1.fasta", "Test2.fasta"],
    "Aegis": ["training.fasta", "testing.fasta"],
}


def load_master(path: str | Path = "merged_all_cdhit08.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sequence", "binary_label", "source_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df


def default_provenance_for(master_csv: str | Path = "merged_all_cdhit08.csv") -> Path | None:
    master = Path(master_csv)
    candidates = [
        master.with_name("provenance_cdhit08_long.csv"),
        master.with_name("provenance_long.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_provenance(path: str | Path | None = None, master_csv: str | Path = "merged_all_cdhit08.csv") -> pd.DataFrame | None:
    provenance_path = Path(path) if path else default_provenance_for(master_csv)
    if provenance_path is None or not provenance_path.exists():
        return None
    prov = pd.read_csv(provenance_path)
    required = {"sequence", "binary_label", "source_file", "source_group"}
    missing = required - set(prov.columns)
    if missing:
        raise ValueError(f"{provenance_path} missing required columns: {sorted(missing)}")
    return prov


def attach_source(split_csv: str | Path, master_csv: str | Path = "merged_all_cdhit08.csv") -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    if "source_file" in split.columns:
        merged = split.copy()
        if "length" not in merged.columns:
            merged["length"] = merged["sequence"].astype(str).str.len()
        merged["source_group"] = merged["source_file"].map(source_group_for)
        return merged
    master = load_master(master_csv)
    cols = [
        "sequence",
        "seq_id",
        "source_file",
        "source_id",
        "raw_label",
        "source_group",
        "length",
        "split_group",
        "all_source_files",
        "all_source_groups",
        "all_raw_labels",
        "all_source_ids",
        "n_raw_records",
    ]
    cols = [c for c in cols if c in master.columns]
    merged = split.merge(master[cols], on="sequence", how="left", validate="one_to_one")
    if merged["source_file"].isna().any():
        n = int(merged["source_file"].isna().sum())
        raise ValueError(f"{split_csv} has {n} sequences absent from {master_csv}")
    if "length" not in merged.columns:
        merged["length"] = merged["sequence"].astype(str).str.len()
    if "source_group" not in merged.columns:
        merged["source_group"] = merged["source_file"].map(source_group_for)
    return merged


def source_group_for(source_file: str) -> str:
    for group, sources in SOURCE_GROUPS.items():
        if source_file in sources:
            return group
    return "Other"


def write_enriched_splits(
    train_csv: str | Path = "train.csv",
    val_csv: str | Path = "validation.csv",
    test_csv: str | Path = "test.csv",
    master_csv: str | Path = "merged_all_cdhit08.csv",
    out_dir: str | Path = "meta_acp/runs/source_data",
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, path in [("train", train_csv), ("validation", val_csv), ("test", test_csv)]:
        df = attach_source(path, master_csv)
        p = out / f"{name}_source.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


def source_summary(master_csv: str | Path = "merged_all_cdhit08.csv", provenance_csv: str | Path | None = None) -> pd.DataFrame:
    df = load_provenance(provenance_csv, master_csv)
    if df is None:
        df = load_master(master_csv)
    tab = df.groupby(["source_file", "binary_label"]).size().unstack(fill_value=0)
    for label in [0, 1]:
        if label not in tab.columns:
            tab[label] = 0
    tab = tab.rename(columns={0: "non_acp", 1: "acp"})
    tab["total"] = tab["non_acp"] + tab["acp"]
    if "source_group" in df.columns:
        groups = df.groupby("source_file")["source_group"].agg(lambda x: sorted(set(x))[0])
        tab["source_group"] = [groups.get(x, source_group_for(x)) for x in tab.index]
    else:
        tab["source_group"] = [source_group_for(x) for x in tab.index]
    return tab.reset_index()[["source_file", "source_group", "non_acp", "acp", "total"]]


def make_loso_split(
    heldout_source: str,
    master_csv: str | Path = "merged_all_cdhit08.csv",
    out_dir: str | Path = "meta_acp/runs/source_data/loso",
    val_size: float = 0.2,
    seed: int = 42,
    exclude_acp_rich_from_train: bool = False,
    provenance_csv: str | Path | None = None,
) -> dict[str, Path]:
    df = load_master(master_csv).copy()
    prov = load_provenance(provenance_csv, master_csv)
    if prov is None:
        prov = df
    if heldout_source not in set(prov["source_file"]):
        raise ValueError(f"unknown heldout_source {heldout_source}")
    key_col = "seq_id" if "seq_id" in df.columns and "seq_id" in prov.columns else "sequence"
    heldout_ids = set(prov.loc[prov["source_file"] == heldout_source, key_col])
    test_df = df[df[key_col].isin(heldout_ids)].reset_index(drop=True)
    trainval = df[~df[key_col].isin(heldout_ids)].reset_index(drop=True)
    if exclude_acp_rich_from_train:
        exclude_ids = set(prov.loc[prov["source_file"].isin(ACP_RICH_SOURCES), key_col])
        trainval = trainval[~trainval[key_col].isin(exclude_ids)].reset_index(drop=True)
    train_df, val_df = stratified_train_val_split(trainval, val_size=val_size, seed=seed)
    out = Path(out_dir) / heldout_source.replace(".", "_")
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out / "train.csv",
        "validation": out / "validation.csv",
        "test": out / "test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def stratified_train_val_split(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    for _, part in df.groupby("binary_label", group_keys=False):
        shuffled = part.sample(frac=1.0, random_state=seed)
        n_val = max(1, int(round(len(shuffled) * val_size))) if len(shuffled) > 1 else 0
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val


def make_all_loso_splits(heldout_sources: Iterable[str] = FULL_BINARY_SOURCES, **kwargs) -> list[dict[str, Path]]:
    return [make_loso_split(source, **kwargs) for source in heldout_sources]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare source-aware PLM-SourceACP data splits.")
    parser.add_argument("--master", default="merged_all_cdhit08.csv")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--out-dir", default="meta_acp/runs/source_data")
    parser.add_argument("--write-enriched", action="store_true")
    parser.add_argument("--write-loso", action="store_true")
    parser.add_argument("--exclude-acp-rich-from-train", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = source_summary(args.master, args.provenance)
    summary.to_csv(out / "table1_source_summary.csv", index=False)
    print(summary.to_string(index=False))
    if args.write_enriched:
        paths = write_enriched_splits(master_csv=args.master, out_dir=out)
        print({k: str(v) for k, v in paths.items()})
    if args.write_loso:
        for source in FULL_BINARY_SOURCES:
            paths = make_loso_split(
                source,
                master_csv=args.master,
                provenance_csv=args.provenance,
                out_dir=out / "loso",
                exclude_acp_rich_from_train=args.exclude_acp_rich_from_train,
            )
            print(source, {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
