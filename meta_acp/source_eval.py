"""Source-aware evaluation utilities for saved prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from metrics import binary_metrics, summarize_by_group
    from source_data import attach_source, load_provenance, source_summary
except ImportError:  # pragma: no cover
    from .metrics import binary_metrics, summarize_by_group
    from .source_data import attach_source, load_provenance, source_summary


def expand_with_provenance(pred: pd.DataFrame, provenance: pd.DataFrame | None) -> pd.DataFrame:
    if provenance is None:
        return pred
    key_col = "seq_id" if "seq_id" in pred.columns and "seq_id" in provenance.columns else "sequence"
    prov_cols = [key_col, "source_file", "source_group", "source_id", "raw_label"]
    prov_cols = [c for c in prov_cols if c in provenance.columns]
    prov = provenance[prov_cols].drop_duplicates([key_col, "source_file"]).copy()
    drop_cols = [c for c in ["source_file", "source_group", "source_id", "raw_label"] if c in pred.columns]
    base = pred.drop(columns=drop_cols)
    expanded = base.merge(prov, on=key_col, how="left")
    if expanded["source_file"].isna().any():
        n = int(expanded["source_file"].isna().sum())
        raise ValueError(f"{n} prediction rows could not be expanded with provenance")
    return expanded


def evaluate_predictions(
    predictions_csv: str | Path,
    test_csv: str | Path | None = None,
    master_csv: str | Path = "data/processed/merged_all_cdhit08.csv",
    provenance_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    pred = pd.read_csv(predictions_csv)
    if "source_file" not in pred.columns:
        if test_csv is None:
            raise ValueError("test_csv is required when predictions lack source_file")
        base = attach_source(test_csv, master_csv)
        pred = base.assign(score=pred["score"].values, prediction=(pred["score"].values >= 0.5).astype(int))
    overall = pd.DataFrame([binary_metrics(pred["binary_label"], pred["score"])])
    provenance = load_provenance(provenance_csv, master_csv)
    source_pred = expand_with_provenance(pred, provenance)
    by_source = summarize_by_group(source_pred, "source_file")
    by_group = summarize_by_group(source_pred, "source_group") if "source_group" in source_pred.columns else pd.DataFrame()
    false_cases = source_pred[source_pred["binary_label"].astype(int) != source_pred["prediction"].astype(int)].copy()
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        overall.to_csv(out / "overall_metrics.csv", index=False)
        by_source.to_csv(out / "source_wise_metrics.csv", index=False)
        if not by_group.empty:
            by_group.to_csv(out / "source_group_metrics.csv", index=False)
        false_cases.to_csv(out / "false_cases.csv", index=False)
    return {"overall": overall, "by_source": by_source, "by_group": by_group, "false_cases": false_cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions by source.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--master", default="data/processed/merged_all_cdhit08.csv")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--print-source-summary", action="store_true")
    args = parser.parse_args()
    if args.print_source_summary:
        print(source_summary(args.master, args.provenance).to_string(index=False))
    res = evaluate_predictions(args.predictions, args.test, args.master, args.provenance, args.out_dir)
    print(res["overall"].to_string(index=False))
    print(res["by_source"].to_string(index=False))


if __name__ == "__main__":
    main()
