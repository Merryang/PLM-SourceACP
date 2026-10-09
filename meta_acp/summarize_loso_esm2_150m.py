"""Summarize the completed ESM2-150M LOSO experiments.

Expected layout:
  meta_acp/runs_gpu/loso/{Train,Test1,Test2}_fasta_seed{42,2024,2026}/

The script keeps the nine raw run directories unchanged and writes processed
tables, a source/seed heatmap, seed-ensemble metrics, and consensus error cases
to a separate summary directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meta_acp.metrics import METRIC_COLUMNS, binary_metrics


SOURCES = ["Train_fasta", "Test1_fasta", "Test2_fasta"]
SOURCE_LABELS = {
    "Train_fasta": "Train.fasta",
    "Test1_fasta": "Test1.fasta",
    "Test2_fasta": "Test2.fasta",
}
SEEDS = [42, 2024, 2026]
SUMMARY_METRICS = ["acc", "sn", "sp", "precision", "f1", "mcc", "auc", "ap"]


def read_run(run_dir: Path, source: str, seed: int) -> tuple[dict, pd.DataFrame]:
    metrics_path = run_dir / "test_metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    config_path = run_dir / "config.json"
    missing = [p.name for p in [metrics_path, predictions_path, config_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {', '.join(missing)}")

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    predictions = pd.read_csv(predictions_path)
    required = {"seq_id", "binary_label", "score"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{predictions_path}: missing columns {sorted(required - set(predictions.columns))}")
    if predictions["seq_id"].duplicated().any():
        raise ValueError(f"{predictions_path}: duplicate seq_id values")

    recomputed = binary_metrics(predictions["binary_label"], predictions["score"])
    for metric in METRIC_COLUMNS:
        expected = metrics.get(metric)
        actual = recomputed.get(metric)
        if pd.isna(expected) and pd.isna(actual):
            continue
        # Prediction CSV scores are rounded relative to the in-memory values
        # used for the stored metrics, so ranking metrics can differ slightly.
        if not np.isclose(float(expected), float(actual), atol=2e-4, rtol=1e-6):
            raise ValueError(
                f"{run_dir}: {metric} mismatch, stored={expected}, recomputed={actual}"
            )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    row = {
        "backbone": "esm2_150m",
        "heldout_source": SOURCE_LABELS[source],
        "source_key": source,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "monitor_metric": config.get("monitor_metric"),
        **metrics,
    }
    return row, predictions


def source_mean_std(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, part in seed_metrics.groupby("heldout_source", sort=False):
        row = {
            "backbone": "esm2_150m",
            "heldout_source": source,
            "n_seeds": len(part),
            "n": int(part["n"].iloc[0]),
            "pos": int(part["pos"].iloc[0]),
            "neg": int(part["neg"].iloc[0]),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_by_seed(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, part in seed_metrics.groupby("seed"):
        if set(part["heldout_source"]) != set(SOURCE_LABELS.values()):
            raise ValueError(f"seed {seed}: incomplete source set")
        row = {"backbone": "esm2_150m", "seed": int(seed), "n_sources": len(part)}
        for metric in SUMMARY_METRICS:
            row[metric] = pd.to_numeric(part[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def summarize_macro(macro_seed: pd.DataFrame) -> pd.DataFrame:
    row = {"backbone": "esm2_150m", "n_sources": 3, "n_seeds": len(macro_seed)}
    for metric in SUMMARY_METRICS:
        values = pd.to_numeric(macro_seed[metric], errors="coerce")
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=0)
    return pd.DataFrame([row])


def ensemble_and_errors(predictions: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble_rows = []
    error_frames = []
    for source in SOURCES:
        merged = None
        metadata_cols = [
            c
            for c in [
                "seq_id",
                "sequence",
                "binary_label",
                "length",
                "source_file",
                "source_group",
                "all_source_files",
                "all_source_groups",
                "all_source_ids",
            ]
            if c in predictions[(source, SEEDS[0])].columns
        ]
        for seed in SEEDS:
            pred = predictions[(source, seed)].copy()
            pred = pred[metadata_cols + ["score"]].rename(columns={"score": f"score_seed{seed}"})
            if merged is None:
                merged = pred
            else:
                shared_metadata = [c for c in metadata_cols if c != "seq_id"]
                check = merged[["seq_id"] + shared_metadata].merge(
                    pred[["seq_id"] + shared_metadata],
                    on="seq_id",
                    how="outer",
                    suffixes=("_left", "_right"),
                    indicator=True,
                )
                if not (check["_merge"] == "both").all():
                    raise ValueError(f"{source}: seed prediction rows are not aligned")
                merged = merged.merge(pred[["seq_id", f"score_seed{seed}"]], on="seq_id", how="inner")

        score_cols = [f"score_seed{seed}" for seed in SEEDS]
        merged["mean_score"] = merged[score_cols].mean(axis=1)
        merged["score_std"] = merged[score_cols].std(axis=1, ddof=0)
        merged["positive_votes"] = (merged[score_cols] >= 0.5).sum(axis=1)
        merged["ensemble_prediction"] = (merged["mean_score"] >= 0.5).astype(int)
        merged["heldout_source"] = SOURCE_LABELS[source]
        metrics = binary_metrics(merged["binary_label"], merged["mean_score"])
        ensemble_rows.append(
            {"backbone": "esm2_150m", "heldout_source": SOURCE_LABELS[source], "method": "mean_probability_3seed", **metrics}
        )

        errors = merged[merged["ensemble_prediction"] != merged["binary_label"]].copy()
        errors["error_type"] = np.where(errors["binary_label"] == 1, "FN", "FP")
        errors["error_confidence"] = np.where(
            errors["error_type"] == "FN", 1.0 - errors["mean_score"], errors["mean_score"]
        )
        error_frames.append(errors)

    ensemble = pd.DataFrame(ensemble_rows)
    macro = {"backbone": "esm2_150m", "heldout_source": "Macro average", "method": "mean_probability_3seed"}
    for metric in SUMMARY_METRICS:
        macro[metric] = ensemble[metric].mean()
    macro["n"] = ensemble["n"].sum()
    macro["pos"] = ensemble["pos"].sum()
    macro["neg"] = ensemble["neg"].sum()
    ensemble = pd.concat([ensemble, pd.DataFrame([macro])], ignore_index=True)

    errors = pd.concat(error_frames, ignore_index=True)
    keep_cols = [
        "heldout_source",
        "seq_id",
        "sequence",
        "binary_label",
        "length",
        "source_file",
        "source_group",
        "mean_score",
        "score_std",
        "positive_votes",
        "ensemble_prediction",
        "error_type",
        "error_confidence",
    ]
    errors = errors[[c for c in keep_cols if c in errors.columns]].sort_values(
        ["error_confidence", "score_std"], ascending=[False, False]
    )
    return ensemble, errors


def save_heatmap(seed_metrics: pd.DataFrame, out_path: Path) -> None:
    matrix = seed_metrics.pivot(index="heldout_source", columns="seed", values="auc").reindex(
        [SOURCE_LABELS[s] for s in SOURCES]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0.85, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(x) for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Held-out source")
    ax.set_title("ESM2-150M LOSO AUC by source and seed")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="black", fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ESM2-150M LOSO runs.")
    parser.add_argument("--runs-dir", default="meta_acp/runs_gpu/loso")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/loso_summary_esm2_150m")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for source in SOURCES:
        for seed in SEEDS:
            run_dir = runs_dir / f"{source}_seed{seed}"
            row, pred = read_run(run_dir, source, seed)
            rows.append(row)
            predictions[(source, seed)] = pred

    seed_metrics = pd.DataFrame(rows).sort_values(["heldout_source", "seed"]).reset_index(drop=True)
    source_summary = source_mean_std(seed_metrics)
    macro_seed = macro_by_seed(seed_metrics)
    macro_summary = summarize_macro(macro_seed)
    ensemble_metrics, errors = ensemble_and_errors(predictions)

    seed_metrics.to_csv(out_dir / "loso_seed_metrics.csv", index=False)
    source_summary.to_csv(out_dir / "loso_source_mean_std.csv", index=False)
    macro_seed.to_csv(out_dir / "loso_macro_by_seed.csv", index=False)
    macro_summary.to_csv(out_dir / "loso_macro_summary.csv", index=False)
    ensemble_metrics.to_csv(out_dir / "loso_seed_ensemble_metrics.csv", index=False)
    errors.to_csv(out_dir / "loso_consensus_error_cases.csv", index=False)
    save_heatmap(seed_metrics, out_dir / "loso_auc_heatmap.png")

    manifest = {
        "backbone": "esm2_150m",
        "sources": [SOURCE_LABELS[s] for s in SOURCES],
        "seeds": SEEDS,
        "n_runs": len(seed_metrics),
        "all_run_metrics_recomputed": True,
        "macro_definition": "Unweighted mean over Train.fasta, Test1.fasta, and Test2.fasta; then summarized over seeds.",
        "seed_ensemble_definition": "Per-source arithmetic mean of the three seed probabilities.",
        "n_consensus_errors": len(errors),
    }
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(source_summary.to_string(index=False))
    print("\nMacro across three held-out sources:")
    print(macro_summary.to_string(index=False))
    print("\nThree-seed probability ensemble:")
    print(ensemble_metrics.to_string(index=False))
    print(f"\nWrote {out_dir.resolve()}")


if __name__ == "__main__":
    main()
