from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEEDS = (42, 2024, 2026)
VARIANTS = ("full", "no_supcon", "no_proto", "classifier_only")
DISPLAY = {
    "full": "Full",
    "no_supcon": "No SupCon",
    "no_proto": "No Proto",
    "classifier_only": "Classifier only",
}


def run_dir(root: Path, variant: str, seed: int) -> Path:
    if variant == "full":
        return root / "plm" / "esm2_150m" / f"seed{seed}"
    return root / "ablations" / "esm2_150m" / f"{variant}_seed{seed}"


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ]
    )


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    metric,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = stratified_indices(y, rng)
        values[i] = metric(y[idx], score[idx])
    return tuple(np.quantile(values, [0.025, 0.975]))


def bootstrap_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = stratified_indices(y, rng)
        diffs[i] = metric(y[idx], score_a[idx]) - metric(y[idx], score_b[idx])
    low, high = np.quantile(diffs, [0.025, 0.975])
    p = min(1.0, 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return float(low), float(high), float(p)


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "sn": recall_score(y, pred, zero_division=0),
        "sp": recall_score(y, pred, pos_label=0, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="meta_acp/runs_gpu")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/ablation_esm2_150m_summary")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    predictions: dict[str, list[pd.DataFrame]] = {}
    reference_ids: list[str] | None = None
    reference_y: np.ndarray | None = None

    for variant in VARIANTS:
        predictions[variant] = []
        for seed in SEEDS:
            folder = run_dir(root, variant, seed)
            metrics = pd.read_csv(folder / "test_metrics.csv").iloc[0].to_dict()
            metric_rows.append({"variant": variant, "seed": seed, **metrics})

            pred = pd.read_csv(folder / "predictions.csv")
            pred = pred.sort_values("seq_id").reset_index(drop=True)
            ids = pred["seq_id"].astype(str).tolist()
            y = pred["binary_label"].to_numpy(dtype=int)
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            elif ids != reference_ids or not np.array_equal(y, reference_y):
                raise ValueError(f"Prediction rows are not aligned: {folder}")
            predictions[variant].append(pred)

    seed_metrics = pd.DataFrame(metric_rows)
    seed_metrics.to_csv(out_dir / "ablation_seed_metrics.csv", index=False)

    numeric_metrics = ["acc", "precision", "f1", "mcc", "sn", "sp", "auc", "ap"]
    summary = (
        seed_metrics.groupby("variant")[numeric_metrics]
        .agg(["mean", "std"])
        .reindex(VARIANTS)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "ablation_mean_std.csv", index=False)

    assert reference_y is not None
    ensemble_scores: dict[str, np.ndarray] = {}
    ensemble_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    for variant in VARIANTS:
        score = np.mean(
            [frame["score"].to_numpy(dtype=float) for frame in predictions[variant]],
            axis=0,
        )
        vote = (
            np.mean(
                [frame["prediction"].to_numpy(dtype=int) for frame in predictions[variant]],
                axis=0,
            )
            >= 0.5
        ).astype(int)
        ensemble_scores[variant] = score
        row = {
            "variant": variant,
            "method": "three_seed_mean_probability_with_majority_vote",
            **classification_metrics(reference_y, vote),
            "auc": roc_auc_score(reference_y, score),
            "ap": average_precision_score(reference_y, score),
        }
        for metric_name, metric_fn in (
            ("auc", roc_auc_score),
            ("ap", average_precision_score),
        ):
            low, high = bootstrap_ci(
                reference_y, score, metric_fn, args.n_bootstrap, rng
            )
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        ensemble_rows.append(row)
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(out_dir / "ablation_three_seed_ensemble.csv", index=False)

    comparison_rows: list[dict] = []
    for variant in VARIANTS[1:]:
        for metric_name, metric_fn in (
            ("auc", roc_auc_score),
            ("ap", average_precision_score),
        ):
            estimate_full = metric_fn(reference_y, ensemble_scores["full"])
            estimate_ablation = metric_fn(reference_y, ensemble_scores[variant])
            low, high, p = bootstrap_difference(
                reference_y,
                ensemble_scores["full"],
                ensemble_scores[variant],
                metric_fn,
                args.n_bootstrap,
                rng,
            )
            comparison_rows.append(
                {
                    "model_a": "full",
                    "model_b": variant,
                    "metric": metric_name,
                    "estimate_a": estimate_full,
                    "estimate_b": estimate_ablation,
                    "difference_a_minus_b": estimate_full - estimate_ablation,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "paired_bootstrap_p": p,
                    "n_bootstrap": args.n_bootstrap,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "ablation_paired_bootstrap.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    colors = ["#245a7a", "#bd5b36", "#a77a18", "#69727d"]
    for ax, metric in zip(axes, ("auc", "mcc", "ap")):
        means = summary[f"{metric}_mean"].to_numpy()
        stds = summary[f"{metric}_std"].to_numpy()
        ax.bar(x, means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(x, [DISPLAY[v] for v in VARIANTS], rotation=22, ha="right")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        lower = max(0.0, float(np.min(means - stds) - 0.03))
        ax.set_ylim(lower, min(1.0, float(np.max(means + stds) + 0.02)))
    fig.suptitle("ESM2-150M key ablations (mean ± SD, three seeds)")
    fig.savefig(out_dir / "ablation_mean_std.png", dpi=220)
    plt.close(fig)

    print(summary.to_string(index=False))
    print()
    print(ensemble.to_string(index=False))
    print()
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
