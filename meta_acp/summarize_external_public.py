"""Summarize completed AntiAngioPred and PepBenchmark CPP experiments."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from meta_acp.calibrate_threshold import select_threshold, threshold_metrics
from meta_acp.dataset import EmbeddingDataset, collate_embedding_batch
from meta_acp.metrics import binary_metrics
from meta_acp.train_plm_metaacp import build_model, predict


TASKS = {
    "antiangiopred": {
        "label": "AntiAngioPred official independent test",
        "data_dir": "meta_acp/external_tasks/antiangiopred_official",
    },
    "pepbenchmark_cpp": {
        "label": "PepBenchmark CPP CD-HIT seed0 test",
        "data_dir": "meta_acp/external_tasks/pepbenchmark_cpp_cdhit_seed0",
    },
}
SEEDS = [42, 2024, 2026]
REPORT_METRICS = ["acc", "mcc", "sn", "sp", "auc", "ap"]


def predict_split(run_dir: Path, split: str, device: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = Namespace(**cfg)
    csv_path = Path(getattr(args, split))
    ds = EmbeddingDataset(csv_path, args.embedding_cache, args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_embedding_batch,
    )
    model = build_model(args).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y, score = predict(model, loader, device)
    return pd.read_csv(csv_path).reset_index(drop=True), y.astype(int), score.astype(float)


def stratified_bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    point = binary_metrics(y, score, threshold)
    values = {metric: np.empty(n_boot) for metric in REPORT_METRICS}
    for i in range(n_boot):
        idx = np.r_[
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ]
        rec = binary_metrics(y[idx], score[idx], threshold)
        for metric in REPORT_METRICS:
            values[metric][i] = rec[metric]
    return [
        {
            "metric": metric,
            "estimate": point[metric],
            "ci_low": float(np.nanpercentile(samples, 2.5)),
            "ci_high": float(np.nanpercentile(samples, 97.5)),
            "n_bootstrap": n_boot,
        }
        for metric, samples in values.items()
    ]


def paired_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    point_a = binary_metrics(y, score_a, threshold_a)
    point_b = binary_metrics(y, score_b, threshold_b)
    rows = []
    for metric in ["auc", "ap", "mcc"]:
        samples = np.empty(n_boot)
        for i in range(n_boot):
            idx = np.r_[
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ]
            a = binary_metrics(y[idx], score_a[idx], threshold_a)[metric]
            b = binary_metrics(y[idx], score_b[idx], threshold_b)[metric]
            samples[i] = a - b
        p_value = min(1.0, 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean())))
        rows.append(
            {
                "metric": metric,
                "estimate_a": point_a[metric],
                "estimate_b": point_b[metric],
                "difference_a_minus_b": point_a[metric] - point_b[metric],
                "difference_ci_low": float(np.nanpercentile(samples, 2.5)),
                "difference_ci_high": float(np.nanpercentile(samples, 97.5)),
                "paired_bootstrap_p": p_value,
                "n_bootstrap": n_boot,
            }
        )
    return rows


def analyze_few_shot(raw_path: Path, task: str, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    rows = []
    for (k, method), part in raw[raw["method"] != "no_adaptation"].groupby(["k_shot", "method"]):
        base = raw[(raw["k_shot"] == k) & (raw["method"] == "no_adaptation")].set_index("repeat")
        adapted = part.set_index("repeat")
        for metric in ["mcc", "auc", "ap"]:
            diff = adapted.loc[base.index, metric].to_numpy() - base[metric].to_numpy()
            boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
            if np.allclose(diff, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "task": task,
                    "k_shot": int(k),
                    "method": method,
                    "metric": metric,
                    "baseline_mean": base[metric].mean(),
                    "adapted_mean": adapted.loc[base.index, metric].mean(),
                    "mean_difference": diff.mean(),
                    "difference_ci_low": float(np.percentile(boot, 2.5)),
                    "difference_ci_high": float(np.percentile(boot, 97.5)),
                    "wins": int((diff > 0).sum()),
                    "ties": int(np.isclose(diff, 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="meta_acp/runs_gpu/external_public")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/external_public_summary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260623)
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    seed_rows = []
    ensemble_rows = []
    ci_rows = []
    comparison_rows = []
    fewshot_parts = []

    for task, task_cfg in TASKS.items():
        task_dir = runs / task
        val_scores = []
        test_scores = []
        val_y_ref = None
        test_y_ref = None
        test_df_ref = None

        for seed in SEEDS:
            run_dir = task_dir / f"full_head_seed{seed}" / "full_head"
            _, y_val, score_val = predict_split(run_dir, "validation", args.device)
            test_df, y_test, score_test = predict_split(run_dir, "test", args.device)
            if val_y_ref is not None and not np.array_equal(val_y_ref, y_val):
                raise ValueError(f"{task}: validation labels differ across seeds")
            if test_y_ref is not None and not np.array_equal(test_y_ref, y_test):
                raise ValueError(f"{task}: test labels differ across seeds")
            val_y_ref = y_val
            test_y_ref = y_test
            test_df_ref = test_df
            val_scores.append(score_val)
            test_scores.append(score_test)

            threshold, val_rec = select_threshold(y_val, score_val, "mcc")
            test_rec = threshold_metrics(y_test, score_test, threshold)
            seed_rows.append(
                {
                    "task": task,
                    "task_label": task_cfg["label"],
                    "seed": seed,
                    "selected_threshold": threshold,
                    **{f"val_{k}": v for k, v in val_rec.items()},
                    **{f"test_{k}": v for k, v in test_rec.items()},
                }
            )

        mean_val = np.column_stack(val_scores).mean(axis=1)
        mean_test = np.column_stack(test_scores).mean(axis=1)
        ensemble_threshold, val_rec = select_threshold(val_y_ref, mean_val, "mcc")
        test_rec = threshold_metrics(test_y_ref, mean_test, ensemble_threshold)
        ensemble_rows.append(
            {
                "task": task,
                "task_label": task_cfg["label"],
                "method": "esm2_150m_full_head_3seed_mean_probability",
                "selected_threshold": ensemble_threshold,
                **{f"val_{k}": v for k, v in val_rec.items()},
                **{f"test_{k}": v for k, v in test_rec.items()},
            }
        )
        pred = test_df_ref.copy()
        pred["score_seed42"] = test_scores[0]
        pred["score_seed2024"] = test_scores[1]
        pred["score_seed2026"] = test_scores[2]
        pred["score"] = mean_test
        pred["prediction"] = (mean_test >= ensemble_threshold).astype(int)
        pred.to_csv(out / f"{task}_full_head_3seed_ensemble_predictions.csv", index=False)

        for rec in stratified_bootstrap_ci(
            test_y_ref, mean_test, ensemble_threshold, args.n_bootstrap, rng
        ):
            ci_rows.append({"task": task, "method": "full_head_3seed_ensemble", **rec})

        linear_path = task_dir / "linear_probe" / "linear_probe_mean_embedding_predictions.csv"
        linear = pd.read_csv(linear_path)
        if "sequence" in linear.columns and "sequence" in test_df_ref.columns:
            merged = pred[["sequence", "binary_label", "score"]].merge(
                linear[["sequence", "binary_label", "score"]],
                on=["sequence", "binary_label"],
                suffixes=("_ensemble", "_linear"),
            )
        else:
            merged = pd.DataFrame(
                {
                    "binary_label": test_y_ref,
                    "score_ensemble": mean_test,
                    "score_linear": linear["score"].to_numpy(),
                }
            )
        if len(merged) != len(test_y_ref):
            raise ValueError(f"{task}: linear/full-head predictions are not aligned")
        for rec in paired_bootstrap(
            merged["binary_label"].to_numpy(),
            merged["score_ensemble"].to_numpy(),
            merged["score_linear"].to_numpy(),
            ensemble_threshold,
            0.5,
            args.n_bootstrap,
            rng,
        ):
            comparison_rows.append(
                {
                    "task": task,
                    "model_a": "full_head_3seed_ensemble",
                    "model_b": "linear_probe_mean_embedding",
                    **rec,
                }
            )

        fewshot_parts.append(
            analyze_few_shot(
                task_dir / "few_shot_acp_checkpoint" / "few_shot_raw.csv",
                task,
                rng,
                args.n_bootstrap,
            )
        )

    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(out / "external_full_head_seed_calibrated_metrics.csv", index=False)
    summary_rows = []
    for task, part in seed_table.groupby("task"):
        row = {"task": task, "n_seeds": len(part)}
        for metric in REPORT_METRICS:
            values = part[f"test_{metric}"].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out / "external_full_head_seed_mean_std.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(out / "external_full_head_ensemble_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(out / "external_full_head_ensemble_bootstrap_ci.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out / "external_full_head_vs_linear_probe.csv", index=False)
    fewshot = pd.concat(fewshot_parts, ignore_index=True)
    fewshot.to_csv(out / "external_few_shot_paired_analysis.csv", index=False)

    manifest = {
        "tasks": TASKS,
        "seeds": SEEDS,
        "threshold_rule": "Select MCC-maximizing threshold on validation only.",
        "primary_external_model": "Arithmetic mean of three independently trained full-head probabilities.",
        "bootstrap": {
            "n_resamples": args.n_bootstrap,
            "method": "Stratified nonparametric bootstrap preserving class counts.",
            "random_seed": args.random_seed,
        },
        "few_shot_note": "Exploratory transfer from the frozen ACP checkpoint; paired across 20 support-set repeats.",
    }
    (out / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(pd.DataFrame(ensemble_rows).to_string(index=False))
    print("\nBootstrap CI:")
    print(pd.DataFrame(ci_rows).to_string(index=False))
    print("\nFull-head ensemble vs linear probe:")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
