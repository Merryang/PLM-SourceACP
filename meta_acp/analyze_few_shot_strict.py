"""Strict paired analysis of ESM2-150M LOSO few-shot adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHODS = ["prototype", "classifier", "head"]
METRICS = ["acc", "mcc", "auc", "ap"]
K_SHOTS = [5, 10]


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return pd.Series(adjusted, index=p_values.index)


def bootstrap_mean_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def wilcoxon_safe(diff: np.ndarray) -> tuple[float, float]:
    if np.allclose(diff, 0):
        return 0.0, 1.0
    result = stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="meta_acp/runs_gpu/few_shot")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/few_shot_strict_analysis")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260618)
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    files = [
        runs / "esm2_150m_loso_Test1_fasta_seed42" / "few_shot_raw.csv",
        runs / "esm2_150m_loso_Test2_fasta_seed42" / "few_shot_raw.csv",
    ]
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw = raw[raw["k_shot"].isin(K_SHOTS)].copy()
    if raw.duplicated(["source_file", "k_shot", "repeat", "method"]).any():
        raise ValueError("duplicate source/k/repeat/method rows")

    expected_methods = {"none", *METHODS}
    for keys, part in raw.groupby(["source_file", "k_shot", "repeat"]):
        if set(part["method"]) != expected_methods:
            raise ValueError(f"incomplete paired methods for {keys}")

    descriptive = (
        raw.groupby(["source_file", "k_shot", "method"])[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    descriptive.columns = [
        "_".join(str(x) for x in col if str(x))
        if isinstance(col, tuple)
        else str(col)
        for col in descriptive.columns
    ]
    descriptive.to_csv(out / "few_shot_5_10_descriptive.csv", index=False)

    rows = []
    for (source, k), part in raw.groupby(["source_file", "k_shot"]):
        pivot = part.pivot(index="repeat", columns="method", values=METRICS)
        for method in METHODS:
            for metric in METRICS:
                adapted = pivot[(metric, method)].to_numpy(dtype=float)
                baseline = pivot[(metric, "none")].to_numpy(dtype=float)
                diff = adapted - baseline
                ci_low, ci_high = bootstrap_mean_ci(diff, rng, args.n_bootstrap)
                statistic, p_value = wilcoxon_safe(diff)
                shapiro_p = (
                    float(stats.shapiro(diff).pvalue)
                    if len(diff) >= 3 and not np.allclose(diff, diff[0])
                    else np.nan
                )
                rows.append(
                    {
                        "source_file": source,
                        "k_shot": int(k),
                        "method": method,
                        "metric": metric,
                        "n_pairs": len(diff),
                        "baseline_mean": baseline.mean(),
                        "adapted_mean": adapted.mean(),
                        "mean_difference": diff.mean(),
                        "median_difference": np.median(diff),
                        "difference_ci_low": ci_low,
                        "difference_ci_high": ci_high,
                        "wins": int((diff > 0).sum()),
                        "ties": int(np.isclose(diff, 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "shapiro_difference_p": shapiro_p,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p": p_value,
                    }
                )
    comparisons = pd.DataFrame(rows)
    comparisons["wilcoxon_p_holm_within_metric"] = comparisons.groupby("metric")["wilcoxon_p"].transform(
        holm_adjust
    )
    comparisons.to_csv(out / "few_shot_paired_comparisons.csv", index=False)

    primary = comparisons[comparisons["metric"].isin(["mcc", "auc"])].copy()
    method_decisions = []
    for method, part in primary.groupby("method"):
        mcc = part[part["metric"] == "mcc"]
        auc = part[part["metric"] == "auc"]
        stable_mcc_gain = bool(
            len(mcc) == 4
            and (mcc["mean_difference"] > 0).all()
            and (mcc["difference_ci_low"] > 0).all()
        )
        no_auc_harm = bool(len(auc) == 4 and (auc["difference_ci_low"] > -0.005).all())
        continue_method = stable_mcc_gain and no_auc_harm
        method_decisions.append(
            {
                "method": method,
                "stable_mcc_gain_all_4_settings": stable_mcc_gain,
                "no_auc_harm_gt_0.005_all_4_settings": no_auc_harm,
                "continue_method": continue_method,
            }
        )
    decisions = pd.DataFrame(method_decisions)
    decisions.to_csv(out / "few_shot_stop_decision.csv", index=False)
    stop_all = not decisions["continue_method"].any()

    manifest = {
        "backbone": "esm2_150m",
        "loso_checkpoint_seed": 42,
        "sources": sorted(raw["source_file"].unique().tolist()),
        "k_shot_analyzed": K_SHOTS,
        "repeats_per_setting": int(raw["repeat"].nunique()),
        "paired_design": "Within each source/k/repeat, adaptation methods and none use the same support/query split.",
        "primary_test": "Wilcoxon signed-rank on paired repeat-level metric differences",
        "multiple_testing": "Holm correction within each metric family",
        "effect_interval": f"{args.n_bootstrap}-resample bootstrap 95% CI for the mean paired difference",
        "stop_rule": "Continue only if a method has positive MCC with CI above zero in all 4 source/k settings and no AUC loss exceeding 0.005.",
        "stop_all_further_few_shot_tuning": stop_all,
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        comparisons[comparisons["metric"].isin(["mcc", "auc"])][
            [
                "source_file",
                "k_shot",
                "method",
                "metric",
                "baseline_mean",
                "adapted_mean",
                "mean_difference",
                "difference_ci_low",
                "difference_ci_high",
                "wilcoxon_p",
                "wilcoxon_p_holm_within_metric",
            ]
        ].to_string(index=False)
    )
    print("\nDecision:")
    print(decisions.to_string(index=False))
    print(f"\nstop_all_further_few_shot_tuning={stop_all}")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
