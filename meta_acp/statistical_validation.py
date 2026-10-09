"""Freeze final main/Aegis candidates and compute bootstrap statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from meta_acp.metrics import binary_metrics


REPORT_METRICS = ["acc", "mcc", "sn", "sp", "auc", "ap"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_predictions(paths: list[Path]) -> list[pd.DataFrame]:
    frames = [pd.read_csv(path) for path in paths]
    reference = frames[0][["seq_id", "binary_label"]].sort_values("seq_id").reset_index(drop=True)
    output = []
    for path, frame in zip(paths, frames):
        current = frame[["seq_id", "binary_label"]].sort_values("seq_id").reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f"prediction rows are not aligned: {path}")
        output.append(frame.sort_values("seq_id").reset_index(drop=True))
    return output


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
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, tuple[float, float]]]:
    point = binary_metrics(y, score, threshold=threshold)
    samples = {metric: np.empty(n_boot, dtype=float) for metric in REPORT_METRICS}
    for i in range(n_boot):
        idx = stratified_indices(y, rng)
        metrics = binary_metrics(y[idx], score[idx], threshold=threshold)
        for metric in REPORT_METRICS:
            samples[metric][i] = metrics[metric]
    ci = {
        metric: (
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )
        for metric, values in samples.items()
    }
    return point, ci


def paired_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    point_a = binary_metrics(y, score_a, threshold=threshold_a)
    point_b = binary_metrics(y, score_b, threshold=threshold_b)
    diff_samples = {metric: np.empty(n_boot, dtype=float) for metric in ["auc", "ap", "mcc"]}
    for i in range(n_boot):
        idx = stratified_indices(y, rng)
        metrics_a = binary_metrics(y[idx], score_a[idx], threshold=threshold_a)
        metrics_b = binary_metrics(y[idx], score_b[idx], threshold=threshold_b)
        for metric in diff_samples:
            diff_samples[metric][i] = metrics_a[metric] - metrics_b[metric]

    rows = []
    for metric, values in diff_samples.items():
        observed = point_a[metric] - point_b[metric]
        p_value = min(
            1.0,
            2.0
            * min(
                float(np.mean(values <= 0)),
                float(np.mean(values >= 0)),
            ),
        )
        rows.append(
            {
                "metric": metric,
                "estimate_a": point_a[metric],
                "estimate_b": point_b[metric],
                "difference_a_minus_b": observed,
                "difference_ci_low": float(np.nanpercentile(values, 2.5)),
                "difference_ci_high": float(np.nanpercentile(values, 97.5)),
                "paired_bootstrap_p": p_value,
                "n_bootstrap": n_boot,
            }
        )
    return rows


def append_ci_rows(
    rows: list[dict],
    dataset: str,
    model: str,
    frame: pd.DataFrame,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> None:
    y = frame["binary_label"].astype(int).to_numpy()
    score = frame["score"].astype(float).to_numpy()
    point, ci = bootstrap_ci(y, score, threshold, n_boot, rng)
    for metric in REPORT_METRICS:
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "threshold": threshold,
                "metric": metric,
                "estimate": point[metric],
                "ci_low": ci[metric][0],
                "ci_high": ci[metric][1],
                "n": point["n"],
                "pos": point["pos"],
                "neg": point["neg"],
                "n_bootstrap": n_boot,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="meta_acp/runs_gpu")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/final_statistics")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260618)
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    main_paths = {
        "main_strict_logistic_stack_C3": runs
        / "ensemble_with_hardcase_strict"
        / "strict_logistic_stack"
        / "predictions.csv",
        "main_prespecified_logistic_stack_C1": runs
        / "ensemble_with_hardcase_strict"
        / "logistic_stack"
        / "predictions.csv",
        "main_esm2_150m_seed2024": runs / "plm" / "esm2_150m" / "seed2024" / "predictions.csv",
    }
    main_frames_list = aligned_predictions(list(main_paths.values()))
    main_frames = dict(zip(main_paths, main_frames_list))

    threshold_table = pd.read_csv(
        runs
        / "branch_benchmarks"
        / "benchmark_aegis_cdhit08"
        / "esm2_150m_threshold_calibrated.csv"
    )
    threshold_by_seed = {}
    for _, row in threshold_table.iterrows():
        match = re.search(r"seed(\d+)$", str(row["run_dir"]).replace("\\", "/"))
        if not match:
            raise ValueError(f"cannot parse seed from {row['run_dir']}")
        threshold_by_seed[int(match.group(1))] = float(row["selected_threshold"])

    aegis_paths = {
        seed: runs
        / "branch_benchmarks"
        / "benchmark_aegis_cdhit08"
        / "plm"
        / "esm2_150m"
        / f"seed{seed}"
        / "predictions.csv"
        for seed in [42, 2024, 2026]
    }
    aegis_frames_list = aligned_predictions(list(aegis_paths.values()))
    aegis_frames = dict(zip(aegis_paths, aegis_frames_list))
    aegis_ensemble = aegis_frames[42].copy()
    score_matrix = np.column_stack([aegis_frames[seed]["score"].astype(float) for seed in [42, 2024, 2026]])
    aegis_ensemble["score"] = score_matrix.mean(axis=1)
    aegis_ensemble_threshold = float(np.median(list(threshold_by_seed.values())))
    aegis_ensemble["prediction"] = (aegis_ensemble["score"] >= aegis_ensemble_threshold).astype(int)
    aegis_ensemble.to_csv(out / "aegis_esm2_150m_3seed_ensemble_predictions.csv", index=False)

    ci_rows: list[dict] = []
    append_ci_rows(
        ci_rows,
        "main_post_audit_test",
        "strict_logistic_stack_C3",
        main_frames["main_strict_logistic_stack_C3"],
        0.5,
        args.n_bootstrap,
        rng,
    )
    append_ci_rows(
        ci_rows,
        "main_post_audit_test",
        "prespecified_logistic_stack_C1_auxiliary",
        main_frames["main_prespecified_logistic_stack_C1"],
        0.5,
        args.n_bootstrap,
        rng,
    )
    append_ci_rows(
        ci_rows,
        "main_post_audit_test",
        "esm2_150m_seed2024_validation_selected_single",
        main_frames["main_esm2_150m_seed2024"],
        0.5,
        args.n_bootstrap,
        rng,
    )
    for seed in [42, 2024, 2026]:
        append_ci_rows(
            ci_rows,
            "aegis_testing_fasta",
            f"esm2_150m_seed{seed}",
            aegis_frames[seed],
            threshold_by_seed[seed],
            args.n_bootstrap,
            rng,
        )
    append_ci_rows(
        ci_rows,
        "aegis_testing_fasta",
        "esm2_150m_3seed_mean_probability",
        aegis_ensemble,
        aegis_ensemble_threshold,
        args.n_bootstrap,
        rng,
    )
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(out / "metric_bootstrap_confidence_intervals.csv", index=False)

    comparison_rows = []
    main_y = main_frames["main_strict_logistic_stack_C3"]["binary_label"].astype(int).to_numpy()
    main_comparisons = [
        (
            "strict_logistic_stack_C3",
            main_frames["main_strict_logistic_stack_C3"]["score"].to_numpy(),
            "esm2_150m_seed2024_validation_selected_single",
            main_frames["main_esm2_150m_seed2024"]["score"].to_numpy(),
        ),
        (
            "strict_logistic_stack_C3",
            main_frames["main_strict_logistic_stack_C3"]["score"].to_numpy(),
            "prespecified_logistic_stack_C1_auxiliary",
            main_frames["main_prespecified_logistic_stack_C1"]["score"].to_numpy(),
        ),
    ]
    for model_a, score_a, model_b, score_b in main_comparisons:
        for row in paired_bootstrap(main_y, score_a, score_b, 0.5, 0.5, args.n_bootstrap, rng):
            comparison_rows.append(
                {
                    "dataset": "main_post_audit_test",
                    "model_a": model_a,
                    "model_b": model_b,
                    **row,
                }
            )

    aegis_y = aegis_ensemble["binary_label"].astype(int).to_numpy()
    for seed in [42, 2024, 2026]:
        for row in paired_bootstrap(
            aegis_y,
            aegis_ensemble["score"].to_numpy(),
            aegis_frames[seed]["score"].to_numpy(),
            aegis_ensemble_threshold,
            threshold_by_seed[seed],
            args.n_bootstrap,
            rng,
        ):
            comparison_rows.append(
                {
                    "dataset": "aegis_testing_fasta",
                    "model_a": "esm2_150m_3seed_mean_probability",
                    "model_b": f"esm2_150m_seed{seed}",
                    **row,
                }
            )
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(out / "paired_bootstrap_comparisons.csv", index=False)

    aegis_seed_rows = []
    for seed in [42, 2024, 2026]:
        metrics = binary_metrics(
            aegis_frames[seed]["binary_label"],
            aegis_frames[seed]["score"],
            threshold=threshold_by_seed[seed],
        )
        aegis_seed_rows.append({"seed": seed, "selected_threshold": threshold_by_seed[seed], **metrics})
    aegis_seed_table = pd.DataFrame(aegis_seed_rows)
    summary = {"seed": "mean±std", "selected_threshold": aegis_seed_table["selected_threshold"].mean()}
    for metric in REPORT_METRICS:
        summary[metric] = f"{aegis_seed_table[metric].mean():.4f}±{aegis_seed_table[metric].std(ddof=0):.4f}"
    pd.concat([aegis_seed_table, pd.DataFrame([summary])], ignore_index=True).to_csv(
        out / "aegis_three_seed_summary.csv", index=False
    )

    search = pd.read_csv(
        runs
        / "ensemble_with_hardcase_strict"
        / "strict_logistic_stack_search_by_validation.csv"
    )
    selected = search.sort_values(["val_auc", "val_ap", "val_mcc"], ascending=False).iloc[0]
    if float(selected["C"]) != 3.0:
        raise ValueError(f"unexpected validation-selected C: {selected['C']}")

    freeze_manifest = {
        "created_from": "existing completed predictions; no test-set refitting",
        "main_primary": {
            "name": "strict_logistic_stack_C3",
            "selection": "C selected by validation AUC from the predeclared C grid",
            "C": 3.0,
            "threshold": 0.5,
            "predictions": str(main_paths["main_strict_logistic_stack_C3"].resolve()),
            "predictions_sha256": sha256(main_paths["main_strict_logistic_stack_C3"]),
            "config": str((runs / "ensemble_with_hardcase_strict" / "config.json").resolve()),
            "config_sha256": sha256(runs / "ensemble_with_hardcase_strict" / "config.json"),
        },
        "main_auxiliary": {
            "name": "prespecified_logistic_stack_C1",
            "reason": "Pre-specified sklearn default; retained as auxiliary, not the validation-selected primary.",
            "threshold": 0.5,
            "predictions_sha256": sha256(main_paths["main_prespecified_logistic_stack_C1"]),
        },
        "aegis_primary": {
            "name": "esm2_150m_3seed_mean_probability",
            "seeds": [42, 2024, 2026],
            "aggregation": "arithmetic mean of test probabilities",
            "threshold": aegis_ensemble_threshold,
            "threshold_rule": "median of the three independently validation-selected MCC thresholds",
            "component_thresholds": threshold_by_seed,
            "predictions": str((out / "aegis_esm2_150m_3seed_ensemble_predictions.csv").resolve()),
            "predictions_sha256": sha256(out / "aegis_esm2_150m_3seed_ensemble_predictions.csv"),
        },
        "bootstrap": {
            "method": "stratified nonparametric bootstrap preserving positive/negative counts",
            "n_resamples": args.n_bootstrap,
            "confidence_level": 0.95,
            "random_seed": args.random_seed,
        },
        "comparison_boundary": "Published Aegis AUC=0.980 and AP=0.995 is descriptive only; no DeLong or paired test is valid without predictions on the same samples.",
    }
    (out / "frozen_model_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        ci_table[
            (ci_table["metric"].isin(["auc", "ap", "mcc"]))
        ][["dataset", "model", "metric", "estimate", "ci_low", "ci_high"]].to_string(index=False)
    )
    print("\nPaired comparisons:")
    print(comparison_table.to_string(index=False))
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
