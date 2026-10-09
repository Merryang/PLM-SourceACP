"""Analyze hard cases and derive validation-driven training weights.

The script intentionally derives sample weights from validation errors only.
Test-set errors are reported for interpretation, but never used for weight
construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


HYDROPHOBIC = set("AVILMFWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seqs = out["sequence"].astype(str)
    out["length"] = seqs.str.len()
    out["net_charge"] = seqs.map(lambda s: sum(aa in POSITIVE for aa in s) - sum(aa in NEGATIVE for aa in s))
    out["hydrophobic_frac"] = seqs.map(lambda s: sum(aa in HYDROPHOBIC for aa in s) / max(len(s), 1))
    out["cationic_frac"] = seqs.map(lambda s: sum(aa in POSITIVE for aa in s) / max(len(s), 1))
    out["length_bin"] = pd.cut(
        out["length"],
        bins=[0, 10, 15, 20, 30, 50, 10_000],
        labels=["<=10", "11-15", "16-20", "21-30", "31-50", ">50"],
        include_lowest=True,
    ).astype(str)
    out["charge_bin"] = pd.cut(
        out["net_charge"],
        bins=[-10_000, -3, 0, 3, 6, 10_000],
        labels=["<=-3", "-2-0", "1-3", "4-6", ">=7"],
        include_lowest=True,
    ).astype(str)
    out["hydrophobic_bin"] = pd.cut(
        out["hydrophobic_frac"],
        bins=[0, 0.30, 0.45, 0.60, 1.0],
        labels=["<=0.30", "0.30-0.45", "0.45-0.60", ">0.60"],
        include_lowest=True,
    ).astype(str)
    return out


def load_member_scores(member_dir: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    files = sorted((member_dir / split).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no member prediction files under {member_dir / split}")
    base = None
    members = []
    for path in files:
        df = pd.read_csv(path)
        member = path.stem
        members.append(member)
        cols = df[["sequence", "binary_label", "score"]].rename(columns={"score": member})
        if base is None:
            base = cols
        else:
            base = base.merge(cols, on=["sequence", "binary_label"], how="inner")
    assert base is not None
    return base, members


def rank_scores(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="mergesort")
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(len(order), dtype=float)
        out[:, j] = ranks / max(1, len(order) - 1)
    return out


def combine(df: pd.DataFrame, members: list[str], method: str, weights: list[float] | None = None) -> np.ndarray:
    x = df[members].to_numpy(float)
    w = np.ones(len(members), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    if method in {"prob", "prob_mean", "mean"}:
        return x @ w
    if method in {"logit", "logit_mean"}:
        x = np.clip(x, 1e-6, 1 - 1e-6)
        logits = np.log(x / (1 - x))
        z = logits @ w
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    if method in {"rank", "rank_mean"}:
        return rank_scores(x) @ w
    raise ValueError(f"unsupported ensemble method: {method}")


def attach_split_columns(pred: pd.DataFrame, split_csv: Path) -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    keep = [c for c in ["sequence", "source_file", "all_source_files", "source_group", "all_source_groups"] if c in split.columns]
    if len(keep) > 1:
        pred = pred.merge(split[keep].drop_duplicates("sequence"), on="sequence", how="left")
    return features(pred)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prediction"] = (out["score"] >= 0.5).astype(int)
    out["is_error"] = out["prediction"] != out["binary_label"].astype(int)
    out["error_type"] = np.where(
        (out["binary_label"].astype(int) == 1) & (out["prediction"] == 0),
        "FN",
        np.where((out["binary_label"].astype(int) == 0) & (out["prediction"] == 1), "FP", "correct"),
    )
    out["margin_to_boundary"] = (out["score"] - 0.5).abs()
    out["hardness"] = np.where(out["is_error"], 1.0 + out["margin_to_boundary"], 1.0 - out["margin_to_boundary"])
    return out


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = g["binary_label"].astype(int).to_numpy()
        s = g["score"].to_numpy(float)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            n=len(g),
            pos=int(y.sum()),
            neg=int((1 - y).sum()),
            error_rate=float(g["is_error"].mean()),
            fn=int((g["error_type"] == "FN").sum()),
            fp=int((g["error_type"] == "FP").sum()),
            mean_score=float(g["score"].mean()),
        )
        if len(np.unique(y)) == 2:
            row["auc"] = float(roc_auc_score(y, s))
            row["ap"] = float(average_precision_score(y, s))
        else:
            row["auc"] = np.nan
            row["ap"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["error_rate", "n"], ascending=[False, False])


def build_train_weights(train_csv: Path, val_pred: pd.DataFrame, out_path: Path, alpha: float, max_weight: float) -> pd.DataFrame:
    train = features(pd.read_csv(train_csv))
    train["sample_weight"] = 1.0
    overall = float(val_pred["is_error"].mean())
    specs = [
        (["binary_label", "source_file"], 0.45),
        (["binary_label", "length_bin"], 0.35),
        (["binary_label", "charge_bin"], 0.25),
        (["binary_label", "hydrophobic_bin"], 0.25),
    ]
    for cols, scale in specs:
        if any(c not in val_pred.columns for c in cols) or any(c not in train.columns for c in cols):
            continue
        stats = val_pred.groupby(cols, dropna=False)["is_error"].agg(["mean", "count"]).reset_index()
        stats = stats[stats["count"] >= 5].copy()
        stats["boost"] = 1.0 + alpha * scale * (stats["mean"] - overall).clip(lower=0.0)
        train = train.merge(stats[cols + ["boost"]], on=cols, how="left")
        train["sample_weight"] *= train["boost"].fillna(1.0)
        train = train.drop(columns=["boost"])
    train["sample_weight"] = train["sample_weight"].clip(1.0, max_weight)
    train["sample_weight"] = train["sample_weight"] / max(float(train["sample_weight"].mean()), 1e-8)
    train[["sequence", "binary_label", "sample_weight", "source_file", "length_bin", "charge_bin", "hydrophobic_bin"]].to_csv(
        out_path, index=False
    )
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation/test hard cases and build training weights.")
    parser.add_argument("--member-pred-dir", default="meta_acp/runs_gpu/ensemble_with_esmc_strict/member_predictions")
    parser.add_argument("--train-csv", default="train_source.csv")
    parser.add_argument("--validation-csv", default="validation_source.csv")
    parser.add_argument("--test-csv", default="test_source.csv")
    parser.add_argument("--members", default="esmc_seed42;esmc_seed2024;prott5_seed42;esm2_650m_cfg03_seed42")
    parser.add_argument("--method", default="prob_mean")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/hard_case_analysis")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=2.5)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    member_dir = Path(args.member_pred_dir)
    members = [m for m in args.members.split(";") if m]
    weights = None if not args.weights else [float(x) for x in args.weights.split(";")]

    summaries = {}
    for split, split_csv in [("validation", args.validation_csv), ("test", args.test_csv)]:
        pred, available = load_member_scores(member_dir, split)
        missing = [m for m in members if m not in available]
        if missing:
            raise ValueError(f"missing members in {split}: {missing}")
        pred["score"] = combine(pred, members, args.method, weights)
        pred = attach_split_columns(pred[["sequence", "binary_label", "score"]], Path(split_csv))
        pred = add_error_columns(pred)
        pred.to_csv(out / f"{split}_predictions_with_features.csv", index=False)
        pred[pred["is_error"]].sort_values("hardness", ascending=False).to_csv(out / f"{split}_false_cases.csv", index=False)
        group_tables = []
        for cols in [
            ["source_file", "binary_label"],
            ["length_bin", "binary_label"],
            ["charge_bin", "binary_label"],
            ["hydrophobic_bin", "binary_label"],
            ["source_file", "length_bin", "binary_label"],
        ]:
            if all(c in pred.columns for c in cols):
                tab = group_summary(pred, cols)
                tab.insert(0, "grouping", "+".join(cols))
                group_tables.append(tab)
        summary = pd.concat(group_tables, ignore_index=True)
        summary.to_csv(out / f"{split}_error_group_summary.csv", index=False)
        summaries[split] = pred

    train_weights = build_train_weights(
        Path(args.train_csv), summaries["validation"], out / "train_hard_weights.csv", args.alpha, args.max_weight
    )
    cfg = {
        "members": members,
        "method": args.method,
        "weights": weights,
        "alpha": args.alpha,
        "max_weight": args.max_weight,
        "note": "Sample weights are derived from validation error groups only; test errors are report-only.",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    lines = [
        "# Hard-Case Analysis",
        "",
        "Weights are derived from validation error groups only. Test errors are used only for reporting.",
        "",
        "## Validation Error Hotspots",
        summaries["validation"][summaries["validation"]["is_error"]]
        .groupby("error_type")
        .size()
        .to_string(),
        "",
        "## Train Weight Distribution",
        train_weights["sample_weight"].describe().to_string(),
    ]
    (out / "hard_case_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[hard_case_analysis] wrote {out}")


if __name__ == "__main__":
    main()
