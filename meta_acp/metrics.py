"""Metric helpers for ACP binary prediction and source-aware reporting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["n", "pos", "neg", "acc", "sn", "sp", "precision", "f1", "mcc", "auc", "ap"]


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    n = int(y_true.size)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    out: Dict[str, float] = {"n": n, "pos": pos, "neg": neg}
    if n == 0:
        return {**out, **{k: np.nan for k in METRIC_COLUMNS if k not in out}}

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    out["acc"] = (tp + tn) / n if pos and neg else np.nan
    out["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    out["f1"] = 2 * out["precision"] * recall / (out["precision"] + recall) if (out["precision"] + recall) else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out["mcc"] = ((tp * tn - fp * fn) / denom) if denom and pos and neg else np.nan
    if pos and neg:
        out["sn"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["sp"] = tn / (tn + fp) if (tn + fp) else np.nan
        out["auc"] = _roc_auc(y_true, y_score)
        out["ap"] = _average_precision(y_true, y_score)
    else:
        out["sn"] = float(((y_pred == 1) & (y_true == 1)).sum() / pos) if pos else np.nan
        out["sp"] = float(((y_pred == 0) & (y_true == 0)).sum() / neg) if neg else np.nan
        out["auc"] = np.nan
        out["ap"] = np.nan
    return out


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ranks for ties.
    values, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    del values
    if (counts > 1).any():
        for group in np.where(counts > 1)[0]:
            idx = np.where(inv == group)[0]
            ranks[idx] = ranks[idx].mean()
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall_step = (y == 1) / max(int((y_true == 1).sum()), 1)
    return float((precision * recall_step).sum())


def summarize_by_group(df: pd.DataFrame, group_col: str = "source_file", threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for name, part in df.groupby(group_col, dropna=False):
        m = binary_metrics(part["binary_label"], part["score"], threshold=threshold)
        m[group_col] = name
        m["mean_score"] = float(part["score"].mean())
        m["median_score"] = float(part["score"].median())
        m["false_negative"] = int(((part["binary_label"] == 1) & (part["score"] < threshold)).sum())
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=[group_col] + METRIC_COLUMNS)
    cols = [group_col] + METRIC_COLUMNS + ["mean_score", "median_score", "false_negative"]
    return pd.DataFrame(rows)[cols]


def mean_std_table(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns and c not in {"n", "pos", "neg"}]
    parts = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in ["n", "pos", "neg"]:
            if col in part:
                row[col] = int(part[col].iloc[0])
        for col in metric_cols:
            row[f"{col}_mean"] = part[col].mean()
            row[f"{col}_std"] = part[col].std(ddof=0)
        parts.append(row)
    return pd.DataFrame(parts)
