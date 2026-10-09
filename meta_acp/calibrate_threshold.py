"""Calibrate binary decision thresholds from validation predictions.

This script does not retrain a model. It reloads a finished run, selects a
threshold on the validation split, then applies the fixed threshold to test.
Ranking metrics (AUC/AP) are unchanged; threshold-dependent metrics become
interpretable when raw probabilities are poorly calibrated.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

try:
    from dataset import collate_embedding_batch, EmbeddingDataset
    from train_plm_metaacp import build_model, predict
except ImportError:  # pragma: no cover
    from .dataset import collate_embedding_batch, EmbeddingDataset
    from .train_plm_metaacp import build_model, predict

from torch.utils.data import DataLoader


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else np.nan,
        "sn": tp / (tp + fn) if (tp + fn) else np.nan,
        "sp": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan,
        "ap": average_precision_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan,
    }


def select_threshold(y_true: np.ndarray, score: np.ndarray, metric: str) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, score])
    best_t = 0.5
    best = None
    best_value = -np.inf
    for t in candidates:
        rec = threshold_metrics(y_true, score, float(t))
        value = rec.get(metric, np.nan)
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_t = float(t)
            best = rec
    if best is None:
        best = threshold_metrics(y_true, score, 0.5)
    return best_t, best


def run_one(run_dir: Path, metric: str, device: str) -> dict[str, float | str]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = Namespace(**cfg)
    args.device = device
    val_ds = EmbeddingDataset(args.validation, args.embedding_cache, args.max_len)
    test_ds = EmbeddingDataset(args.test, args.embedding_cache, args.max_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embedding_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embedding_batch)
    model = build_model(args).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y_val, s_val = predict(model, val_loader, device)
    y_test, s_test = predict(model, test_loader, device)
    threshold, val_rec = select_threshold(y_val, s_val, metric)
    test_rec = threshold_metrics(y_test, s_test, threshold)
    out = {
        "run_dir": str(run_dir),
        "selection_metric": metric,
        "selected_threshold": threshold,
        **{f"val_{k}": v for k, v in val_rec.items()},
        **{f"test_{k}": v for k, v in test_rec.items()},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-threshold calibration for finished PLM-SourceACP runs.")
    parser.add_argument("--run-dir", action="append", required=True, help="Finished run directory containing config.json and best.pt.")
    parser.add_argument("--metric", default="mcc", choices=["mcc", "f1", "acc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [run_one(Path(p), args.metric, args.device) for p in args.run_dir]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
