"""Traditional ML baselines for ACP prediction.

Implements AAC/DPC/CKSAAP feature extraction with SVM and LightGBM/GBM
classifiers. LightGBM is optional; sklearn GradientBoosting is used when the
package is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from metrics import binary_metrics
except ImportError:  # pragma: no cover
    from .metrics import binary_metrics


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {aa: i for i, aa in enumerate(AA)}


def aac(seq: str) -> np.ndarray:
    seq = str(seq)
    x = np.zeros(len(AA), dtype=np.float32)
    for ch in seq:
        if ch in AA_TO_I:
            x[AA_TO_I[ch]] += 1.0
    denom = max(float(x.sum()), 1.0)
    return x / denom


def dpc(seq: str) -> np.ndarray:
    seq = str(seq)
    x = np.zeros(len(AA) * len(AA), dtype=np.float32)
    total = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if a in AA_TO_I and b in AA_TO_I:
            x[AA_TO_I[a] * len(AA) + AA_TO_I[b]] += 1.0
            total += 1
    return x / max(float(total), 1.0)


def cksaap(seq: str, max_gap: int = 2) -> np.ndarray:
    seq = str(seq)
    parts = []
    for gap in range(max_gap + 1):
        x = np.zeros(len(AA) * len(AA), dtype=np.float32)
        total = 0
        step = gap + 1
        for i in range(0, len(seq) - step):
            a, b = seq[i], seq[i + step]
            if a in AA_TO_I and b in AA_TO_I:
                x[AA_TO_I[a] * len(AA) + AA_TO_I[b]] += 1.0
                total += 1
        parts.append(x / max(float(total), 1.0))
    return np.concatenate(parts, axis=0)


def make_features(df: pd.DataFrame, kind: str) -> np.ndarray:
    kind = kind.lower()
    if kind == "aac":
        rows = [aac(s) for s in df["sequence"]]
    elif kind == "dpc":
        rows = [dpc(s) for s in df["sequence"]]
    elif kind == "cksaap":
        rows = [cksaap(s) for s in df["sequence"]]
    elif kind == "aac_cksaap":
        rows = [np.concatenate([aac(s), cksaap(s)]) for s in df["sequence"]]
    else:
        raise ValueError(f"unknown feature kind: {kind}")
    return np.stack(rows, axis=0)


def build_classifier(name: str, seed: int):
    name = name.lower()
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=4.0, kernel="rbf", probability=True, random_state=seed))
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
            )
        except Exception:
            return GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, random_state=seed)
    raise ValueError(f"unknown classifier: {name}")


def run_baseline(
    train_csv: str,
    test_csv: str,
    val_csv: str | None = None,
    feature: str = "aac",
    classifier: str = "svm",
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    train = pd.read_csv(train_csv)
    if val_csv:
        train = pd.concat([train, pd.read_csv(val_csv)], ignore_index=True)
    test = pd.read_csv(test_csv)
    x_train = make_features(train, feature)
    y_train = train["binary_label"].astype(int).values
    x_test = make_features(test, feature)
    clf = build_classifier(classifier, seed)
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(test["binary_label"], score)
    pred = test.copy()
    pred["score"] = score
    pred["prediction"] = (pred["score"] >= 0.5).astype(int)
    return metrics, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AAC/DPC/CKSAAP ML baselines.")
    parser.add_argument("--train", default="data/splits/train.csv")
    parser.add_argument("--validation", default="data/splits/validation.csv")
    parser.add_argument("--test", default="data/splits/test.csv")
    parser.add_argument("--feature", default="aac", choices=["aac", "dpc", "cksaap", "aac_cksaap"])
    parser.add_argument("--classifier", default="svm", choices=["svm", "lightgbm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="meta_acp/runs/baselines")
    args = parser.parse_args()

    out = Path(args.out_dir) / f"{args.feature}_{args.classifier}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    metrics, pred = run_baseline(args.train, args.test, args.validation, args.feature, args.classifier, args.seed)
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
