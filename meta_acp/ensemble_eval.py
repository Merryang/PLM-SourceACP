"""Build validation-calibrated ensembles from saved PLM-SourceACP checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

try:
    from dataset import EmbeddingDataset, collate_embedding_batch
    from metrics import binary_metrics
    from source_data import attach_source
    from source_eval import evaluate_predictions
    from train_plm_metaacp import build_model, move_batch
except ImportError:  # pragma: no cover
    from .dataset import EmbeddingDataset, collate_embedding_batch
    from .metrics import binary_metrics
    from .source_data import attach_source
    from .source_eval import evaluate_predictions
    from .train_plm_metaacp import build_model, move_batch


def parse_model_specs(specs: list[str]) -> list[dict[str, str]]:
    parsed = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"model spec must be name:checkpoint:cache, got {spec}")
        parsed.append({"name": parts[0], "checkpoint": parts[1], "cache": parts[2]})
    return parsed


@torch.no_grad()
def predict_csv(checkpoint: str | Path, cache: str | Path, csv_path: str | Path, device: str, batch_size: int, max_len: int) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    args.embedding_cache = str(cache)
    args.embedding_dim = None
    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = EmbeddingDataset(csv_path, cache, max_len=max_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_embedding_batch)
    scores = []
    for batch in loader:
        batch = move_batch(batch, device)
        scores.append(model.predict_proba(batch).detach().cpu().numpy())
    return np.concatenate(scores)


def build_prediction_matrix(
    specs: list[dict[str, str]],
    csv_path: str | Path,
    out_dir: Path,
    split_name: str,
    device: str,
    batch_size: int,
    max_len: int,
    reuse: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    base = pd.read_csv(csv_path)
    matrix = []
    score_paths = []
    pred_dir = out_dir / "member_predictions" / split_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        score_path = pred_dir / f"{spec['name']}.csv"
        score_paths.append(score_path)
        if reuse and score_path.exists():
            score = pd.read_csv(score_path)["score"].to_numpy()
        else:
            score = predict_csv(spec["checkpoint"], spec["cache"], csv_path, device, batch_size, max_len)
            pd.DataFrame({"sequence": base["sequence"], "binary_label": base["binary_label"], "score": score}).to_csv(score_path, index=False)
        matrix.append(score)
    return base, np.vstack(matrix).T


def write_prediction_outputs(
    base: pd.DataFrame,
    score: np.ndarray,
    name: str,
    out_dir: Path,
    master_csv: str | Path,
    provenance_csv: str | Path | None,
    test_csv: str | Path,
) -> dict:
    pred = attach_source(test_csv, master_csv).copy()
    pred["score"] = score
    pred["prediction"] = (pred["score"] >= 0.5).astype(int)
    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_path = variant_dir / "predictions.csv"
    pred.to_csv(pred_path, index=False)
    metrics = binary_metrics(pred["binary_label"], pred["score"])
    pd.DataFrame([metrics]).to_csv(variant_dir / "test_metrics.csv", index=False)
    evaluate_predictions(pred_path, test_csv, master_csv, provenance_csv, variant_dir / "source_eval")
    return metrics


def _clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1.0 - 1e-6)


def logit_mean(matrix: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    logits = np.log(_clip_prob(matrix[:, idx]) / (1.0 - _clip_prob(matrix[:, idx])))
    return 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))


def prob_mean(matrix: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    return matrix[:, idx].mean(axis=1)


def rank_mean(matrix: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    ranks = []
    for i in idx:
        ranks.append(pd.Series(matrix[:, i]).rank(method="average", pct=True).to_numpy())
    return np.vstack(ranks).T.mean(axis=1)


def temperature_scaled_mean(val_matrix: np.ndarray, test_matrix: np.ndarray, y_val: np.ndarray, idx: tuple[int, ...]) -> tuple[np.ndarray, float]:
    best_t = 1.0
    best_loss = float("inf")
    val_logits = np.log(_clip_prob(val_matrix[:, idx]) / (1.0 - _clip_prob(val_matrix[:, idx])))
    for t in np.linspace(0.5, 5.0, 46):
        p = 1.0 / (1.0 + np.exp(-(val_logits / t).mean(axis=1)))
        loss = -np.mean(y_val * np.log(_clip_prob(p)) + (1 - y_val) * np.log(_clip_prob(1 - p)))
        if loss < best_loss:
            best_loss = float(loss)
            best_t = float(t)
    test_logits = np.log(_clip_prob(test_matrix[:, idx]) / (1.0 - _clip_prob(test_matrix[:, idx])))
    score = 1.0 / (1.0 + np.exp(-(test_logits / best_t).mean(axis=1)))
    return score, best_t


def subset_iter(n_members: int, max_subset_size: int) -> list[tuple[int, ...]]:
    max_k = min(max_subset_size, n_members)
    out = []
    for k in range(1, max_k + 1):
        out.extend(itertools.combinations(range(n_members), k))
    return out


def evaluate_validation_selected_ensembles(
    specs: list[dict[str, str]],
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    max_subset_size: int,
) -> pd.DataFrame:
    rows = []
    methods = {
        "prob_mean": prob_mean,
        "logit_mean": logit_mean,
        "rank_mean": rank_mean,
    }
    for idx in subset_iter(x_val.shape[1], max_subset_size):
        members = ";".join(specs[i]["name"] for i in idx)
        for method, fn in methods.items():
            val_score = fn(x_val, idx)
            test_score = fn(x_test, idx)
            val_metrics = binary_metrics(y_val, val_score)
            test_metrics = binary_metrics(y_test, test_score)
            rows.append(
                {
                    "method": method,
                    "members": members,
                    "k": len(idx),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )
        test_score, temp = temperature_scaled_mean(x_val, x_test, y_val, idx)
        val_logits = np.log(_clip_prob(x_val[:, idx]) / (1.0 - _clip_prob(x_val[:, idx])))
        val_score = 1.0 / (1.0 + np.exp(-(val_logits / temp).mean(axis=1)))
        val_metrics = binary_metrics(y_val, val_score)
        test_metrics = binary_metrics(y_test, test_score)
        rows.append(
            {
                "method": "temperature_logit_mean",
                "members": members,
                "k": len(idx),
                "temperature": temp,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def score_for_selection_row(row: pd.Series, specs: list[dict[str, str]], x_val: np.ndarray, x_test: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    names = str(row["members"]).split(";")
    index_by_name = {spec["name"]: i for i, spec in enumerate(specs)}
    idx = tuple(index_by_name[name] for name in names)
    method = row["method"]
    if method == "prob_mean":
        return prob_mean(x_test, idx)
    if method == "logit_mean":
        return logit_mean(x_test, idx)
    if method == "rank_mean":
        return rank_mean(x_test, idx)
    if method == "temperature_logit_mean":
        score, _ = temperature_scaled_mean(x_val, x_test, y_val, idx)
        return score
    raise ValueError(f"unsupported ensemble method {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation-calibrated PLM ensembles.")
    parser.add_argument("--models", nargs="+", required=True, help="name:checkpoint:embedding_cache entries")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--master-csv", default="merged_all_cdhit08.csv")
    parser.add_argument("--provenance-csv", default=None)
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/ensemble")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--reuse-member-preds", action="store_true")
    parser.add_argument("--max-subset-size", type=int, default=5)
    parser.add_argument("--selection-metric", default="auc", choices=["auc", "ap", "mcc"])
    args = parser.parse_args()

    specs = parse_model_specs(args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_base, x_val = build_prediction_matrix(specs, args.validation, out_dir, "validation", device, args.batch_size, args.max_len, args.reuse_member_preds)
    test_base, x_test = build_prediction_matrix(specs, args.test, out_dir, "test", device, args.batch_size, args.max_len, args.reuse_member_preds)
    y_val = val_base["binary_label"].astype(int).to_numpy()
    y_test = test_base["binary_label"].astype(int).to_numpy()

    rows = []
    variants: dict[str, np.ndarray] = {}
    variants["mean_all"] = x_test.mean(axis=1)
    top_idx = np.argsort([binary_metrics(y_val, x_val[:, i])["auc"] for i in range(x_val.shape[1])])[::-1][: min(5, x_val.shape[1])]
    variants["mean_top_val_auc"] = x_test[:, top_idx].mean(axis=1)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
    lr.fit(x_val, y_val)
    variants["logistic_stack"] = lr.predict_proba(x_test)[:, 1]

    for name, score in variants.items():
        metrics = write_prediction_outputs(test_base, score, name, out_dir, args.master_csv, args.provenance_csv, args.test)
        rows.append({"ensemble": name, **metrics})

    search = evaluate_validation_selected_ensembles(specs, x_val, x_test, y_val, y_test, args.max_subset_size)
    val_sort_cols = [f"val_{args.selection_metric}", "val_ap", "val_mcc", "test_auc"]
    search_by_validation = search.sort_values(val_sort_cols, ascending=False).reset_index(drop=True)
    search_by_validation.to_csv(out_dir / "strict_ensemble_search_by_validation.csv", index=False)
    search.sort_values(["test_auc", "test_ap", "test_mcc"], ascending=False).to_csv(
        out_dir / "diagnostic_ensemble_search_by_test_do_not_select.csv", index=False
    )

    best_rows = []
    for metric in ["auc", "ap", "mcc"]:
        best = search.sort_values([f"val_{metric}", "val_ap", "val_mcc"], ascending=False).iloc[0]
        score = score_for_selection_row(best, specs, x_val, x_test, y_val)
        out_name = f"strict_best_val_{metric}"
        metrics = write_prediction_outputs(test_base, score, out_name, out_dir, args.master_csv, args.provenance_csv, args.test)
        row = {"ensemble": out_name, "selection_metric": metric, "method": best["method"], "members": best["members"], **metrics}
        rows.append(row)
        best_rows.append(row)
    pd.DataFrame(best_rows).to_csv(out_dir / "strict_best_selected_ensembles.csv", index=False)

    lr_rows = []
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        stacker = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=c)
        stacker.fit(x_val, y_val)
        val_score = stacker.predict_proba(x_val)[:, 1]
        test_score = stacker.predict_proba(x_test)[:, 1]
        lr_rows.append(
            {
                "method": "regularized_logistic_stack",
                "C": c,
                **{f"val_{k}": v for k, v in binary_metrics(y_val, val_score).items()},
                **{f"test_{k}": v for k, v in binary_metrics(y_test, test_score).items()},
            }
        )
    lr_search = pd.DataFrame(lr_rows).sort_values(f"val_{args.selection_metric}", ascending=False).reset_index(drop=True)
    lr_search.to_csv(out_dir / "strict_logistic_stack_search_by_validation.csv", index=False)
    best_lr = lr_search.iloc[0]
    stacker = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=float(best_lr["C"]))
    stacker.fit(x_val, y_val)
    score = stacker.predict_proba(x_test)[:, 1]
    metrics = write_prediction_outputs(test_base, score, "strict_logistic_stack", out_dir, args.master_csv, args.provenance_csv, args.test)
    rows.append({"ensemble": "strict_logistic_stack", "selection_metric": args.selection_metric, "C": float(best_lr["C"]), **metrics})

    member_rows = []
    for i, spec in enumerate(specs):
        member_rows.append({"member": spec["name"], **binary_metrics(y_test, x_test[:, i]), "val_auc": binary_metrics(y_val, x_val[:, i])["auc"]})
    pd.DataFrame(member_rows).to_csv(out_dir / "member_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "ensemble_metrics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
