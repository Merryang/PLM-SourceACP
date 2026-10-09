"""Run external-task transfer experiments for PLM-SourceACP.

Inputs are prepared by `prepare_external_task.py` and use the same
sequence,binary_label CSV contract as the ACP task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

try:
    from dataset import EmbeddingDataset, collate_embedding_batch
    from metrics import binary_metrics
    from train_plm_metaacp import build_model, move_batch, train_one
except ImportError:  # pragma: no cover
    from .dataset import EmbeddingDataset, collate_embedding_batch
    from .metrics import binary_metrics
    from .train_plm_metaacp import build_model, move_batch, train_one


def mean_pool_features(csv_path: str | Path, cache_path: str | Path, max_len: int) -> tuple[pd.DataFrame, np.ndarray]:
    ds = EmbeddingDataset(csv_path, cache_path, max_len=max_len)
    rows = pd.read_csv(csv_path).reset_index(drop=True)
    features = []
    for i in range(len(ds)):
        emb = ds[i]["embedding"].float()
        features.append(emb.mean(dim=0).numpy())
    return rows, np.vstack(features)


@torch.no_grad()
def checkpoint_scores(
    checkpoint: str | Path,
    csv_path: str | Path,
    cache_path: str | Path,
    device: str,
    batch_size: int,
    max_len: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_args = argparse.Namespace(**ckpt["args"])
    model_args.embedding_cache = str(cache_path)
    model_args.embedding_dim = None
    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = EmbeddingDataset(csv_path, cache_path, max_len=max_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_embedding_batch)
    ys, scores, z_all = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        z = model.embed(batch)
        score = model.predict_proba(batch)
        ys.append(batch["label"].detach().cpu().numpy())
        scores.append(score.detach().cpu().numpy())
        z_all.append(z.detach().cpu().numpy())
    return pd.read_csv(csv_path).reset_index(drop=True), np.concatenate(ys), np.concatenate(scores), np.vstack(z_all)


def write_predictions(out_dir: Path, name: str, df: pd.DataFrame, score: np.ndarray) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = df.copy()
    pred["score"] = score
    pred["prediction"] = (pred["score"] >= 0.5).astype(int)
    metrics = binary_metrics(pred["binary_label"], pred["score"])
    pred.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    pd.DataFrame([{"method": name, **metrics}]).to_csv(out_dir / f"{name}_metrics.csv", index=False)
    return {"method": name, **metrics}


def run_zero_shot(args) -> list[dict]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    df, _, score, _ = checkpoint_scores(args.checkpoint, args.test, args.embedding_cache, device, args.batch_size, args.max_len)
    return [write_predictions(Path(args.out_dir), "zero_shot", df, score)]


def run_linear_probe(args) -> list[dict]:
    train_df, x_train = mean_pool_features(args.train, args.embedding_cache, args.max_len)
    test_df, x_test = mean_pool_features(args.test, args.embedding_cache, args.max_len)
    y_train = train_df["binary_label"].astype(int).to_numpy()
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=args.linear_c)
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_test)[:, 1]
    return [write_predictions(Path(args.out_dir), "linear_probe_mean_embedding", test_df, score)]


def run_full_head(args) -> list[dict]:
    out_dir = Path(args.out_dir) / "full_head"
    train_args = argparse.Namespace(
        train=args.train,
        validation=args.validation,
        test=args.test,
        master_csv=None,
        encoder="plm",
        embedding_cache=args.embedding_cache,
        embedding_dim=None,
        out_dir=str(out_dir),
        seed=args.seed,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=5.0,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
        proj_dim=args.proj_dim,
        pooling=args.pooling,
        no_multiscale_cnn=False,
        no_se=False,
        loss=args.loss,
        focal_alpha=0.25,
        focal_gamma=args.focal_gamma,
        lambda_supcon=args.lambda_supcon,
        lambda_proto=args.lambda_proto,
        supcon_temperature=0.1,
        device=args.device,
        monitor_metric=args.monitor_metric,
        freeze_encoder=False,
        freeze_projection=False,
        balanced_sampler=args.balanced_sampler,
    )
    metrics = train_one(train_args)
    return [{"method": "full_head", **metrics}]


def prototype_scores(z_support: np.ndarray, y_support: np.ndarray, z_query: np.ndarray) -> np.ndarray:
    proto0 = z_support[y_support == 0].mean(axis=0)
    proto1 = z_support[y_support == 1].mean(axis=0)
    proto0 = proto0 / max(np.linalg.norm(proto0), 1e-8)
    proto1 = proto1 / max(np.linalg.norm(proto1), 1e-8)
    logits = np.vstack([z_query @ proto0, z_query @ proto1]).T
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def run_few_shot(args) -> list[dict]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for few_shot")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_df, y_train, base_train, z_train = checkpoint_scores(args.checkpoint, args.train, args.embedding_cache, device, args.batch_size, args.max_len)
    test_df, y_test, base_test, z_test = checkpoint_scores(args.checkpoint, args.test, args.embedding_cache, device, args.batch_size, args.max_len)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    out = Path(args.out_dir)
    rows = []
    raw_rows = []
    rng = np.random.default_rng(args.seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    for k in args.k_shots:
        for repeat in range(args.repeats):
            support_pos = rng.choice(pos_idx, size=k, replace=len(pos_idx) < k)
            support_neg = rng.choice(neg_idx, size=k, replace=len(neg_idx) < k)
            support = np.concatenate([support_pos, support_neg])
            y_support = y_train[support]
            methods = {
                "no_adaptation": base_test,
                "prototype_adaptation": prototype_scores(z_train[support], y_support, z_test),
            }
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
            clf.fit(z_train[support], y_support)
            methods["support_logistic"] = clf.predict_proba(z_test)[:, 1]
            for method, score in methods.items():
                m = binary_metrics(y_test, score)
                row = {"k_shot": k, "repeat": repeat, "method": method, **m}
                raw_rows.append(row)
        part = pd.DataFrame([r for r in raw_rows if r["k_shot"] == k])
        for method, g in part.groupby("method"):
            summary = {"k_shot": k, "method": method}
            for col in ["acc", "sn", "sp", "precision", "f1", "mcc", "auc", "ap"]:
                summary[f"{col}_mean"] = float(g[col].mean())
                summary[f"{col}_std"] = float(g[col].std(ddof=0))
            rows.append(summary)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(out / "few_shot_raw.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "few_shot_summary.csv", index=False)
    write_predictions(out, "zero_shot_reference", test_df, base_test)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external transfer experiments.")
    parser.add_argument("--mode", required=True, choices=["zero_shot", "linear_probe", "full_head", "few_shot"])
    parser.add_argument("--train", default=None)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--test", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--pooling", default="attn_mean_max", choices=["attn_mean_max", "mean"])
    parser.add_argument("--loss", default="focal", choices=["focal", "bce"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lambda-supcon", type=float, default=0.1)
    parser.add_argument("--lambda-proto", type=float, default=0.1)
    parser.add_argument("--monitor-metric", default="auc", choices=["mcc", "auc", "ap", "f1", "acc"])
    parser.add_argument("--balanced-sampler", default="none", choices=["none", "class", "source_class"])
    parser.add_argument("--k-shots", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if args.mode in {"linear_probe", "full_head", "few_shot"} and not args.train:
        raise ValueError(f"--train is required for mode {args.mode}")
    if args.mode == "full_head" and not args.validation:
        raise ValueError("--validation is required for full_head")
    if args.mode == "zero_shot" and not args.checkpoint:
        raise ValueError("--checkpoint is required for zero_shot")

    if args.mode == "zero_shot":
        rows = run_zero_shot(args)
    elif args.mode == "linear_probe":
        rows = run_linear_probe(args)
    elif args.mode == "full_head":
        rows = run_full_head(args)
    else:
        rows = run_few_shot(args)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / f"{args.mode}_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
