"""Train PLM-SourceACP or scratch baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from dataset import EmbeddingDataset, SequenceDataset, collate_embedding_batch, collate_token_batch
    from encoders import GenericPLMFrozenEncoder, ScratchEncoder
    from losses import FocalBinaryLoss
    from metrics import binary_metrics
    from proto_model import PrototypicalNetwork
    from source_data import attach_source
except ImportError:  # pragma: no cover
    from .dataset import EmbeddingDataset, SequenceDataset, collate_embedding_batch, collate_token_batch
    from .encoders import GenericPLMFrozenEncoder, ScratchEncoder
    from .losses import FocalBinaryLoss
    from .metrics import binary_metrics
    from .proto_model import PrototypicalNetwork
    from .source_data import attach_source


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def infer_cache_dim(cache_path: str | Path) -> int:
    cache = torch.load(cache_path, map_location="cpu")
    first = next(iter(cache.values()))
    return int(first.shape[-1])


def apply_sample_weights(dataset, args) -> None:
    weight_csv = getattr(args, "sample_weight_csv", None)
    if not weight_csv:
        return
    df = pd.read_csv(weight_csv)
    sequence_col = getattr(args, "sample_weight_sequence_col", "sequence")
    weight_col = getattr(args, "sample_weight_col", "sample_weight")
    if sequence_col not in df.columns or weight_col not in df.columns:
        raise ValueError(f"{weight_csv} must contain {sequence_col!r} and {weight_col!r}")
    weights = (
        df[[sequence_col, weight_col]]
        .dropna()
        .assign(**{sequence_col: lambda x: x[sequence_col].astype(str)})
        .drop_duplicates(sequence_col, keep="last")
    )
    merged = dataset.df.merge(weights, left_on="sequence", right_on=sequence_col, how="left")
    raw = merged[weight_col].fillna(1.0).astype(float).clip(args.sample_weight_min, args.sample_weight_max)
    if args.sample_weight_normalize:
        raw = raw / max(float(raw.mean()), 1e-8)
    dataset.df["sample_weight"] = raw.to_numpy(dtype=np.float32)
    print(
        "[sample_weight] loaded "
        f"{weight_csv} min={dataset.df['sample_weight'].min():.3f} "
        f"mean={dataset.df['sample_weight'].mean():.3f} "
        f"max={dataset.df['sample_weight'].max():.3f}",
        flush=True,
    )


def build_loaders(args):
    if args.encoder == "scratch":
        train_ds = SequenceDataset(args.train, args.max_len)
        val_ds = SequenceDataset(args.validation, args.max_len)
        test_ds = SequenceDataset(args.test, args.max_len)
        collate = collate_token_batch
    else:
        if not args.embedding_cache:
            raise ValueError("--embedding-cache is required for PLM encoders")
        train_ds = EmbeddingDataset(args.train, args.embedding_cache, args.max_len)
        val_ds = EmbeddingDataset(args.validation, args.embedding_cache, args.max_len)
        test_ds = EmbeddingDataset(args.test, args.embedding_cache, args.max_len)
        collate = collate_embedding_batch
    apply_sample_weights(train_ds, args)
    sampler = None
    shuffle = True
    if getattr(args, "balanced_sampler", "none") != "none":
        df = train_ds.df.copy()
        if args.balanced_sampler == "source_class" and "source_file" in df.columns:
            groups = df["source_file"].astype(str) + "::" + df["binary_label"].astype(str)
        else:
            groups = df["binary_label"].astype(str)
        counts = groups.value_counts().to_dict()
        weights = groups.map(lambda g: 1.0 / counts[g]).to_numpy(dtype=np.float64)
        if getattr(args, "sampler_use_sample_weight", False) and "sample_weight" in df.columns:
            weights = weights * df["sample_weight"].astype(float).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


def build_model(args) -> PrototypicalNetwork:
    if args.encoder == "scratch":
        encoder = ScratchEncoder(dropout=args.dropout)
    else:
        in_dim = args.embedding_dim or infer_cache_dim(args.embedding_cache)
        encoder = GenericPLMFrozenEncoder(
            in_dim=in_dim,
            cnn_channels=args.cnn_channels,
            dropout=args.dropout,
            use_multiscale=not args.no_multiscale_cnn,
            use_se=not args.no_se,
            pooling=args.pooling,
        )
    return PrototypicalNetwork(encoder=encoder, proj_dim=args.proj_dim, dropout=args.dropout)


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    labels = labels.long()
    n = z.shape[0]
    if n <= 1:
        return z.sum() * 0.0
    sim = z @ z.t() / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)
    if same.sum() == 0:
        return z.sum() * 0.0
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = same.sum(dim=1).clamp(min=1)
    loss = -(log_prob * same.float()).sum(dim=1) / denom
    return loss[same.any(dim=1)].mean()


def batch_proto_loss(z: torch.Tensor, labels: torch.Tensor, log_tau: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if labels.unique().numel() < 2:
        return z.sum() * 0.0
    proto0 = F.normalize(z[labels == 0].mean(dim=0), p=2, dim=0)
    proto1 = F.normalize(z[labels == 1].mean(dim=0), p=2, dim=0)
    protos = torch.stack([proto0, proto1], dim=0)
    logits = log_tau.exp().clamp(min=1e-3, max=100.0) * z @ protos.t()
    return F.cross_entropy(logits, labels)


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return loss.mean()
    weights = weights.float().clamp(min=0.0)
    return (loss * weights).sum() / weights.sum().clamp(min=1e-8)


def compute_loss(model: PrototypicalNetwork, batch: dict, args, criterion) -> tuple[torch.Tensor, dict]:
    labels = batch["label"].float()
    z = model.embed(batch)
    logits = model.classifier(z).squeeze(-1)
    cls_raw = criterion(logits, labels)
    cls_loss = weighted_mean(cls_raw, batch.get("sample_weight"))
    supcon = supervised_contrastive_loss(z, labels, args.supcon_temperature) if args.lambda_supcon else z.sum() * 0.0
    proto = batch_proto_loss(z, labels, model.log_tau) if args.lambda_proto else z.sum() * 0.0
    loss = cls_loss + args.lambda_supcon * supcon + args.lambda_proto * proto
    return loss, {"cls": float(cls_loss.detach()), "supcon": float(supcon.detach()), "proto": float(proto.detach())}


@torch.no_grad()
def predict(model: PrototypicalNetwork, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        score = model.predict_proba(batch)
        ys.append(batch["label"].detach().cpu().numpy())
        scores.append(score.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def train_one(args) -> dict:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(args)
    model = build_model(args).to(device)
    if getattr(args, "freeze_encoder", False):
        for param in model.encoder.parameters():
            param.requires_grad = False
    if getattr(args, "freeze_projection", False):
        for param in model.proj.parameters():
            param.requires_grad = False
    if args.loss == "focal":
        criterion = FocalBinaryLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="none")
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("all model parameters are frozen")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = {"monitor": -2.0, "epoch": -1}
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            loss, parts = compute_loss(model, batch, args, criterion)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach()))
        y_val, s_val = predict(model, val_loader, device)
        val_metrics = binary_metrics(y_val, s_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} val_mcc={val_metrics['mcc']:.4f} val_ap={val_metrics['ap']:.4f}")
        raw_monitor = val_metrics.get(args.monitor_metric, np.nan)
        monitor = raw_monitor if np.isfinite(raw_monitor) else -1.0
        if monitor > best["monitor"]:
            best = {"monitor": monitor, "epoch": epoch, "metric": args.monitor_metric}
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, out / "best.pt")
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y_test, s_test = predict(model, test_loader, device)
    test_metrics = binary_metrics(y_test, s_test)
    pred = pd.read_csv(args.test).copy()
    pred["score"] = s_test
    pred["prediction"] = (pred["score"] >= 0.5).astype(int)
    if args.master_csv:
        try:
            pred = attach_source(args.test, args.master_csv).assign(score=s_test, prediction=(s_test >= 0.5).astype(int))
        except Exception as exc:
            print(f"[WARN] could not attach source columns: {exc}")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(out / "test_metrics.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(pd.DataFrame([test_metrics]).to_string(index=False))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLM-SourceACP.")
    parser.add_argument("--train", default="data/splits/train.csv")
    parser.add_argument("--validation", default="data/splits/validation.csv")
    parser.add_argument("--test", default="data/splits/test.csv")
    parser.add_argument("--master-csv", default="data/processed/merged_all_cdhit08.csv")
    parser.add_argument("--encoder", default="scratch", choices=["scratch", "plm"])
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--out-dir", default="meta_acp/runs/plm_metaacp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--pooling", default="attn_mean_max", choices=["attn_mean_max", "mean"])
    parser.add_argument("--no-multiscale-cnn", action="store_true")
    parser.add_argument("--no-se", action="store_true")
    parser.add_argument("--loss", default="focal", choices=["focal", "bce"])
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lambda-supcon", type=float, default=0.1)
    parser.add_argument("--lambda-proto", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--monitor-metric", default="mcc", choices=["mcc", "auc", "ap", "f1", "acc"])
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-projection", action="store_true")
    parser.add_argument("--balanced-sampler", default="none", choices=["none", "class", "source_class"])
    parser.add_argument("--sampler-use-sample-weight", action="store_true")
    parser.add_argument("--sample-weight-csv", default=None)
    parser.add_argument("--sample-weight-col", default="sample_weight")
    parser.add_argument("--sample-weight-sequence-col", default="sequence")
    parser.add_argument("--sample-weight-min", type=float, default=0.25)
    parser.add_argument("--sample-weight-max", type=float, default=3.0)
    parser.add_argument("--sample-weight-normalize", action="store_true")
    args = parser.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
