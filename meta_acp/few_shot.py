"""Few-shot source adaptation experiments for PLM-SourceACP."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from dataset import collate_embedding_batch, seq_hash
    from metrics import binary_metrics
    from source_data import load_provenance
    from train_plm_metaacp import build_model, compute_loss, move_batch, set_seed
except ImportError:  # pragma: no cover
    from .dataset import collate_embedding_batch, seq_hash
    from .metrics import binary_metrics
    from .source_data import load_provenance
    from .train_plm_metaacp import build_model, compute_loss, move_batch, set_seed


class FrameEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_path: str | Path, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.cache = torch.load(cache_path, map_location="cpu")
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"])
        emb = self.cache[seq_hash(seq)][: self.max_len]
        return {
            "embedding": emb,
            "label": torch.tensor(int(row["binary_label"]), dtype=torch.float32),
            "length": torch.tensor(emb.shape[0], dtype=torch.long),
        }


def loader_for(df, cache_path, max_len, batch_size, shuffle=False):
    return DataLoader(FrameEmbeddingDataset(df, cache_path, max_len), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_embedding_batch)


@torch.no_grad()
def scores_aux(model, df, args, device):
    ys, scores = [], []
    model.eval()
    for batch in loader_for(df, args.embedding_cache, args.max_len, args.batch_size):
        batch = move_batch(batch, device)
        ys.append(batch["label"].cpu().numpy())
        scores.append(model.predict_proba(batch).cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


@torch.no_grad()
def scores_proto(model, support, query, args, device):
    model.eval()
    sup_pos = support[support["binary_label"] == 1]
    sup_neg = support[support["binary_label"] == 0]
    z_pos = []
    z_neg = []
    for part, dest in [(sup_pos, z_pos), (sup_neg, z_neg)]:
        for batch in loader_for(part, args.embedding_cache, args.max_len, args.batch_size):
            dest.append(model.embed(move_batch(batch, device)))
    proto_pos = torch.cat(z_pos, dim=0).mean(dim=0)
    proto_neg = torch.cat(z_neg, dim=0).mean(dim=0)
    protos = torch.nn.functional.normalize(torch.stack([proto_neg, proto_pos], dim=0), p=2, dim=-1)
    ys, scores = [], []
    for batch in loader_for(query, args.embedding_cache, args.max_len, args.batch_size):
        batch = move_batch(batch, device)
        logits = model.prototype_logits(batch, protos)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(batch["label"].cpu().numpy())
        scores.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def finetune_model(base_model, support, args, device, mode: str):
    model = copy.deepcopy(base_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    if mode == "classifier":
        modules = [model.classifier]
    elif mode == "head":
        modules = [model.proj, model.classifier]
    else:
        raise ValueError(mode)
    params = []
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)
    opt = torch.optim.AdamW(params, lr=args.finetune_lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loader = loader_for(support, args.embedding_cache, args.max_len, args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.finetune_epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            loss, _ = compute_loss(model, batch, args, criterion)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def sample_support_query(df, k_shot: int, rng: np.random.Generator):
    pos = df[df["binary_label"] == 1]
    neg = df[df["binary_label"] == 0]
    if len(pos) <= k_shot or len(neg) <= k_shot:
        raise ValueError(f"not enough samples for k={k_shot}")
    sup_pos = pos.sample(n=k_shot, random_state=int(rng.integers(0, 2**31 - 1)))
    sup_neg = neg.sample(n=k_shot, random_state=int(rng.integers(0, 2**31 - 1)))
    support = pd.concat([sup_pos, sup_neg], ignore_index=True)
    query = df.drop(index=support.index, errors="ignore")
    query = df[~df["sequence"].isin(set(support["sequence"]))].reset_index(drop=True)
    return support.reset_index(drop=True), query


def run_few_shot(args):
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = argparse.Namespace(**ckpt["args"])
    model_args.embedding_cache = args.embedding_cache
    model_args.embedding_dim = args.embedding_dim
    base_model = build_model(model_args).to(device)
    base_model.load_state_dict(ckpt["model"])
    provenance = load_provenance(args.provenance_csv, args.master_csv)
    master = pd.read_csv(args.master_csv)
    rows = []
    for source in args.sources:
        if provenance is not None:
            target = provenance[provenance["source_file"] == source][["sequence", "binary_label"]].drop_duplicates().reset_index(drop=True)
        else:
            target = master[master["source_file"] == source][["sequence", "binary_label"]].drop_duplicates().reset_index(drop=True)
        rng = np.random.default_rng(args.seed)
        for k in args.k_shot:
            for repeat in range(args.repeats):
                support, query = sample_support_query(target, k, rng)
                for method in ["none", "prototype", "classifier", "head"]:
                    if method == "none":
                        y, s = scores_aux(base_model, query, args, device)
                    elif method == "prototype":
                        y, s = scores_proto(base_model, support, query, args, device)
                    elif method == "classifier":
                        tuned = finetune_model(base_model, support, args, device, "classifier")
                        y, s = scores_aux(tuned, query, args, device)
                    else:
                        tuned = finetune_model(base_model, support, args, device, "head")
                        y, s = scores_aux(tuned, query, args, device)
                    rows.append({"source_file": source, "k_shot": k, "repeat": repeat, "method": method, **binary_metrics(y, s)})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "few_shot_raw.csv", index=False)
    summary = df.groupby(["source_file", "k_shot", "method"]).agg(["mean", "std"])
    summary.to_csv(out / "table6_few_shot_summary.csv")
    print(df.groupby(["source_file", "k_shot", "method"])[["mcc", "ap", "auc", "f1"]].mean().to_string())


def main():
    parser = argparse.ArgumentParser(description="Run few-shot source adaptation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--master-csv", default="merged_all_cdhit08.csv")
    parser.add_argument("--provenance-csv", default=None)
    parser.add_argument("--sources", nargs="+", default=["Test1.fasta", "Test2.fasta"])
    parser.add_argument("--k-shot", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", default="meta_acp/runs/few_shot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-supcon", type=float, default=0.0)
    parser.add_argument("--lambda-proto", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_few_shot(args)


if __name__ == "__main__":
    main()
