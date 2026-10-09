"""Dataset utilities for the meta_acp pipeline.

Reads Doubao-format CSVs (columns: source_file, source_id, raw_label,
binary_label, sequence, length, split_group), produces:

- `SequenceDataset` for the auxiliary supervised BCE head (token id batches).
- `EmbeddingDataset` for ProtT5 cached residue-level embeddings.
- `BalancedEpisodicSampler` that yields (support, query) index tuples with
  exactly K_shot + Q_query samples per class, the central trick that lets
  Prototypical Networks ignore base-rate class imbalance.
- `cache_prott5_embeddings` that runs ProtT5 once over all sequences in a
  CSV and dumps a `.pt` file keyed by sequence hash for downstream reuse.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


AA_VOCAB: Dict[str, int] = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
PAD_IDX = 0
VOCAB_SIZE = len(AA_VOCAB) + 1  # 21


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    arr = np.zeros((max_len,), dtype=np.int64)
    seq = (seq or "")[:max_len]
    for i, ch in enumerate(seq):
        arr[i] = AA_VOCAB.get(ch, 0)
    return arr


def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


class SequenceDataset(Dataset):
    """AA-token dataset for the Scratch encoder."""

    def __init__(self, csv_path: str, max_len: int = 256):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        seq = str(row["sequence"])
        x = encode_sequence(seq, self.max_len)
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "label": torch.tensor(int(row["binary_label"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "length": torch.tensor(min(len(seq), self.max_len), dtype=torch.long),
            "seq_hash": seq_hash(seq),
        }


class EmbeddingDataset(Dataset):
    """Wraps a CSV + a hash->embedding tensor cache produced by ProtT5.

    Each item returns the residue-level tensor (L, 1024). The collate fn
    pads to the batch max length.
    """

    def __init__(self, csv_path: str, cache_path: str, max_len: int = 256):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.max_len = max_len
        self.cache: Dict[str, torch.Tensor] = torch.load(cache_path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        seq = str(row["sequence"])
        h = seq_hash(seq)
        emb = self.cache[h][: self.max_len]  # (L, 1024)
        return {
            "embedding": emb,
            "label": torch.tensor(int(row["binary_label"]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "length": torch.tensor(emb.shape[0], dtype=torch.long),
            "seq_hash": h,
        }


def collate_token_batch(items: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in items], dim=0),
        "label": torch.stack([x["label"] for x in items], dim=0),
        "sample_weight": torch.stack([x["sample_weight"] for x in items], dim=0),
        "length": torch.stack([x["length"] for x in items], dim=0),
    }


def collate_embedding_batch(items: List[dict]) -> dict:
    lens = [x["embedding"].shape[0] for x in items]
    max_l = max(lens)
    dim = items[0]["embedding"].shape[1]
    pad = torch.zeros(len(items), max_l, dim, dtype=items[0]["embedding"].dtype)
    mask = torch.zeros(len(items), max_l, dtype=torch.bool)
    for i, x in enumerate(items):
        L = x["embedding"].shape[0]
        pad[i, :L] = x["embedding"]
        mask[i, :L] = True
    return {
        "embedding": pad,
        "mask": mask,
        "label": torch.stack([x["label"] for x in items], dim=0),
        "sample_weight": torch.stack([x["sample_weight"] for x in items], dim=0),
        "length": torch.tensor(lens, dtype=torch.long),
    }


class BalancedEpisodicSampler:
    """Yields (support_idx, query_idx) with K_shot+Q_query per class.

    Falls back to with-replacement sampling and warns if a class is
    smaller than K+Q (defends against future skewed splits).
    """

    def __init__(self, df: pd.DataFrame, k_shot: int = 5, q_query: int = 10, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.k = k_shot
        self.q = q_query
        self.rng = np.random.default_rng(seed)
        self.pos_idx = np.where(self.df["binary_label"].values == 1)[0]
        self.neg_idx = np.where(self.df["binary_label"].values == 0)[0]
        need = k_shot + q_query
        self.replace_pos = len(self.pos_idx) < need
        self.replace_neg = len(self.neg_idx) < need
        if self.replace_pos or self.replace_neg:
            print(
                f"[BalancedEpisodicSampler] WARN: class too small for K+Q={need}, "
                f"using with-replacement (pos={len(self.pos_idx)}, neg={len(self.neg_idx)})"
            )

    def sample_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (s_pos, s_neg, q_pos, q_neg) index arrays."""
        pos = self.rng.choice(self.pos_idx, size=self.k + self.q, replace=self.replace_pos)
        neg = self.rng.choice(self.neg_idx, size=self.k + self.q, replace=self.replace_neg)
        return pos[: self.k], neg[: self.k], pos[self.k :], neg[self.k :]


# ---------------------------------------------------------------------------
# ProtT5 caching
# ---------------------------------------------------------------------------


def cache_prott5_embeddings(
    csv_paths: List[str],
    cache_path: str,
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    max_len: int = 256,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> str:
    """Compute residue-level ProtT5 embeddings for every unique sequence in the
    given CSVs and dump them as `{seq_hash: tensor(L, 1024)}`.

    First call downloads ~5 GB of weights from HuggingFace; subsequent runs
    reuse the local HF cache.
    """
    cache_path = str(cache_path)
    if Path(cache_path).exists():
        print(f"[cache_prott5] reusing existing cache {cache_path}")
        return cache_path

    from transformers import T5EncoderModel, T5Tokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache_prott5] loading {model_name} on {device}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    # Collect unique sequences across all splits.
    unique: Dict[str, str] = {}
    for p in csv_paths:
        df = pd.read_csv(p)
        for s in df["sequence"].astype(str).tolist():
            unique[seq_hash(s)] = s
    seqs = list(unique.items())
    print(f"[cache_prott5] {len(seqs)} unique sequences to embed")

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="prott5"):
            batch = seqs[i : i + batch_size]
            spaced = [" ".join(list(s[:max_len])) for _, s in batch]
            enc = tokenizer.batch_encode_plus(
                spaced,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state  # (B, L, 1024)
            for j, (k, s) in enumerate(batch):
                L = min(len(s), max_len)
                # Drop the appended </s> token; keep first L residue vectors.
                out[k] = h[j, :L].detach().cpu().clone()

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    print(f"[cache_prott5] wrote {len(out)} embeddings -> {cache_path}")
    return cache_path
