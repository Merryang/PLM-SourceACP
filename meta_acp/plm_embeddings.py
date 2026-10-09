"""Cache residue-level embeddings from frozen protein language models."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

try:
    from dataset import seq_hash
except ImportError:  # pragma: no cover
    from .dataset import seq_hash


@dataclass(frozen=True)
class PLMConfig:
    hf_name: str
    spaced: bool = False
    trust_remote_code: bool = False


PLM_REGISTRY = {
    "prott5": PLMConfig("Rostlab/prot_t5_xl_uniref50", spaced=True),
    "protbert": PLMConfig("Rostlab/prot_bert", spaced=True),
    "esm2_35m": PLMConfig("facebook/esm2_t12_35M_UR50D"),
    "esm2": PLMConfig("facebook/esm2_t33_650M_UR50D"),
    "esm2_150m": PLMConfig("facebook/esm2_t30_150M_UR50D"),
    "esmc": PLMConfig("EvolutionaryScale/esmc-300m-2024-12", trust_remote_code=True),
    "esmc_600m": PLMConfig("EvolutionaryScale/esmc-600m-2024-12", trust_remote_code=True),
    "ankh": PLMConfig("ElnaggarLab/ankh-base"),
}


def normalize_sequence(seq: str) -> str:
    return re.sub(r"[UZOB]", "X", str(seq).upper())


def format_sequence(seq: str, spaced: bool) -> str:
    seq = normalize_sequence(seq)
    return " ".join(seq) if spaced else seq


def collect_unique_sequences(csv_paths: list[str | Path]) -> dict[str, str]:
    unique = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        for seq in df["sequence"].astype(str):
            unique[seq_hash(seq)] = seq
    return unique


def cache_embeddings(
    csv_paths: list[str | Path],
    cache_path: str | Path,
    model_key: str = "esm2_150m",
    model_name: str | None = None,
    batch_size: int = 4,
    max_len: int = 256,
    device: str | None = None,
    dtype: str = "float16",
    save_every: int = 25,
    limit: int | None = None,
) -> Path:
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    if cache_path.exists():
        print(f"[plm_embeddings] reusing {cache_path}", flush=True)
        return cache_path

    cfg = PLM_REGISTRY.get(model_key, PLMConfig(model_name or model_key))
    hf_name = model_name or cfg.hf_name
    local_files_only = os.environ.get("PLM_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if dtype == "float16" and device.startswith("cuda") else torch.float32
    print(f"[plm_embeddings] loading {hf_name} on {device} ({torch_dtype})", flush=True)

    unique = collect_unique_sequences(csv_paths)
    if limit is not None:
        unique = dict(list(unique.items())[:limit])
    out: dict[str, torch.Tensor] = {}
    if partial_path.exists():
        out = torch.load(partial_path, map_location="cpu")
        print(f"[plm_embeddings] resumed {len(out)} partial embeddings from {partial_path}", flush=True)
    items = [(key, seq) for key, seq in unique.items() if key not in out]
    hidden_dim = next(iter(out.values())).shape[-1] if out else None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plm_embeddings] embedding {len(items)} remaining / {len(unique)} total sequences", flush=True)

    if model_key.startswith("esmc"):
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        esmc_name = model_name or ("esmc_600m" if model_key == "esmc_600m" else "esmc_300m")
        model = ESMC.from_pretrained(esmc_name, device=torch.device(device)).eval()
        config = LogitsConfig(sequence=False, return_embeddings=True)
        with torch.no_grad():
            for idx, (key, seq) in enumerate(tqdm(items, desc=model_key), start=1):
                raw_seq = normalize_sequence(seq[:max_len])
                protein = ESMProtein(sequence=raw_seq)
                protein_tensor = model.encode(protein)
                emb = model.logits(protein_tensor, config).embeddings[0].detach().cpu()
                if emb.shape[0] >= len(raw_seq) + 2:
                    emb = emb[1 : len(raw_seq) + 1]
                else:
                    emb = emb[: len(raw_seq)]
                emb = emb.float().contiguous()
                hidden_dim = emb.shape[-1]
                out[key] = emb
                if save_every > 0 and idx % save_every == 0:
                    torch.save(out, partial_path)

        torch.save(out, cache_path)
        if partial_path.exists():
            partial_path.unlink()
        meta = {
            "model_key": model_key,
            "model_name": esmc_name,
            "num_sequences": len(out),
            "hidden_dim": hidden_dim,
            "max_len": max_len,
            "limit": limit,
            "csv_paths": [str(p) for p in csv_paths],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[plm_embeddings] wrote {len(out)} embeddings -> {cache_path}", flush=True)
        return cache_path

    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

    if model_key == "prott5" or "prot_t5" in hf_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(hf_name, do_lower_case=False, local_files_only=local_files_only)
        model = T5EncoderModel.from_pretrained(hf_name, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device).eval()
    elif model_key == "ankh" or "ankh" in hf_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=cfg.trust_remote_code, use_fast=True, local_files_only=local_files_only)
        model = T5EncoderModel.from_pretrained(hf_name, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device).eval()
    elif model_key == "protbert" or "prot_bert" in hf_name.lower():
        tokenizer = BertTokenizer.from_pretrained(hf_name, do_lower_case=False, local_files_only=local_files_only)
        config = BertConfig.from_pretrained(hf_name, local_files_only=local_files_only)
        model = BertModel.from_pretrained(hf_name, config=config, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=cfg.trust_remote_code, use_fast=False, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(hf_name, trust_remote_code=cfg.trust_remote_code, torch_dtype=torch_dtype, local_files_only=local_files_only)
        model = model.to(device).eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size), desc=model_key):
            batch = items[i : i + batch_size]
            raw = [seq[:max_len] for _, seq in batch]
            texts = [format_sequence(seq, cfg.spaced) for seq in raw]
            enc = tokenizer(
                texts,
                add_special_tokens=True,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = enc.pop("special_tokens_mask")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state.detach().cpu()
            attn = enc["attention_mask"].detach().cpu()
            for j, (key, seq) in enumerate(batch):
                keep = (attn[j].bool()) & (~special[j].bool())
                emb = h[j, keep][: min(len(seq), max_len)].float().contiguous()
                hidden_dim = emb.shape[-1]
                out[key] = emb
            if save_every > 0 and ((i // batch_size) + 1) % save_every == 0:
                torch.save(out, partial_path)

    torch.save(out, cache_path)
    if partial_path.exists():
        partial_path.unlink()
    meta = {
        "model_key": model_key,
        "model_name": hf_name,
        "num_sequences": len(out),
        "hidden_dim": hidden_dim,
        "max_len": max_len,
        "limit": limit,
        "csv_paths": [str(p) for p in csv_paths],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[plm_embeddings] wrote {len(out)} embeddings -> {cache_path}", flush=True)
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen PLM residue embeddings.")
    parser.add_argument("--csv", nargs="+", default=["data/splits/train.csv", "data/splits/validation.csv", "data/splits/test.csv"])
    parser.add_argument("--model-key", default="esm2_150m", choices=sorted(PLM_REGISTRY))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--save-every", type=int, default=25, help="Save partial embedding cache every N batches.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit on unique sequences.")
    args = parser.parse_args()
    cache = args.cache or f"meta_acp/runs/embeddings/{args.model_key}_embeddings.pt"
    cache_embeddings(
        args.csv,
        cache,
        args.model_key,
        args.model_name,
        args.batch_size,
        args.max_len,
        args.device,
        args.dtype,
        args.save_every,
        args.limit,
    )


if __name__ == "__main__":
    main()
