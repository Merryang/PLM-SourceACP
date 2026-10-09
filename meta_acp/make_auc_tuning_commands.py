"""Generate reproducible AUC-tuning commands for frozen-PLM heads."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write hyperparameter tuning command matrix.")
    parser.add_argument("--backbone", default="esm2_150m")
    parser.add_argument("--embedding-cache", default="meta_acp/runs_gpu/embeddings/esm2_150m_embeddings.pt")
    parser.add_argument("--train", default="train_source.csv")
    parser.add_argument("--validation", default="validation_source.csv")
    parser.add_argument("--test", default="test_source.csv")
    parser.add_argument("--out-dir", default="meta_acp/runs_gpu/auc_tuning")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--top-configs-only", action="store_true", help="Use a compact hand-picked grid for first pass.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.top_configs_only:
        configs = [
            {"proj_dim": 128, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "focal_gamma": 2, "lambda_supcon": 0.1, "lambda_proto": 0.1, "monitor": "auc", "sampler": "none"},
            {"proj_dim": 256, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4, "focal_gamma": 2, "lambda_supcon": 0.1, "lambda_proto": 0.1, "monitor": "auc", "sampler": "source_class"},
            {"proj_dim": 256, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-5, "focal_gamma": 1, "lambda_supcon": 0.05, "lambda_proto": 0.1, "monitor": "ap", "sampler": "class"},
            {"proj_dim": 512, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-4, "focal_gamma": 2, "lambda_supcon": 0.2, "lambda_proto": 0.05, "monitor": "auc", "sampler": "source_class"},
            {"proj_dim": 256, "dropout": 0.1, "lr": 3e-4, "weight_decay": 1e-4, "focal_gamma": 3, "lambda_supcon": 0.1, "lambda_proto": 0.2, "monitor": "ap", "sampler": "source_class"},
        ]
    else:
        configs = [
            {
                "proj_dim": proj_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": wd,
                "focal_gamma": gamma,
                "lambda_supcon": lam_sup,
                "lambda_proto": lam_proto,
                "monitor": monitor,
                "sampler": sampler,
            }
            for proj_dim, dropout, lr, wd, gamma, lam_sup, lam_proto, monitor, sampler in itertools.product(
                [256, 512],
                [0.1, 0.2, 0.3],
                [1e-4, 3e-4, 1e-3],
                [1e-5, 1e-4],
                [1, 2, 3],
                [0.0, 0.05, 0.1, 0.2],
                [0.0, 0.05, 0.1, 0.2],
                ["auc", "ap"],
                ["none", "class", "source_class"],
            )
        ]

    rows = []
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source ~/anaconda3/etc/profile.d/conda.sh",
        "conda activate metaacp_gpu || conda activate lungenv",
    ]
    for ci, cfg in enumerate(configs, start=1):
        for seed in args.seeds:
            name = (
                f"{args.backbone}_cfg{ci:04d}_seed{seed}_"
                f"pd{cfg['proj_dim']}_do{cfg['dropout']}_lr{cfg['lr']}_g{cfg['focal_gamma']}_"
                f"ls{cfg['lambda_supcon']}_lp{cfg['lambda_proto']}_{cfg['monitor']}_{cfg['sampler']}"
            ).replace(".", "p")
            run_dir = f"{args.out_dir}/{name}"
            cmd = (
                "python meta_acp/train_plm_metaacp.py "
                f"--train {args.train} --validation {args.validation} --test {args.test} "
                "--encoder plm "
                f"--embedding-cache {args.embedding_cache} "
                f"--out-dir {run_dir} --seed {seed} --device {args.device} "
                "--batch-size 32 --epochs 60 --patience 10 "
                f"--proj-dim {cfg['proj_dim']} --dropout {cfg['dropout']} "
                f"--lr {cfg['lr']} --weight-decay {cfg['weight_decay']} "
                f"--focal-gamma {cfg['focal_gamma']} "
                f"--lambda-supcon {cfg['lambda_supcon']} --lambda-proto {cfg['lambda_proto']} "
                f"--monitor-metric {cfg['monitor']} --balanced-sampler {cfg['sampler']}"
            )
            commands.append(f'if [ ! -s "{run_dir}/test_metrics.csv" ]; then {cmd}; fi')
            rows.append({"name": name, "seed": seed, "run_dir": run_dir, "command": cmd, **cfg})

    (out / "auc_tuning_jobs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (out / "run_auc_tuning.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} jobs to {out}")


if __name__ == "__main__":
    main()
