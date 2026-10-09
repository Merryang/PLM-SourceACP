# Repository directory guide

The repository is organized by research role. Existing input and archive paths are preserved so the released training configurations and reproduction commands continue to work.

| Location | Contents |
| --- | --- |
| `README.md` | Installation, reproduction and training instructions |
| `docs/` | Repository navigation and publication checks |
| `data/raw/` | Five original ACP input files |
| Root `train.csv`, `validation.csv`, `test.csv` | Integrated partitions used directly by training scripts |
| Root `*_source.csv`, `provenance*.csv` | Source annotations and retained memberships |
| Root `merged_all*.csv`, `raw_records_label_audit.csv`, `label_*` | Merge and label-audit records |
| Root `cdhit_result_08*` | Archived clustering outputs |
| `meta_acp/*.py` | Model, embedding, training, evaluation and statistical code |
| `meta_acp/tests/` | Existing project tests |
| `meta_acp/dataset_branches/` | Aegis benchmark and 90% identity branch datasets |
| `meta_acp/external_tasks/` | Related-task datasets and provenance |
| `meta_acp/runs_gpu/` | Archived configurations, available prediction-head checkpoints, predictions and analyses |
| `results/manuscript_tables/` | Final numbered manuscript tables and captions |
| `results/missing_original_artifacts.json` | Explicit inventory of unavailable original artifacts |
| `third_party/` | Preserved third-party license material |
| Root `reproduce_*.py` | Lightweight reproduction entry points |
| Root `requirements*.txt` | Analysis, training and optional dependencies |
| `SOURCE_MANIFEST.csv` | Original packaging provenance; retained unchanged |
| `RELEASE_MANIFEST.csv` | SHA-256 inventory of published files, excluding itself and generated outputs |

## Working outputs

Run commands from the repository root. Numerical and figure reproduction writes to `reproduced/`; training writes to `retrained/`. These generated directories, virtual environments, logs and downloaded embedding caches are ignored by Git.

## Scope

All 710 files in the supplied release folder are represented in this repository. The repository's existing Apache 2.0 license is preserved. Pretrained encoder weights are obtained separately from their providers. Missing original training artifacts are documented rather than reconstructed. See `DATA_SOURCES.md`, `EXPERIMENT_MAP.md` and `VALIDATION.md` for provenance, experiment mapping and scientific reproduction limits.
