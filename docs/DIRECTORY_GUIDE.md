# Repository directory guide

All paths below are relative to the repository root. Run the documented commands from that root.

| Directory | Purpose |
| --- | --- |
| `data/raw/` | Five original ACP source files |
| `data/splits/` | Integrated train, validation and test CSVs, with source-annotated companions |
| `data/processed/` | Merged sequence tables and long-form provenance records; provenance stays beside its master table for automatic discovery |
| `data/audit/` | Raw-record audit, conflicting labels and audit summary |
| `data/clustering/` | Archived CD-HIT 80% identity outputs and representatives |
| `scripts/reproduction/` | `reproduce_results.py`, `reproduce_stack.py`, `reproduce_length_analysis.py` and figure scripts |
| `scripts/training/` | `run_config.py`, which inspects or executes archived configurations |
| `meta_acp/` | Importable model, training, embedding, evaluation and analysis code |
| `meta_acp/tests/` | Existing project regression tests |
| `meta_acp/dataset_branches/` | Aegis benchmark and 90% identity branch datasets |
| `meta_acp/external_tasks/` | Related-task datasets and provenance |
| `meta_acp/runs_gpu/` | Archived configurations, available prediction heads, predictions and analyses |
| `results/` | Final manuscript tables, comparator results and missing-artifact inventory |
| `requirements/` | Dependency files for analysis, training and optional workflows |
| `docs/` | Data attribution, experiment mapping, validation and publication checks |
| `docs/figures/` | Figure 1 displayed on the repository homepage |
| `docs/manifests/` | Release checksums and original source provenance |
| `third_party/` | Preserved third-party license materials |

## Commands

```bash
python -m pip install -r requirements/requirements-analysis.txt
python scripts/reproduction/reproduce_results.py
python scripts/reproduction/reproduce_stack.py
python scripts/reproduction/reproduce_length_analysis.py
python scripts/reproduction/reproduce_figures.py
python scripts/reproduction/reproduce_figure4.py
python scripts/training/run_config.py meta_acp/runs_gpu/plm/esm2_150m/seed42/config.json
```

## Compatibility and archives

The root CSVs and Python entry points were moved on 9 October 2026. See [manifests/PATH_MIGRATION.csv](manifests/PATH_MIGRATION.csv) for every old and new path. Module entry points remain `python -m meta_acp.<module>`.

Archived configurations are preserved byte for byte. The training runner translates their legacy root input paths into `data/splits/` and `data/processed/` before displaying or executing commands. If using an archived configuration outside that runner, apply the same mapping explicitly. The runner also resolves the historical `source_data/loso/` prefix to the packaged `source_data/` partition directories. Branch-specific dataset paths and archived result paths are unchanged.

Data-building scripts still write their output basenames to their selected working directory. Follow the README instruction to rebuild in a separate workspace, then compare against the archived data. They do not overwrite the reorganized data automatically.

`SOURCE_MANIFEST.csv` retains the original packaging paths and checksums as historical provenance. `PATH_MIGRATION.csv` connects those old paths to the current layout. `RELEASE_MANIFEST.csv` describes current repository-relative paths and checksums and excludes itself. Generated `reproduced/` and `retrained/` outputs, caches and virtual environments remain ignored.

No input datasets, archived model weights, predictions or original configuration contents were changed by the reorganization.
