# Publication checks

Checked on 9 October 2026 before publishing to https://github.com/Merryang/PLM-SourceACP.

- Copied all 710 supplied release files into a separate Git checkout, preserving the existing repository license. Verified SHA-256 equality before editing publication documentation.
- Initial publication kept all experimental paths unchanged. Added a directory guide, refreshed the README publication section, and corrected the ignore rule to include embedding metadata JSON files while excluding downloaded embedding tensors.
- Parsed every packaged Python source with Python 3.12.14.
- Ran `python scripts/reproduction/reproduce_results.py`: passed all reported checks, including six AUC estimates, primary AP/MCC/Brier score, Aegis AP, split sizes and separation, 18 backbone predictions and nine LOSO predictions.
- Ran `python scripts/reproduction/reproduce_stack.py`: selected C = 3; AUC = 0.9708969839404622, AP = 0.9658522424645535, MCC = 0.8600367057266719. Maximum probability difference from the archive = 1.4787078636002704e-6.
- Validation environment: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0. Original packaging checks and their environment remain in `docs/VALIDATION.md`.
- Rebuilt `docs/manifests/RELEASE_MANIFEST.csv` for the final published inventory. Git attributes preserve file bytes across operating systems.

Generated reproduction outputs are local and ignored. Full GPU embedding extraction, model training and exhaustive statistical resampling were not repeated for this publication check. Unavailable original artifacts remain listed in `results/missing_original_artifacts.json`.


## Directory reorganization and homepage figure, 9 October 2026

- Moved root datasets to `data/splits/`, `data/processed/`, `data/audit/` and `data/clustering/`; moved runnable scripts to `scripts/reproduction/` and `scripts/training/`.
- Grouped dependency files under `requirements/` and supporting documentation and manifests under `docs/`. The README, Apache license and notice remain at the root.
- Updated module default input paths, test fixtures, README commands and directory documentation. Preserved raw data, archived configurations, prediction files and model tensors byte for byte.
- Checked all 60 archived training configurations through the configuration runner; all referenced dataset and provenance inputs resolve. The runner adapts legacy root paths and the original extra LOSO partition prefix without modifying archived JSON files.
- Reran numerical reproduction, stack refitting and length analysis successfully. Regenerated Figures 3, 4, 5 and S1 with the relocated scripts. The four existing label-audit regression tests passed.
- Parsed Python files and checked relative Markdown links, including the homepage image reference.
- Copied the supplied Figure 1 unchanged to `docs/figures/Figure_1.png` and embedded it below the README introduction.
- Recorded the moved paths in `docs/manifests/PATH_MIGRATION.csv` and rebuilt the release checksums for the current tree. The source manifest remains historical provenance.
