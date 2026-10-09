# Publication checks

Checked on 9 October 2026 before publishing to https://github.com/Merryang/PLM-SourceACP.

- Copied all 710 supplied release files into a separate Git checkout, preserving the existing repository license. Verified SHA-256 equality before editing publication documentation.
- Kept all experimental paths unchanged. Added a directory guide, refreshed the README publication section, and corrected the ignore rule to include embedding metadata JSON files while excluding downloaded embedding tensors.
- Parsed every packaged Python source with Python 3.12.14.
- Ran `python reproduce_results.py`: passed all reported checks, including six AUC estimates, primary AP/MCC/Brier score, Aegis AP, split sizes and separation, 18 backbone predictions and nine LOSO predictions.
- Ran `python reproduce_stack.py`: selected C = 3; AUC = 0.9708969839404622, AP = 0.9658522424645535, MCC = 0.8600367057266719. Maximum probability difference from the archive = 1.4787078636002704e-6.
- Validation environment: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0. Original packaging checks and their environment remain in `VALIDATION.md`.
- Rebuilt `RELEASE_MANIFEST.csv` for the final published inventory. Git attributes preserve file bytes across operating systems.

Generated reproduction outputs are local and ignored. Full GPU embedding extraction, model training, figure regeneration and exhaustive statistical resampling were not repeated for this publication check. Unavailable original artifacts remain listed in `results/missing_original_artifacts.json`.
