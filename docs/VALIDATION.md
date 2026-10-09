# Release validation

The release was checked on 29–30 September 2026 using Python 3.12.14 on Windows. Analysis dependency versions are listed below.

- numpy 2.3.5
- pandas 3.0.1
- scipy 1.18.1
- matplotlib 3.11.1
- scikit-learn 1.9.1
- joblib 1.6.0
- threadpoolctl 3.7.0

## Completed checks

- Recomputed six reported AUC estimates, primary AP and MCC, Aegis AP, and the primary Brier score from frozen predictions.
- Verified integrated partition sizes and absence of exact sequence overlap between training, validation and test partitions.
- Recomputed the 18 standard backbone and nine LOSO prediction summaries.
- Refit the 30-member stack on saved validation predictions. Validation again selected C = 3. Test AUC, AP and MCC matched manuscript rounding. The largest probability difference from the archive was approximately 1.48e-6 under the current solver version.
- Regenerated Figures 3, 4, 5 and S1 from archived numerical inputs. Figure labels use AP.
- Parsed packaged Python sources for syntax errors and checked configuration-derived training commands without starting model training.

- Recomputed the integrated-test length strata and source-specific length summaries.
- Ran the statistical-validation entry point with 20 bootstrap draws as an execution check. This smoke run does not replace the archived 10,000-draw analyses.

## Not repeated

Frozen embedding extraction, downstream model training, the full original test suite and every 10,000-resample statistical analysis were not rerun during packaging. Training dependencies and hardware were not recreated. Minimum dependency versions are installation guidance, not the original environment lockfile.

Original files missing locally are listed in results/missing_original_artifacts.json. There are 42 missing member artifact entries, counting checkpoint and configuration separately. All 30 candidate validation and test prediction files are present. This distinction permits stack refitting and result verification but does not constitute complete restoration of the original training state.

The manuscript and supplementary numerical tables retain the reported values. New outputs are written outside the archived result folders.
