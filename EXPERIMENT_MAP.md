# Manuscript reproduction map

Paths below are relative to meta_acp/runs_gpu unless another base is given. The primary integrated evaluation, Aegis benchmark and related tasks use separate predictors and partitions.

| Manuscript item | Authoritative inputs or archived results | Code |
|---|---|---|
| Table 1 and Supplementary Table S1 | Root master, provenance and raw-record audit files | rebuild_data_audit.py and source_data.py |
| Table 2 and Supplementary Table S2 | Archived configs, plm_embeddings.py model registry and final manuscript table exports | train_plm_metaacp.py and plm_embeddings.py |
| Table 3 | ensemble_with_hardcase_strict and final_statistics | reproduce_results.py, reproduce_stack.py and statistical_validation.py |
| Supplementary Table S3 | final_statistics and branch_benchmarks/integrated_cdhit09 | statistical_validation.py |
| Supplementary Table S4 | final_statistics/aegis_esm2_150m_3seed_ensemble_predictions.csv and results/Aegis_benchmark_comparison.csv | reproduce_results.py, with published comparator values transcribed from Aegis |
| Figure 3a and Supplementary Table S5 | ensemble_with_hardcase_strict/member_predictions/test, 18 standard candidates | reproduce_results.py and reproduce_figures.py |
| Figure 3b | frozen_manuscript_tables/source_wise_mean_std.csv | source_eval.py and reproduce_figures.py |
| Figure 3c and Supplementary Table S6 | loso predictions, configs and loso_summary_esm2_150m | summarize_loso_esm2_150m.py and reproduce_figures.py |
| Supplementary Tables S7 and S8 | LOSO and primary stack predictions grouped by sequence length | reproduce_length_analysis.py |
| Figure 3d and Supplementary Table S9 | external_public and external_public_summary | external_transfer.py, summarize_external_public.py and reproduce_figures.py |
| Supplementary Table S10 | Few-shot settings reported in the archived implementation and manuscript | few_shot.py with explicit manuscript settings |
| Figure 4a,b and Supplementary Tables S11 and S12 | few_shot and few_shot_strict_analysis | analyze_few_shot_strict.py and reproduce_figure4.py |
| Figure 4c,d and Supplementary Table S13 | ablations and ablation_esm2_150m_summary | summarize_esm2_150m_ablations.py and reproduce_figure4.py |
| Figure 5 and Supplementary Table S14 | physchem_provenance | physchem_provenance_analysis.py and reproduce_figures.py |
| Supplementary Figure S1 | physchem_provenance/calibration_bins.csv | reproduce_figures.py |

Root-level reproduction scripts run from the repository root. Named training and statistical scripts reside in meta_acp. Manuscript table CSVs use the final supplementary numbering. The independent source-exclusion models must not be replaced by source-wise slices of primary-stack predictions.

Figures 1 and 2 are author-created workflow and architecture diagrams. Their submission files are supplied separately. Figure 1 was exported from its editable source after correcting the ESM-C spelling. Figure 2 is available as the final high-resolution image. It is not labelled as an editable vector source.
