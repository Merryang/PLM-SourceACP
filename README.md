# PLM-SourceACP

PLM-SourceACP uses frozen protein language model representations for anticancer peptide prediction and evaluates how performance varies across released data sources. This repository accompanies **PLM-SourceACP for provenance-aware anticancer peptide prediction and source generalization**.

Project repository https://github.com/Merryang/PLM-SourceACP

Corresponding author Huan Yang, huanyang@csj.uestc.edu.cn

## What is included

The release contains the integrated data partitions, provenance records, related-task partitions, final candidate predictions, available trained prediction heads, statistical summaries, and project code. The primary analysis uses 2,882 unique sequences split into 1,728 training, 577 validation, and 577 test sequences. A separate Aegis benchmark retains its original 138-sequence test set.

Two workflows serve different purposes.

1. **Recompute results from saved predictions** checks the reported estimates and rebuilds quantitative figures without downloading a protein language model.
2. **Train models again** extracts frozen embeddings and trains prediction heads using the archived settings. This requires additional dependencies, model downloads and suitable computing resources.

The archived member predictions cover all 30 ensemble candidates. Original checkpoint and configuration files are incomplete for some candidates. The exact inventory is in `results/missing_original_artifacts.json`. Consequently, the release supports numerical result reproduction more completely than restoration of every original training run. Do not interpret a successful result check as a full training replication.

For a complete navigation guide, see [docs/DIRECTORY_GUIDE.md](docs/DIRECTORY_GUIDE.md). Publication checks are recorded in [docs/PUBLICATION_CHECKS.md](docs/PUBLICATION_CHECKS.md).

## Directory layout

```text
data/raw/                         Original ACP source files used in this study
train.csv                         Integrated training partition
validation.csv                    Integrated validation partition
test.csv                          Integrated test partition
merged_all_cdhit08.csv             Integrated sequence representatives
provenance_cdhit08_long.csv         All retained source memberships
meta_acp/                         Training, evaluation and analysis code
meta_acp/dataset_branches/         Aegis and 90% identity partitions
meta_acp/external_tasks/            Antiangiogenic and cell-penetrating tasks
meta_acp/runs_gpu/                 Frozen configs, predictions and results
results/manuscript_tables/         Tables using final manuscript numbering
results/missing_original_artifacts.json
reproduce_results.py               Lightweight numerical verification
reproduce_figures.py               Figures 3, 5 and S1
reproduce_figure4.py                Figure 4
run_config.py                     Portable archived-configuration runner
SOURCE_MANIFEST.csv                Source and packaged file checksums
DATA_SOURCES.md                    Data attribution and access conditions
EXPERIMENT_MAP.md                  Results, code and input correspondence
```

The `meta_acp` folder name and `runs_gpu` paths are retained for compatibility with the experimental code. Outputs placed under `runs_gpu` are archived research files and do not imply that a GPU is required to read them.

## Environment

Use Python 3.12 and an isolated environment. Lightweight reproduction was checked with the versions listed in `VALIDATION.md`. Dependency ranges describe installation requirements, not a recovered lockfile for the original training environment.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell. Run subsequent commands from the repository root.

```bash
python -m pip install -r requirements-analysis.txt
python reproduce_results.py
python reproduce_figures.py
python reproduce_figure4.py
python reproduce_length_analysis.py
python reproduce_stack.py
```

These commands write new files to `reproduced/`. They do not overwrite archived predictions. Figure 1 is an author-prepared workflow diagram, and Figure 2 is an architecture diagram. They are provided with the submission assets rather than inferred from numerical data.

## Expected numerical checks

| Evaluation | Test sequences | AUC | AP |
|---|---|---|---|
| Integrated primary stack | 577 | 0.9709 | 0.9659 |
| Integrated auxiliary stack | 577 | 0.9737 | 0.9683 |
| Validation-selected single model | 577 | 0.9562 | 0.9498 |
| Original Aegis benchmark ensemble | 138 | 0.9849 | 0.9960 |
| Antiangiogenic task ensemble | 56 | 0.8036 | Recomputed in output |
| Cell-penetrating task ensemble | 229 | 0.7979 | Recomputed in output |

The primary stack has MCC 0.8600 and Brier score 0.0530. `reproduce_results.py` also checks integrated split sizes, exact sequence separation, 18 standard-backbone prediction files and nine source-exclusion prediction files. It writes `performance_recomputed.csv`, backbone summaries, LOSO metrics and `verification.json`.

AUC denotes area under the receiver operating characteristic curve, AP denotes average precision, MCC denotes Matthews correlation coefficient, and LOSO denotes leave-one-source-out. AP is computed by the archived implementation in `meta_acp/metrics.py`. The release consistently uses the key `ap`. Older local archives used a different key for the same calculated quantity. Renaming the key does not change any numerical values. Metric computation, including treatment of tied scores, is preserved for traceability.

## Data and provenance

The five ACP input files are `Train.fasta`, `Test1.fasta`, `Test2.fasta`, `training.fasta`, and `testing.fasta`. Despite their extensions, the first three use the released label and sequence format. Do not parse all five files using a single generic FASTA reader.

The audit starts with 3,418 records. Exact duplicate merging gives 3,036 unique sequences, and one conflicting-label sequence is removed. Clustering at 80% identity retains 2,882 representatives. The provided partitions preserve the choices made during the original processing. Re-running clustering with another version, sequence order or representative-selection setting may change the retained sequences.

To examine the raw processing without altering the released partitions, create a separate working directory and run the script there with absolute input and script paths. The original script writes its processed CSVs to the working directory.

```bash
python -m meta_acp.rebuild_data_audit --help
python -m meta_acp.source_data --help
```

Reclustering requires CD-HIT 4.8.1. The existing `cdhit_result_08` files preserve the original representative choices. The audit script also accepts `--reuse-cdhit-dir` to reuse these files. The retained LOSO partitions are under `meta_acp/runs_gpu/source_data/`. They exclude every membership in the target file, including sequences shared with another source. File-level provenance does not imply an independent laboratory or biological cohort.

Dataset origins and third-party terms are documented in `DATA_SOURCES.md`. The Apache license for this project does not change those terms.

## Frozen representations and training

Install a PyTorch build appropriate for the target hardware, then install the training dependencies.

```bash
python -m pip install -r requirements-training.txt
```

The supported manuscript backbones are ProtT5-XL-UniRef50, ProtBERT, Ankh-base, ESM2-150M, ESM2-650M and ESM-C 300M. Model identifiers are recorded in `meta_acp/plm_embeddings.py`. In the command-line registry, `esm2` selects the 650M model and `esm2_150m` selects the 150M model. Pretrained weights and embedding caches are not bundled. Download and access conditions remain those of the model providers. ESM-C uses the EvolutionaryScale `esm` package.

The following example builds the integrated ESM2-150M cache. Cached residue vectors are stored as float32 by the implementation. The `--dtype` option controls encoder computation, which can differ from storage precision. Check archived cache metadata before claiming exact numerical reproduction.

```bash
python -m meta_acp.plm_embeddings --csv train.csv validation.csv test.csv --model-key esm2_150m --cache meta_acp/runs_gpu/embeddings/esm2_150m_embeddings.pt --batch-size 4 --device cuda
python run_config.py meta_acp/runs_gpu/plm/esm2_150m/seed42/config.json
python run_config.py meta_acp/runs_gpu/plm/esm2_150m/seed42/config.json --execute
```

The first `run_config.py` command displays the effective invocation. Adding `--execute` starts training. It uses the archived configuration, changes the device only as requested and redirects new training outputs into `retrained/`. It does not replace frozen results. Use the seed-2024 and seed-2026 configurations for the other standard runs. The configuration records batch size 64, unlike some historical command-line defaults, so use the configuration runner instead of relying on defaults.

For another backbone, first generate its embedding cache with the appropriate model key, then use its archived configuration. Some original ESM2-650M configurations and auxiliary candidate artifacts are absent. Shared architecture settings and the candidate settings in manuscript Table 2 describe those experiments, but reconstructed settings must not be represented as recovered original files.

The largest encoders may require substantially more memory than ESM2-150M. Reduce embedding batch size when necessary. No new end-to-end timing or hardware benchmark was performed for this release, and no universal GPU-memory requirement is asserted.

## Additional experiments

LOSO training configurations are under `meta_acp/runs_gpu/loso/`. Use `run_config.py` on each `config.json` after building the integrated embedding cache. Aegis and 90% identity branches use their own partitions and caches. Their archived configurations are under `meta_acp/runs_gpu/branch_benchmarks/`.

Related-task configurations are under `meta_acp/runs_gpu/external_public/`. Build an ESM2-150M embedding cache from each task’s training, validation and test files before running its outer `full_head_seed*/config.json` or `linear_probe/config.json`. The outer configuration runs `external_transfer.py`, including task-specific threshold selection. It must not be replaced by direct evaluation of an ACP checkpoint.

Few-shot adaptation uses the matching source-excluded checkpoint for each target. After retraining, substitute the new checkpoint path in the following command.

```bash
python -m meta_acp.few_shot --checkpoint retrained/meta_acp/runs_gpu/loso/Test1_fasta_seed42/best.pt --embedding-cache meta_acp/runs_gpu/embeddings/esm2_150m_embeddings.pt --master-csv merged_all_cdhit08.csv --provenance-csv provenance_cdhit08_long.csv --sources Test1.fasta --k-shot 5 10 20 --repeats 20 --seed 42 --batch-size 64 --finetune-epochs 25 --finetune-lr 0.0005 --weight-decay 0.0001 --lambda-supcon 0 --lambda-proto 0 --out-dir retrained/few_shot/Test1 --device cuda
```

Repeat with the Test2 checkpoint and source. Five and ten examples per class form the main paired analysis. Twenty examples per class are descriptive. Query sequences overlap between support draws, so the draws are not independent biological replications.

The `ensemble_with_hardcase_strict` directory contains the frozen 30-member validation and test predictions. The primary logistic stack uses validation-selected C = 3, with class balancing and unconstrained coefficients. The C = 1 model is auxiliary. Never select a predictor using its test results. `reproduce_stack.py` refits this stack using only the saved validation predictions and compares its test estimates with the archived primary stack.

## Statistical reproduction

Archived confidence intervals and adjusted comparisons are supplied alongside their input predictions. Resampling scripts provide the original settings, including bootstrap counts and random seeds. The normal lightweight check verifies point estimates and input consistency. It does not silently rerun 10,000 bootstrap resamples for every comparison.

```bash
python -m meta_acp.statistical_validation --runs-dir meta_acp/runs_gpu --out-dir reproduced/statistics --n-bootstrap 10000 --random-seed 20260618
python -m meta_acp.analyze_few_shot_strict --help
python -m meta_acp.summarize_esm2_150m_ablations --help
python -m meta_acp.summarize_external_public --help
python -m meta_acp.physchem_provenance_analysis --help
```

The manuscript tables in `results/manuscript_tables/` use the revised numbering. Historical analysis filenames retain their semantic names and do not determine the manuscript table number. Use `EXPERIMENT_MAP.md` to identify the correct source.

## Validation and limits

See `VALIDATION.md` for the checks actually completed during packaging. Full training and embedding extraction were not repeated. Original candidate checkpoints and configurations that were not present locally are listed rather than reconstructed silently. The provided results evaluate retrospective sequence collections and do not establish anticancer potency, selectivity, safety or clinical suitability.

## License and citation

Project-authored code is available under the Apache License, Version 2.0. Consult `LICENSE`, `NOTICE` and `DATA_SOURCES.md` for scope and third-party attribution. This license does not relicense third-party pretrained weights, upstream datasets or publications.

Until publication metadata are available, cite the repository and manuscript title. Do not use the Aegis DOI as the identifier of PLM-SourceACP. Update the citation information after the article is published.

## Release inventory

`RELEASE_MANIFEST.csv` records the size and SHA-256 digest of each published file except the manifest itself. `SOURCE_MANIFEST.csv` retains the original packaging provenance. Generated `reproduced/` outputs and downloaded embedding caches are excluded by `.gitignore`. Large pretrained encoder weights remain at their official download locations.
