# Data sources and third-party materials

The Apache License 2.0 applies to project-authored code. It does not replace the terms attached to the datasets and pretrained models below. The files here preserve the public-source extracts used for the reported analysis, including inherited labels and released-file provenance.

| Resource | Files and role | Primary source | License evidence |
|---|---|---|---|
| Aegis | training.fasta and testing.fasta, integrated data and separate benchmark | https://doi.org/10.1186/s12915-026-02520-y | Cite the original article and obtain upstream data terms. No independent dataset license was present with the supplied local files. |
| ProtT5-MSCRNet | Train.fasta, Test1.fasta and Test2.fasta | https://doi.org/10.3389/frai.2026.1856815 | Cite the original article and its underlying resources. No independent dataset license was present with the supplied local files. |
| AntiAngioPred | Official development and test partitions, with two noncanonical development records excluded | https://doi.org/10.1371/journal.pone.0136990 and https://github.com/sachini-tech/AntiAngioPred/tree/V1 | The downloaded source-repository license is preserved in third_party/AntiAngioPred_LICENSE. |
| PepBenchmark CPP | PepBenchData-50/cpp with cdhit_split.json seed_0 | https://github.com/ZGCI-AI4S-Pep/PepBenchmark and https://huggingface.co/datasets/jiahuizhang/PepBenchData/tree/main/PepBenchData-50/cpp | The source and exact input hashes are in the task summary. A separate dataset redistribution license was not verified during packaging. |

The absence of a verified dataset license is not an Apache grant for that dataset. Before publicly redistributing the source extracts, confirm the upstream data conditions. The local reproducibility archive preserves them for the authors. Public users can obtain the inputs from the cited providers and apply the included processing scripts when redistribution is not permitted.

The integrated provenance table retains multiple memberships. Do not infer that distinct filenames describe disjoint biological populations. The original Train.fasta and Test2.fasta overlap, which is documented in the audit.

The following pretrained models are downloaded separately through the providers configured in meta_acp/plm_embeddings.py. Their weights are not included or relicensed here.

| Model | Provider identifier |
|---|---|
| ProtT5-XL-UniRef50 | Rostlab/prot_t5_xl_uniref50 |
| ProtBERT | Rostlab/prot_bert |
| Ankh-base | ElnaggarLab/ankh-base |
| ESM2-150M | facebook/esm2_t30_150M_UR50D |
| ESM2-650M | facebook/esm2_t33_650M_UR50D |
| ESM-C 300M | EvolutionaryScale/esmc-300m-2024-12 |

Dependency packages retain their own licenses. Published papers and downloaded software installers are excluded from this repository package.
