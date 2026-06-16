# Corpus Evaluation Framework

This package evaluates an existing generated synthetic corpus. It does not call
or modify the synthetic generator.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m evaluation.main \
  --config evaluation/configs/default.yaml \
  --corpus-dir synthetic_enterprise_generator/outputs/smoke_all_world_types \
  --output-dir evaluation/outputs/smoke_eval
```

For a fast server smoke test without benchmark downloads:

```bash
python -m evaluation.main \
  --corpus-dir synthetic_enterprise_generator/outputs/smoke_all_world_types \
  --output-dir evaluation/outputs/smoke_eval \
  --sample-rows 1000 \
  --skip-benchmarks
```

For publication benchmark analysis, remove `--skip-benchmarks`. OpenML datasets
will be downloaded according to `evaluation/configs/default.yaml`. Add UCI,
RelBench, or TabArena CSV/Parquet exports under `benchmarks.local_paths` in the
config.

## Outputs

```text
evaluation/outputs/
├── tables/
│   ├── dataset_summary.csv
│   ├── corpus_summary.csv
│   ├── world_distribution.csv
│   ├── schema_diversity.csv
│   ├── statistical_diversity.csv
│   ├── benchmark_metafeatures.csv
│   ├── nearest_benchmark_similarity.csv
│   └── readiness_scores.csv
├── figures/
│   ├── *.png
│   ├── *.pdf
│   ├── *.svg
│   └── world_task_feature_sankey.html
└── reports/
    ├── corpus_evaluation_report.md
    └── corpus_evaluation_report.pdf
```

## Main Metrics

- `WDI`: normalized Shannon entropy over world labels.
- `TDI`: normalized Shannon entropy over task labels.
- `FDI`: normalized Shannon entropy over feature-type support.
- `BCI`: benchmark meta-feature coverage using nearest-neighbor and cluster
  coverage in standardized benchmark/synthetic meta-feature space.
- `SDS`: statistical diversity score based on entropy, shape, correlation, and
  mutual information.
- `SDSchema`: schema diversity score based on schema entropy, size variation,
  and target diversity.
- `FMRS`: weighted readiness score:

```text
FMRS = 0.20*WDI + 0.15*TDI + 0.15*FDI + 0.20*BCI + 0.15*SDS + 0.15*SDSchema
```

