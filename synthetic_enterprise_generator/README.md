# Synthetic Enterprise Generator

Package-local README for the requested project tree. See the repository-level
`README.md` for full usage, design notes, and examples.

Run:

```bash
python main.py --config synthetic_enterprise_generator/configs/default.yaml
```

or:

```bash
python synthetic_enterprise_generator/main.py --rows 250 --worlds 1
```

For the 100k-row GPU paper dataset:

```bash
python main.py --config synthetic_enterprise_generator/configs/astar_100k_gpu.yaml
```

Outputs are written to `synthetic_enterprise_generator/outputs/astar_100k_gpu`.
The config requires CUDA-enabled PyTorch and will fail before generation if no
GPU is visible.
The CLI shows `Rows generated` and `Rows exported` progress bars by default.
Use `--no-progress` if you need plain batch logs.

If PyTorch fails to import with `undefined symbol: ncclCommWindowDeregister`,
the server has an incompatible PyTorch/NCCL/CUDA mix. Use a clean environment
and reinstall PyTorch from the matching CUDA channel before running this config.

Configs that include `xlsx` in `export.formats` also write
`generated_data_preview.xlsx`, an Excel workbook with a summary sheet and
sampled generated tables.

Stage X mixture APIs are available from `synthetic_enterprise_generator.generators`:
`generate_enterprise_worlds`, `generate_scientific_worlds`, `generate_iid_worlds`,
`generate_temporal_worlds`, `generate_relational_worlds`, `generate_sparse_worlds`,
`generate_advanced_forecasting_worlds`, and `sample_world_type`.

Target generation lives in `synthetic_enterprise_generator/generators/targets.py`.
It now uses tree-biased hybrid priors for classification, regression, ordinal,
and count labels while preserving forecasting, next-event, imputation, anomaly,
missingness, noise, outlier, seasonality, and drift modules.

Scientific worlds start from the same base TabPFN ecosystem adapter used by IID
worlds, then add correlated assay panels, latent-factor redundancy, batch
effects, low-sample regimes, and continuous scientific-property targets. Sparse
worlds likewise start from base-prior tables, then sparsify feature columns while
preserving rare informative signals. Advanced forecasting worlds reuse the
existing temporal augmentation and add trend, autoregression, shocks, concept
drift, regime changes, and multi-horizon forecast targets.
