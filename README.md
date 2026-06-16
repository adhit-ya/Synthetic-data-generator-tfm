# Synthetic Enterprise Generator for TabFM Pretraining

This project generates synthetic structured worlds for pretraining a next-generation Tabular Foundation Model. It uses the TabPFN ecosystem as the preferred base synthetic prior source, then layers lightweight postprocessing for temporal, relational, workflow, missingness, scientific, sparse, and multi-task structure.

The generated data is intentionally not limited to simple IID rows or only enterprise workflows. The Stage X mixture-of-worlds sampler creates enterprise, scientific, IID benchmark-style, temporal, relational, and sparse high-dimensional worlds so the corpus better resembles a broad PFN-style prior over OpenML, TabArena, RelBench, UCI, Kaggle-style, scientific, forecasting, and enterprise distributions.

## Structure

```text
synthetic_enterprise_generator/
├── augmentation/      # entities, sessions, temporal features, missingness, noise
├── configs/           # YAML configs
├── datasets/          # reserved local dataset area
├── export/            # CSV/Parquet/PyTorch export
├── generators/        # TabPFN adapter, base tables, targets, mixture world orchestration
├── outputs/           # generated outputs
├── relational/        # multi-table relational generation
├── utils/             # logging, seeds, schema metadata, splits
└── workflows/         # NetworkX workflow graphs and event simulation
```

## Install

```bash
pip install -r requirements.txt
```

`tabpfn`, `tabpfn-extensions`, and `TabPFGen` are listed because the base prior generator is designed to reuse that ecosystem. The adapter dynamically discovers available generator functions across installed versions. If no callable ecosystem generator is present and `require_tabpfn_ecosystem` is `false`, the demo uses a deterministic scikit-learn fallback for base arrays so the enterprise augmentation pipeline remains executable.

To force strict ecosystem use, set:

```yaml
require_tabpfn_ecosystem: true
```

## Run

```bash
python main.py --config synthetic_enterprise_generator/configs/default.yaml
```

For a quick smoke run:

```bash
python main.py --rows 250 --worlds 1 --output-dir synthetic_enterprise_generator/outputs/smoke
```

For the 100k-row GPU dataset intended for paper-scale training/evaluation:

```bash
python main.py --config synthetic_enterprise_generator/configs/astar_100k_gpu.yaml
```

This writes all split CSV, Parquet, PyTorch tensor, and metadata outputs under:

```text
synthetic_enterprise_generator/outputs/astar_100k_gpu
```

The paper config sets `compute_device: "cuda"` and fails at startup unless
CUDA-enabled PyTorch can see an NVIDIA GPU. Its 100k base-prior path skips the
small TabPFN ecosystem generator limit and uses the scalable Torch CUDA prior.
The CLI shows live progress bars by default: `Rows generated` tracks base data
point creation, and `Rows exported` tracks train/validation/test file writes.
For non-interactive batch logs, pass `--no-progress`.

If the GPU server fails while importing PyTorch with an error like
`undefined symbol: ncclCommWindowDeregister`, the installed PyTorch package is
loading an incompatible NCCL/CUDA library. Create a clean environment and install
PyTorch from one CUDA channel instead of mixing system, conda, and pip CUDA
packages. Example for CUDA 12.1:

```bash
conda create -n synthetic-tfm python=3.12 -y
conda activate synthetic-tfm
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `requirements.txt` reinstalls a different PyTorch build, rerun the PyTorch
install command above after installing the other requirements.

For a large run, the row override is treated as an exact per-world count and
the base feature matrix is bounded by `max_table_cells` to avoid accidental
memory thrashing:

```bash
python main.py --rows 100000 --worlds 1 --device auto
```

`--device auto` uses CUDA for compatible TabPFN ecosystem calls and the scalable
PyTorch base-prior fallback when CUDA-enabled PyTorch and an NVIDIA GPU are
available. Use `--device cuda` to require GPU execution or `--device cpu` to
force CPU execution. Pandas-based relational, temporal, missingness, and export
stages remain CPU operations.

To make the sampled world family reproducible, pass `--world-type` or set
`world_type` in YAML. Supported values are `enterprise`, `scientific`, `iid`,
`temporal`, `relational`, `sparse`, and `advanced_forecasting`.

## Outputs

Each world exports:

- `fact_events`
- `customers`
- `products`
- `sessions`
- `transactions`
- `machine_logs`

Each table is split into train/validation/test and saved as configured:

- CSV
- Parquet
- `.pt` tensors for PyTorch pretraining
- optional `generated_data_preview.xlsx` workbook when `xlsx` is in `export.formats`
- `metadata.json` with schema descriptions and missing-value statistics

## Main APIs

Stage 1 base generation:

```python
from synthetic_enterprise_generator.generators import (
    generate_classification_table,
    generate_regression_table,
    generate_mixed_schema_table,
)
```

Full world generation:

```python
from synthetic_enterprise_generator.config import load_config
from synthetic_enterprise_generator.generators.worlds import generate_multiple_worlds

config = load_config("synthetic_enterprise_generator/configs/default.yaml")
worlds = generate_multiple_worlds(config)
```

Mixture-of-worlds APIs:

```python
from synthetic_enterprise_generator.generators import (
    sample_world_type,
    generate_enterprise_worlds,
    generate_scientific_worlds,
    generate_iid_worlds,
    generate_temporal_worlds,
    generate_relational_worlds,
    generate_sparse_worlds,
)
```

PyTorch formatting:

```python
from synthetic_enterprise_generator.export import build_pytorch_dataset

dataset = build_pytorch_dataset(worlds["WORLD_00000"]["fact_events"])
item = dataset[0]
```

## Design Notes

- Base priors are isolated in `generators/tabpfn_adapter.py`, making it easy to plug in a concrete TabPFGen API.
- Base-prior calls propagate the configured compute device. Requests larger
  than `tabpfn_max_rows` use the scalable sklearn/torch fallback unless strict
  ecosystem use is enabled.
- `generators/worlds.py` implements a balanced mixture prior over enterprise, scientific, IID, temporal, relational, and sparse world categories.
- `generators/targets.py` uses tree-biased hybrid target priors: threshold rules, rule conjunctions, categorical effects, gated feature interactions, and smaller smooth components. This is meant to resemble the structure GBDTs exploit without becoming a pure tree simulator.
- Enterprise structure is composed as small independent stages, so you can disable, reorder, or replace stages.
- IDs and foreign keys are stable and repeated across rows to support relational reasoning.
- Timestamps are monotonic within sessions and include drift, seasonality, and regime shifts.
- Workflow events are sampled from weighted NetworkX directed graphs.
- Missingness supports MCAR, MAR, and MNAR patterns with protected key/label columns.
- Exported tensors include numeric values, categorical IDs, masks, targets, and encoding metadata.
