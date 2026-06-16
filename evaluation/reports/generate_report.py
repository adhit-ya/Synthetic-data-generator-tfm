"""Automated research report generation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _fmt(value: float) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```\n" + df.head(max_rows).to_string(index=False) + "\n```"


def generate_markdown_report(
    *,
    output_path: Path,
    title: str,
    dataset_stats: pd.DataFrame,
    corpus_summary: pd.DataFrame,
    world_coverage: pd.DataFrame,
    schema_stats: pd.DataFrame,
    statistical_stats: pd.DataFrame,
    benchmark_stats: pd.DataFrame,
    distribution_similarity: pd.DataFrame,
    nearest_similarity: pd.DataFrame,
    readiness: pd.DataFrame,
    bci_table: pd.DataFrame,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmrs = readiness.loc[readiness["metric"] == "FMRS", "score"]
    fmrs_value = _fmt(fmrs.iloc[0]) if not fmrs.empty else "0.000"
    bci = readiness.loc[readiness["metric"] == "BCI", "score"]
    bci_value = _fmt(bci.iloc[0]) if not bci.empty else "0.000"

    text = f"""# {title}

## Methodology

This evaluation analyzes an already generated Mixture-of-Worlds synthetic corpus.
The evaluator does not call or modify the generator. Each exported table is
reconstructed from its train, validation, and test splits. Dataset morphology,
schema diversity, missingness, sparsity, statistical diversity, benchmark
coverage, and foundation-model readiness are computed from observed corpus
outputs.

Benchmark coverage is estimated in a unified meta-feature space containing row
counts, column counts, feature-type ratios/counts, missingness, sparsity,
skewness, kurtosis, entropy, correlation, and mutual-information summaries.
Real benchmark datasets are downloaded from OpenML when enabled, and local
benchmark files can be supplied for UCI, RelBench, and TabArena.

## Corpus Morphology

{_markdown_table(corpus_summary)}

### Dataset-Level Statistics

{_markdown_table(dataset_stats)}

## World Coverage Analysis

The World Diversity Index (WDI) is Shannon entropy over inferred world labels,
normalized to [0, 1] in the readiness score.

{_markdown_table(world_coverage)}

## Schema Diversity Analysis

Schema entropy is computed over feature-count/type signatures. The schema
diversity score combines schema entropy, feature-count variation, and target
diversity.

{_markdown_table(schema_stats)}

## Statistical Diversity Analysis

The Statistical Diversity Score (SDS) summarizes entropy, skewness, kurtosis,
correlation, and mutual-information diversity.

{_markdown_table(statistical_stats)}

## Benchmark Coverage Analysis

Benchmark Coverage Index (BCI) = {bci_value}. BCI uses standardized meta-feature
vectors. A benchmark dataset is covered when its nearest synthetic dataset lies
within a benchmark-derived nearest-neighbor radius. The final BCI averages
benchmark point coverage and benchmark cluster coverage.

### Benchmark Datasets

{_markdown_table(benchmark_stats)}

### Benchmark Coverage Table

{_markdown_table(bci_table)}

## Distribution Similarity Analysis

For each synthetic dataset, the nearest real benchmark dataset is identified in
meta-feature space. Wasserstein distance, Jensen-Shannon divergence, Gaussian RBF
Maximum Mean Discrepancy, and Kolmogorov-Smirnov statistics are then computed.

### Nearest-Benchmark Ranking

{_markdown_table(nearest_similarity)}

### Global Meta-Feature Similarity

{_markdown_table(distribution_similarity)}

## Foundation Model Readiness

The final score combines:

FMRS = 0.20*WDI + 0.15*TDI + 0.15*FDI + 0.20*BCI + 0.15*SDS + 0.15*SDSchema

Final Foundation Model Readiness Score: **{fmrs_value}**

{_markdown_table(readiness)}

## Strengths

- The corpus is evaluated as a mixture over enterprise, IID, relational,
  temporal, forecasting, scientific, and sparse worlds.
- The evaluation directly measures schema, target, feature-type, statistical,
  missingness, and sparsity diversity.
- Benchmark comparison is based on real downloaded or user-supplied datasets,
  not placeholder statistics.

## Weaknesses and Limitations

- Benchmark coverage depends on which OpenML and local benchmark datasets are
  available at evaluation time.
- Meta-feature similarity does not prove semantic equivalence between synthetic
  and real datasets; it measures coverage of observable statistical regions.
- RelBench and TabArena support requires user-supplied local benchmark exports
  unless stable public APIs are installed and integrated.

## Future Work

- Add task-performance transfer experiments using pretrained TabFM checkpoints.
- Expand benchmark panels with more UCI, OpenML-CC18, RelBench, and TabArena
  datasets.
- Add privacy/memorization tests if any real-data-conditioned generation is ever
  introduced.
"""
    output_path.write_text(textwrap.dedent(text), encoding="utf-8")
    return text


def generate_pdf_report(markdown_text: str, output_path: Path) -> None:
    """Generate a lightweight PDF report from Markdown text.

    The PDF is intentionally dependency-light. It preserves the report text for
    archival use; figures are saved separately as publication-ready assets.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = markdown_text.splitlines()
    pages = []
    current = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=95) or [""]
        for wrapped_line in wrapped:
            current.append(wrapped_line)
            if len(current) >= 48:
                pages.append(current)
                current = []
    if current:
        pages.append(current)

    if not pages:
        pages = [["Empty report"]]

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, "\n".join(page), va="top", ha="left", family="monospace", fontsize=8)
            fig.patch.set_facecolor("white")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
