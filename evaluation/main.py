"""Single-command corpus evaluation entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from evaluation.config import load_evaluation_config


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated TabFM synthetic corpus.")
    parser.add_argument("--config", default="evaluation/configs/default.yaml", help="Evaluation YAML config.")
    parser.add_argument("--corpus-dir", default=None, help="Override generated corpus directory.")
    parser.add_argument("--output-dir", default=None, help="Override evaluation output directory.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Override max rows loaded per dataset.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip OpenML/local benchmark loading.")
    return parser.parse_args()


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8"),
        ],
    )


def _missingness_bins(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    if dataset_stats.empty:
        return pd.DataFrame()
    out = dataset_stats[["dataset_id", "missing_percentage", "sparsity_percentage"]].copy()
    out["missingness_band"] = pd.cut(
        out["missing_percentage"],
        bins=[-0.001, 0.05, 0.25, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    return out


def run_evaluation(config_path: str | Path, args: argparse.Namespace | None = None) -> dict[str, pd.DataFrame]:
    import pandas as pd

    from evaluation.benchmark_analysis.loaders import (
        benchmark_metafeatures,
        load_local_benchmarks,
        load_openml_benchmarks,
    )
    from evaluation.io import ensure_output_dirs, load_corpus, write_tables
    from evaluation.metrics.corpus_statistics import (
        dataset_statistics,
        global_statistics,
        schema_diversity,
        world_coverage,
    )
    from evaluation.metrics.quality import distribution_similarity, nearest_dataset_similarity
    from evaluation.metrics.readiness import (
        benchmark_coverage_index,
        feature_distribution,
        readiness_table,
        task_distribution,
    )
    from evaluation.metrics.statistical_diversity import metafeatures, statistical_summary
    from evaluation.reports.generate_report import generate_markdown_report, generate_pdf_report
    from evaluation.visualizations.plots import (
        create_sankey,
        plot_benchmark_outputs,
        plot_embeddings,
        plot_example_correlation,
        plot_missingness_sparsity,
        plot_morphology,
        plot_radar,
        plot_statistical,
        plot_task_feature_charts,
        plot_world_coverage,
        setup_style,
    )

    config = load_evaluation_config(config_path)
    if args is not None:
        if args.corpus_dir:
            config.corpus_dir = args.corpus_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.sample_rows is not None:
            config.sample_rows_per_dataset = args.sample_rows
        if args.skip_benchmarks:
            config.benchmarks.enabled = False

    output_root = Path(config.output_dir)
    dirs = ensure_output_dirs(output_root)
    _setup_logging(output_root)
    setup_style()
    np.random.seed(config.seed)

    LOGGER.info("Loading synthetic corpus from %s", config.corpus_dir)
    records = load_corpus(config.corpus_dir, sample_rows=config.sample_rows_per_dataset)
    if not records:
        raise RuntimeError(
            f"No exported datasets found in {config.corpus_dir}. Run the generator first, then point --corpus-dir to that output."
        )
    LOGGER.info("Loaded %d synthetic datasets.", len(records))

    dataset_stats = dataset_statistics(records)
    corpus_summary = global_statistics(dataset_stats)
    coverage = world_coverage(dataset_stats)
    schema_stats = schema_diversity(dataset_stats)
    statistical_stats = statistical_summary(records)
    synthetic_meta = metafeatures(dataset_stats, statistical_stats, source="Synthetic")
    missingness_bands = _missingness_bins(dataset_stats)
    tasks = task_distribution(dataset_stats)
    features = feature_distribution(dataset_stats)

    benchmark_stats = pd.DataFrame()
    benchmark_statistical = pd.DataFrame()
    benchmark_meta = pd.DataFrame()
    if config.benchmarks.enabled:
        benchmarks = []
        if config.benchmarks.openml.enabled:
            benchmarks.extend(
                load_openml_benchmarks(
                    config.benchmarks.openml.dataset_ids,
                    max_datasets=config.benchmarks.openml.max_datasets,
                    sample_rows=config.sample_rows_per_dataset,
                    seed=config.seed,
                )
            )
        benchmarks.extend(load_local_benchmarks(config.benchmarks.local_paths, config.sample_rows_per_dataset, config.seed))
        benchmark_stats, benchmark_statistical, benchmark_meta = benchmark_metafeatures(benchmarks)
        LOGGER.info("Loaded %d benchmark datasets.", len(benchmarks))

    all_meta = synthetic_meta.copy()
    if not benchmark_meta.empty:
        all_meta = pd.concat([synthetic_meta, benchmark_meta], axis=0, ignore_index=True)

    bci, bci_table = benchmark_coverage_index(synthetic_meta, benchmark_meta)
    dist_similarity = distribution_similarity(synthetic_meta, benchmark_meta)
    nearest_similarity = nearest_dataset_similarity(synthetic_meta, benchmark_meta)
    readiness = readiness_table(dataset_stats, statistical_stats, schema_stats, bci)

    tables = {
        "dataset_summary": dataset_stats,
        "corpus_summary": corpus_summary,
        "world_distribution": coverage,
        "schema_diversity": schema_stats,
        "missingness_sparsity": missingness_bands,
        "statistical_diversity": statistical_stats,
        "synthetic_metafeatures": synthetic_meta,
        "benchmark_dataset_summary": benchmark_stats,
        "benchmark_statistical_diversity": benchmark_statistical,
        "benchmark_metafeatures": benchmark_meta,
        "distribution_similarity": dist_similarity,
        "nearest_benchmark_similarity": nearest_similarity,
        "benchmark_coverage": bci_table,
        "task_distribution": tasks,
        "feature_distribution": features,
        "readiness_scores": readiness,
    }
    write_tables(tables, dirs["tables"])

    formats = config.visualization.formats
    dpi = config.visualization.dpi
    plot_world_coverage(coverage, dirs["figures"], formats, dpi)
    plot_morphology(dataset_stats, dirs["figures"], formats, dpi)
    plot_missingness_sparsity(dataset_stats, dirs["figures"], formats, dpi)
    plot_statistical(statistical_stats, dirs["figures"], formats, dpi)
    plot_example_correlation(records, dirs["figures"], formats, dpi, max_features=config.max_correlation_features)
    embeddings = plot_embeddings(all_meta, dirs["figures"], formats, dpi, config.seed)
    for name, frame in embeddings.items():
        if not frame.empty:
            frame.to_csv(dirs["tables"] / f"{name}_embedding.csv", index=False)
    plot_benchmark_outputs(nearest_similarity, bci_table, dirs["figures"], formats, dpi)
    plot_radar(readiness, dirs["figures"], "foundation_model_readiness_radar", "Foundation Model Readiness", formats, dpi)
    plot_radar(readiness[readiness["metric"].isin(["WDI", "TDI", "FDI", "SDS", "SDSchema"])], dirs["figures"], "diversity_radar", "Diversity Metrics", formats, dpi)
    plot_task_feature_charts(tasks, features, dirs["figures"], formats, dpi)
    create_sankey(dataset_stats, dirs["figures"])

    markdown = generate_markdown_report(
        output_path=dirs["reports"] / "corpus_evaluation_report.md",
        title=config.report.title,
        dataset_stats=dataset_stats,
        corpus_summary=corpus_summary,
        world_coverage=coverage,
        schema_stats=schema_stats,
        statistical_stats=statistical_stats,
        benchmark_stats=benchmark_stats,
        distribution_similarity=dist_similarity,
        nearest_similarity=nearest_similarity,
        readiness=readiness,
        bci_table=bci_table,
    )
    if config.report.generate_pdf:
        generate_pdf_report(markdown, dirs["reports"] / "corpus_evaluation_report.pdf")

    LOGGER.info("Evaluation complete. Outputs written to %s", output_root.resolve())
    return tables


def main() -> None:
    args = parse_args()
    run_evaluation(args.config, args)


if __name__ == "__main__":
    main()
