"""Publication-quality plots for corpus evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from evaluation.metrics.quality import META_FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
        }
    )


def save_figure(fig: plt.Figure, path_base: Path, formats: list[str], dpi: int = 300) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        try:
            fig.savefig(path_base.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=dpi)
        except Exception as exc:
            LOGGER.warning("Could not save %s as %s: %s", path_base.name, fmt, exc)
    plt.close(fig)


def _empty_plot(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig


def plot_world_coverage(world_coverage: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    if world_coverage.empty:
        save_figure(_empty_plot("No world coverage data"), figure_dir / "world_coverage_bar", formats, dpi)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=world_coverage, x="world_family", y="dataset_count", ax=ax, palette="tab10")
    ax.set_title("World Coverage")
    ax.set_xlabel("World family")
    ax.set_ylabel("Dataset count")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, figure_dir / "world_coverage_bar", formats, dpi)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        world_coverage["dataset_count"],
        labels=world_coverage["world_family"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("World Distribution")
    save_figure(fig, figure_dir / "world_coverage_pie", formats, dpi)

    heat = world_coverage.set_index("world_family")[["dataset_count", "row_count", "coverage_fraction"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heat, annot=True, fmt=".3g", cmap="viridis", ax=ax)
    ax.set_title("World Coverage Heatmap")
    save_figure(fig, figure_dir / "world_coverage_heatmap", formats, dpi)


def plot_morphology(dataset_stats: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    if dataset_stats.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(dataset_stats["rows"], bins=30, kde=True, ax=ax)
    ax.set_title("Dataset Size Distribution")
    ax.set_xlabel("Rows")
    save_figure(fig, figure_dir / "dataset_size_histogram", formats, dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(dataset_stats["columns"], bins=30, kde=True, ax=ax)
    ax.set_title("Feature Count Distribution")
    ax.set_xlabel("Columns")
    save_figure(fig, figure_dir / "feature_count_histogram", formats, dpi)

    long = dataset_stats.melt(
        id_vars=["dataset_id", "world_family"],
        value_vars=[
            "numerical_features",
            "categorical_features",
            "boolean_features",
            "identifier_features",
            "temporal_features",
        ],
        var_name="feature_type",
        value_name="count",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=long, x="feature_type", y="count", estimator=sum, errorbar=None, ax=ax)
    ax.set_title("Feature-Type Composition")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, figure_dir / "feature_type_composition", formats, dpi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=dataset_stats, x="world_family", y="columns", ax=ax)
    ax.set_title("Schema Size by World Family")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, figure_dir / "schema_size_boxplot", formats, dpi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(data=dataset_stats, x="world_family", y="columns", ax=ax, inner="quartile")
    ax.set_title("Schema Size Violin Plot")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, figure_dir / "schema_size_violin", formats, dpi)

    schema_heat = dataset_stats.set_index("dataset_id")[
        [
            "numerical_features",
            "categorical_features",
            "boolean_features",
            "identifier_features",
            "temporal_features",
            "target_count",
        ]
    ]
    fig, ax = plt.subplots(figsize=(9, max(4, min(12, len(schema_heat) * 0.28))))
    sns.heatmap(schema_heat, cmap="mako", ax=ax)
    ax.set_title("Schema Diversity Heatmap")
    save_figure(fig, figure_dir / "schema_diversity_heatmap", formats, dpi)


def plot_missingness_sparsity(dataset_stats: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    if dataset_stats.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(dataset_stats["missing_percentage"], bins=30, kde=True, ax=ax)
    ax.set_title("Missingness Distribution")
    ax.set_xlabel("Missingness percentage")
    save_figure(fig, figure_dir / "missingness_distribution", formats, dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(dataset_stats["sparsity_percentage"], bins=30, kde=True, ax=ax)
    ax.set_title("Sparsity Distribution")
    ax.set_xlabel("Sparsity percentage")
    save_figure(fig, figure_dir / "sparsity_distribution", formats, dpi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=dataset_stats, x="world_family", y="sparsity_percentage", ax=ax)
    ax.set_title("Sparsity by World Family")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, figure_dir / "sparsity_boxplot", formats, dpi)

    heat = dataset_stats.set_index("dataset_id")[["missing_percentage", "sparsity_percentage"]]
    fig, ax = plt.subplots(figsize=(7, max(4, min(12, len(heat) * 0.28))))
    sns.heatmap(heat, annot=False, cmap="rocket_r", ax=ax)
    ax.set_title("Missingness and Sparsity Heatmap")
    save_figure(fig, figure_dir / "missingness_heatmap", formats, dpi)


def plot_statistical(statistical_stats: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    if statistical_stats.empty:
        return
    for column, title in [
        ("entropy_mean", "Entropy Distribution"),
        ("skewness_mean", "Skewness Distribution"),
        ("kurtosis_mean", "Kurtosis Distribution"),
        ("abs_correlation_mean", "Correlation Distribution"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(statistical_stats[column].fillna(0), bins=30, kde=True, ax=ax)
        ax.set_title(title)
        save_figure(fig, figure_dir / column.replace("_mean", "_distribution"), formats, dpi)


def plot_example_correlation(records, figure_dir: Path, formats: list[str], dpi: int, max_features: int = 60) -> None:
    if not records:
        return
    best = max(records, key=lambda r: r.dataframe.select_dtypes(include=[np.number]).shape[1])
    numeric = best.dataframe.select_dtypes(include=[np.number])
    numeric = numeric[[c for c in numeric.columns if not c.endswith("_target")]].iloc[:, :max_features]
    if numeric.shape[1] < 2:
        save_figure(_empty_plot("No numeric correlation matrix available"), figure_dir / "correlation_heatmap", formats, dpi)
        return
    corr = numeric.corr().fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation Heatmap: {best.dataset_id}")
    save_figure(fig, figure_dir / "correlation_heatmap", formats, dpi)


def _embedding_frame(meta: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    available = [column for column in META_FEATURE_COLUMNS if column in meta.columns]
    if meta.empty or len(meta) < 2 or not available:
        return pd.DataFrame()
    X = meta[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = StandardScaler().fit_transform(X)
    if method == "pca":
        coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "tsne":
        perplexity = max(2, min(30, len(meta) - 1))
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto").fit_transform(X)
    elif method == "umap":
        try:
            import umap
        except ModuleNotFoundError:
            LOGGER.warning("umap-learn is not installed; skipping UMAP plot.")
            return pd.DataFrame()
        n_neighbors = max(2, min(15, len(meta) - 1))
        coords = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors).fit_transform(X)
    else:
        raise ValueError(method)
    out = meta[["dataset_id", "source"]].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out


def plot_embeddings(meta: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int, seed: int) -> dict[str, pd.DataFrame]:
    embeddings = {}
    for method, title in [("pca", "PCA Visualization"), ("tsne", "t-SNE Visualization"), ("umap", "UMAP Visualization")]:
        emb = _embedding_frame(meta, method, seed)
        embeddings[method] = emb
        if emb.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=emb, x="x", y="y", hue="source", style="source", s=70, ax=ax)
        ax.set_title(title)
        save_figure(fig, figure_dir / f"{method}_visualization", formats, dpi)
    return embeddings


def plot_benchmark_outputs(
    nearest_similarity: pd.DataFrame,
    coverage_table: pd.DataFrame,
    figure_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    if not nearest_similarity.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        top = nearest_similarity.head(25)
        sns.barplot(data=top, x="meta_feature_distance", y="synthetic_dataset_id", hue="nearest_benchmark_family", ax=ax)
        ax.set_title("Benchmark Similarity Ranking")
        ax.set_xlabel("Nearest benchmark distance")
        ax.set_ylabel("Synthetic dataset")
        save_figure(fig, figure_dir / "benchmark_similarity_ranking", formats, dpi)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=nearest_similarity,
            x="meta_feature_distance",
            y="wasserstein_distance",
            hue="nearest_benchmark_family",
            ax=ax,
        )
        ax.set_title("Benchmark Coverage Scatter Plot")
        save_figure(fig, figure_dir / "benchmark_coverage_scatter", formats, dpi)

    if not coverage_table.empty:
        pivot = coverage_table.pivot_table(
            index="benchmark_family",
            columns="covered",
            values="benchmark_dataset_id",
            aggfunc="count",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt="g", cmap="crest", ax=ax)
        ax.set_title("Benchmark Coverage Heatmap")
        save_figure(fig, figure_dir / "benchmark_coverage_heatmap", formats, dpi)


def plot_radar(table: pd.DataFrame, figure_dir: Path, name: str, title: str, formats: list[str], dpi: int) -> None:
    rows = table[table["metric"] != "FMRS"] if "metric" in table.columns else table
    if rows.empty:
        return
    labels = rows["metric"].astype(str).tolist()
    values = rows["score"].astype(float).tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    save_figure(fig, figure_dir / name, formats, dpi)


def plot_task_feature_charts(task_dist: pd.DataFrame, feature_dist: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    if not task_dist.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=task_dist, x="task_type", y="count", ax=ax)
        ax.set_title("Task Diversity")
        ax.tick_params(axis="x", rotation=35)
        save_figure(fig, figure_dir / "task_diversity_chart", formats, dpi)
    if not feature_dist.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=feature_dist, x="feature_type", y="count", ax=ax)
        ax.set_title("Feature Diversity")
        ax.tick_params(axis="x", rotation=35)
        save_figure(fig, figure_dir / "feature_diversity_chart", formats, dpi)


def create_sankey(dataset_stats: pd.DataFrame, figure_dir: Path) -> None:
    if dataset_stats.empty:
        return
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        LOGGER.warning("plotly is not installed; skipping Sankey diagram.")
        return
    links = []
    labels = []
    label_index = {}

    def idx(label: str) -> int:
        if label not in label_index:
            label_index[label] = len(labels)
            labels.append(label)
        return label_index[label]

    for _, row in dataset_stats.iterrows():
        world = str(row["world_family"])
        tasks = [task for task in str(row["target_type"]).split(",") if task and task != "none"]
        feature_types = []
        for feature_type, column in [
            ("numerical", "numerical_features"),
            ("categorical", "categorical_features"),
            ("boolean", "boolean_features"),
            ("identifier", "identifier_features"),
            ("temporal", "temporal_features"),
        ]:
            if row[column] > 0:
                feature_types.append(feature_type)
        if row["sparsity_percentage"] >= 0.85:
            feature_types.append("sparse")
        for task in tasks or ["none"]:
            links.append((idx(world), idx(task), 1))
            for feature_type in feature_types:
                links.append((idx(task), idx(feature_type), 1))

    if not links:
        return
    source, target, value = zip(*links)
    fig = go.Figure(data=[go.Sankey(node={"label": labels}, link={"source": source, "target": target, "value": value})])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_dir / "world_task_feature_sankey.html")
    try:
        fig.write_image(figure_dir / "world_task_feature_sankey.png", scale=2)
        fig.write_image(figure_dir / "world_task_feature_sankey.pdf")
        fig.write_image(figure_dir / "world_task_feature_sankey.svg")
    except Exception as exc:
        LOGGER.warning("Static Sankey export requires kaleido; wrote HTML only. Error: %s", exc)

