"""Foundation-model readiness indices for TabFM corpus evaluation."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from evaluation.metrics.corpus_statistics import normalized_entropy


READINESS_COMPONENTS = ["WDI", "TDI", "FDI", "BCI", "SDS", "SDSchema"]


def entropy_index(labels: list[str]) -> float:
    """Normalized Shannon entropy in [0, 1]."""

    return normalized_entropy(labels)


def world_diversity_index(dataset_stats: pd.DataFrame) -> float:
    if dataset_stats.empty:
        return 0.0
    return entropy_index(dataset_stats["world_family"].astype(str).tolist())


def task_diversity_index(dataset_stats: pd.DataFrame) -> float:
    tasks = []
    for target_types in dataset_stats.get("target_type", pd.Series(dtype=str)).fillna("none"):
        tasks.extend([task.strip() for task in str(target_types).split(",") if task.strip() and task != "none"])
    return entropy_index(tasks)


def feature_diversity_index(dataset_stats: pd.DataFrame) -> float:
    if dataset_stats.empty:
        return 0.0
    feature_counts = {
        "numerical": float(dataset_stats["numerical_features"].sum()),
        "categorical": float(dataset_stats["categorical_features"].sum()),
        "boolean": float(dataset_stats["boolean_features"].sum()),
        "temporal": float(dataset_stats["temporal_features"].sum()),
        "identifier": float(dataset_stats["identifier_features"].sum()),
        "sparse": float((dataset_stats["sparsity_percentage"] >= 0.85).sum()),
    }
    expanded = []
    for label, count in feature_counts.items():
        expanded.extend([label] * int(max(0, round(count))))
    return entropy_index(expanded)


def benchmark_coverage_index(
    synthetic_meta: pd.DataFrame,
    benchmark_meta: pd.DataFrame,
    *,
    radius_quantile: float = 0.25,
    n_clusters: int = 8,
) -> tuple[float, pd.DataFrame]:
    """Nearest-neighbor/cluster Benchmark Coverage Index.

    Mathematical definition:

    Let B be benchmark meta-feature vectors and S be synthetic meta-feature
    vectors after standardization on B union S. Define r as the q-th quantile of
    pairwise benchmark-to-benchmark nearest-neighbor distances. A benchmark
    dataset b is covered when min_s ||b - s||_2 <= r. The BCI is the average of
    point coverage and benchmark-cluster coverage:

        BCI = 0.5 * (|{b covered}| / |B|) + 0.5 * (|{clusters covered}| / K)
    """

    if synthetic_meta.empty or benchmark_meta.empty:
        return 0.0, pd.DataFrame()
    numeric_columns = [
        column
        for column in synthetic_meta.columns
        if column in benchmark_meta.columns and pd.api.types.is_numeric_dtype(synthetic_meta[column])
    ]
    if not numeric_columns:
        return 0.0, pd.DataFrame()
    syn = synthetic_meta[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    bench = benchmark_meta[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    combined = scaler.fit_transform(pd.concat([syn, bench], axis=0, ignore_index=True))
    syn_scaled = combined[: len(syn)]
    bench_scaled = combined[len(syn) :]
    syn_to_bench = pairwise_distances(bench_scaled, syn_scaled)
    nearest_synthetic_distance = syn_to_bench.min(axis=1)

    if len(bench_scaled) >= 2:
        bench_distances = pairwise_distances(bench_scaled, bench_scaled)
        np.fill_diagonal(bench_distances, np.inf)
        radius = float(np.quantile(bench_distances.min(axis=1), radius_quantile))
    else:
        radius = float(np.median(nearest_synthetic_distance) + 1e-9)
    covered = nearest_synthetic_distance <= radius
    point_coverage = float(covered.mean())

    cluster_coverage = point_coverage
    cluster_ids = np.zeros(len(bench_scaled), dtype=int)
    if len(bench_scaled) >= 3:
        k = min(n_clusters, len(bench_scaled))
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        cluster_ids = model.fit_predict(bench_scaled)
        cluster_coverage = float(len(set(cluster_ids[covered])) / k)
    bci = float(np.clip(0.5 * point_coverage + 0.5 * cluster_coverage, 0, 1))
    table = pd.DataFrame(
        {
            "benchmark_dataset_id": benchmark_meta["dataset_id"].to_numpy(),
            "benchmark_family": benchmark_meta.get("benchmark_family", benchmark_meta.get("source", "benchmark")).to_numpy(),
            "nearest_synthetic_distance": nearest_synthetic_distance,
            "coverage_radius": radius,
            "covered": covered,
            "benchmark_cluster": cluster_ids,
        }
    )
    return bci, table


def statistical_diversity_score(statistical_stats: pd.DataFrame) -> float:
    if statistical_stats.empty or "statistical_diversity_score" not in statistical_stats:
        return 0.0
    return float(np.clip(statistical_stats["statistical_diversity_score"].mean(), 0, 1))


def schema_diversity_score(schema_stats: pd.DataFrame) -> float:
    if schema_stats.empty or "schema_diversity_score" not in schema_stats:
        return 0.0
    return float(np.clip(schema_stats["schema_diversity_score"].iloc[0], 0, 1))


def readiness_table(
    dataset_stats: pd.DataFrame,
    statistical_stats: pd.DataFrame,
    schema_stats: pd.DataFrame,
    bci: float,
) -> pd.DataFrame:
    """Compute all readiness components and the final FMRS.

    Formula:

        FMRS = 0.20*WDI + 0.15*TDI + 0.15*FDI
             + 0.20*BCI + 0.15*SDS + 0.15*SDSchema
    """

    components = {
        "WDI": world_diversity_index(dataset_stats),
        "TDI": task_diversity_index(dataset_stats),
        "FDI": feature_diversity_index(dataset_stats),
        "BCI": float(np.clip(bci, 0, 1)),
        "SDS": statistical_diversity_score(statistical_stats),
        "SDSchema": schema_diversity_score(schema_stats),
    }
    weights = {
        "WDI": 0.20,
        "TDI": 0.15,
        "FDI": 0.15,
        "BCI": 0.20,
        "SDS": 0.15,
        "SDSchema": 0.15,
    }
    fmrs = float(sum(components[name] * weights[name] for name in READINESS_COMPONENTS))
    rows = [
        {"metric": name, "score": components[name], "weight": weights[name], "weighted_score": components[name] * weights[name]}
        for name in READINESS_COMPONENTS
    ]
    rows.append({"metric": "FMRS", "score": fmrs, "weight": 1.0, "weighted_score": fmrs})
    return pd.DataFrame(rows)


def task_distribution(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for target_types in dataset_stats.get("target_type", pd.Series(dtype=str)).fillna("none"):
        for task in str(target_types).split(","):
            task = task.strip()
            if task and task != "none":
                counter[task] += 1
    total = sum(counter.values()) or 1
    return pd.DataFrame(
        [{"task_type": task, "count": count, "fraction": count / total} for task, count in counter.items()]
    ).sort_values("count", ascending=False)


def feature_distribution(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    if dataset_stats.empty:
        return pd.DataFrame(columns=["feature_type", "count", "fraction"])
    counts = {
        "numerical": int(dataset_stats["numerical_features"].sum()),
        "categorical": int(dataset_stats["categorical_features"].sum()),
        "boolean": int(dataset_stats["boolean_features"].sum()),
        "identifier": int(dataset_stats["identifier_features"].sum()),
        "temporal": int(dataset_stats["temporal_features"].sum()),
        "sparse_dataset": int((dataset_stats["sparsity_percentage"] >= 0.85).sum()),
    }
    total = sum(counts.values()) or 1
    return pd.DataFrame(
        [{"feature_type": key, "count": value, "fraction": value / total} for key, value in counts.items()]
    ).sort_values("count", ascending=False)
