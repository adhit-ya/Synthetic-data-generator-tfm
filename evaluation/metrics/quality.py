"""Synthetic-vs-benchmark quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


META_FEATURE_COLUMNS = [
    "rows",
    "columns",
    "numerical_features",
    "categorical_features",
    "boolean_features",
    "identifier_features",
    "temporal_features",
    "target_count",
    "missing_percentage",
    "sparsity_percentage",
    "skewness_mean",
    "kurtosis_mean",
    "entropy_mean",
    "abs_correlation_mean",
    "mutual_information_mean",
]


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        out[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
    return out


def _histogram(values: np.ndarray, bins: int = 30) -> np.ndarray:
    if len(values) == 0:
        return np.ones(bins) / bins
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or np.nanstd(finite) < 1e-12:
        hist = np.zeros(bins)
        hist[0] = 1.0
        return hist
    hist, _ = np.histogram(finite, bins=bins, density=False)
    hist = hist.astype(float) + 1e-12
    return hist / hist.sum()


def maximum_mean_discrepancy(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    combined = np.vstack([X, Y])
    variance = np.nanmedian(np.var(combined, axis=0))
    gamma = 1.0 / max(2.0 * variance, 1e-6)
    k_xx = rbf_kernel(X, X, gamma=gamma).mean()
    k_yy = rbf_kernel(Y, Y, gamma=gamma).mean()
    k_xy = rbf_kernel(X, Y, gamma=gamma).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def distribution_similarity(synthetic_meta: pd.DataFrame, benchmark_meta: pd.DataFrame) -> pd.DataFrame:
    """Compute official distribution metrics on meta-feature distributions."""

    if synthetic_meta.empty or benchmark_meta.empty:
        return pd.DataFrame()
    syn = _safe_numeric(synthetic_meta, META_FEATURE_COLUMNS)
    bench = _safe_numeric(benchmark_meta, META_FEATURE_COLUMNS)
    rows = []
    for column in META_FEATURE_COLUMNS:
        x = syn[column].to_numpy(dtype=float)
        y = bench[column].to_numpy(dtype=float)
        rows.append(
            {
                "feature": column,
                "wasserstein_distance": float(wasserstein_distance(x, y)),
                "jensen_shannon_divergence": float(jensenshannon(_histogram(x), _histogram(y)) ** 2),
                "ks_statistic": float(ks_2samp(x, y).statistic),
            }
        )
    X = syn.to_numpy(dtype=float)
    Y = bench.to_numpy(dtype=float)
    center = np.nanmean(np.vstack([X, Y]), axis=0)
    scale = np.nanstd(np.vstack([X, Y]), axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    mmd = maximum_mean_discrepancy((X - center) / scale, (Y - center) / scale)
    result = pd.DataFrame(rows)
    result["mmd"] = mmd
    result["mean_rank_score"] = result[
        ["wasserstein_distance", "jensen_shannon_divergence", "ks_statistic", "mmd"]
    ].rank(pct=True).mean(axis=1)
    return result.sort_values("mean_rank_score").reset_index(drop=True)


def benchmark_coverage_score(synthetic_meta: pd.DataFrame, benchmark_meta: pd.DataFrame) -> float:
    """Score how much benchmark meta-feature support is covered by synthetic data."""

    if synthetic_meta.empty or benchmark_meta.empty:
        return 0.0
    syn = _safe_numeric(synthetic_meta, META_FEATURE_COLUMNS)
    bench = _safe_numeric(benchmark_meta, META_FEATURE_COLUMNS)
    coverage_values = []
    for column in META_FEATURE_COLUMNS:
        low = syn[column].quantile(0.05)
        high = syn[column].quantile(0.95)
        if high <= low:
            coverage_values.append(0.0)
        else:
            coverage_values.append(float(((bench[column] >= low) & (bench[column] <= high)).mean()))
    return float(np.mean(coverage_values))


def nearest_dataset_similarity(
    synthetic_meta: pd.DataFrame,
    benchmark_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each synthetic dataset to its nearest benchmark dataset.

    Distances are computed over standardized meta-features. Per-pair metrics are
    defined on the meta-feature vector distributions, which is appropriate here
    because synthetic datasets and benchmark datasets do not share raw feature
    schemas.
    """

    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    if synthetic_meta.empty or benchmark_meta.empty:
        return pd.DataFrame()
    syn = _safe_numeric(synthetic_meta, META_FEATURE_COLUMNS)
    bench = _safe_numeric(benchmark_meta, META_FEATURE_COLUMNS)
    scaler = StandardScaler()
    combined = scaler.fit_transform(pd.concat([syn, bench], axis=0, ignore_index=True))
    syn_scaled = combined[: len(syn)]
    bench_scaled = combined[len(syn) :]
    distances = pairwise_distances(syn_scaled, bench_scaled)
    nearest = np.argmin(distances, axis=1)
    rows = []
    for syn_index, bench_index in enumerate(nearest):
        x = syn_scaled[syn_index]
        y = bench_scaled[int(bench_index)]
        rows.append(
            {
                "synthetic_dataset_id": synthetic_meta.iloc[syn_index]["dataset_id"],
                "nearest_benchmark_dataset_id": benchmark_meta.iloc[int(bench_index)]["dataset_id"],
                "nearest_benchmark_family": benchmark_meta.iloc[int(bench_index)].get("benchmark_family", "benchmark"),
                "meta_feature_distance": float(distances[syn_index, int(bench_index)]),
                "wasserstein_distance": float(wasserstein_distance(x, y)),
                "jensen_shannon_divergence": float(jensenshannon(_histogram(x), _histogram(y)) ** 2),
                "mmd": maximum_mean_discrepancy(x.reshape(1, -1), y.reshape(1, -1)),
                "ks_statistic": float(ks_2samp(x, y).statistic),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["similarity_rank"] = result["meta_feature_distance"].rank(method="dense", ascending=True).astype(int)
    return result.sort_values("meta_feature_distance").reset_index(drop=True)
