"""Real benchmark dataset loaders and meta-feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.io import DatasetRecord
from evaluation.metrics.corpus_statistics import dataset_statistics
from evaluation.metrics.statistical_diversity import metafeatures, statistical_summary

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    family: str
    dataframe: pd.DataFrame


def _sample_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_openml_benchmarks(
    dataset_ids: list[int],
    *,
    max_datasets: int,
    sample_rows: int | None,
    seed: int,
) -> list[BenchmarkDataset]:
    """Download real OpenML benchmark datasets.

    This function intentionally fails soft per dataset so one unavailable OpenML
    entry does not invalidate the whole evaluation.
    """

    if not dataset_ids or max_datasets <= 0:
        return []
    try:
        import openml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("openml is required for real OpenML benchmark downloads.") from exc

    benchmarks: list[BenchmarkDataset] = []
    for dataset_id in dataset_ids[:max_datasets]:
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            frame = X.copy()
            if y is not None:
                frame["benchmark_target"] = y
            frame = _sample_frame(frame, sample_rows, seed)
            benchmarks.append(
                BenchmarkDataset(
                    dataset_id=f"OpenML_{dataset_id}_{dataset.name}",
                    family="OpenML",
                    dataframe=frame,
                )
            )
            LOGGER.info("Loaded OpenML benchmark %s (%s): %s", dataset_id, dataset.name, frame.shape)
        except Exception as exc:
            LOGGER.warning("Skipping OpenML dataset %s due to error: %s", dataset_id, exc)
    return benchmarks


def load_local_benchmarks(paths: list[str], sample_rows: int | None, seed: int) -> list[BenchmarkDataset]:
    """Load real benchmark datasets from local CSV/Parquet files.

    The benchmark family is inferred from the parent directory or filename. This
    supports UCI, RelBench, and TabArena exports without depending on unstable
    repository-specific APIs.
    """

    datasets: list[BenchmarkDataset] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates = []
        if path.is_dir():
            candidates.extend(path.rglob("*.parquet"))
            candidates.extend(path.rglob("*.csv"))
        elif path.exists():
            candidates.append(path)
        for candidate in candidates:
            try:
                if candidate.suffix.lower() == ".parquet":
                    frame = pd.read_parquet(candidate)
                elif candidate.suffix.lower() == ".csv":
                    frame = pd.read_csv(candidate, low_memory=False)
                else:
                    continue
                lower = str(candidate).lower()
                if "relbench" in lower:
                    family = "RelBench"
                elif "tabarena" in lower:
                    family = "TabArena"
                elif "uci" in lower:
                    family = "UCI"
                else:
                    family = "LocalBenchmark"
                datasets.append(
                    BenchmarkDataset(
                        dataset_id=f"{family}_{candidate.stem}",
                        family=family,
                        dataframe=_sample_frame(frame, sample_rows, seed),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping local benchmark %s due to error: %s", candidate, exc)
    return datasets


def benchmark_records(benchmarks: list[BenchmarkDataset]) -> list[DatasetRecord]:
    records = []
    for benchmark in benchmarks:
        records.append(
            DatasetRecord(
                dataset_id=benchmark.dataset_id,
                world_id=benchmark.family,
                table_name=benchmark.dataset_id,
                path=Path("."),
                dataframe=benchmark.dataframe,
            )
        )
    return records


def benchmark_metafeatures(benchmarks: list[BenchmarkDataset]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return benchmark dataset stats, statistical stats, and meta-features."""

    records = benchmark_records(benchmarks)
    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    stats = dataset_statistics(records)
    stats["benchmark_family"] = [record.world_id for record in records]
    statistical = statistical_summary(records)
    meta = metafeatures(stats, statistical, source="benchmark")
    meta["benchmark_family"] = stats["benchmark_family"].to_numpy()
    meta["source"] = meta["benchmark_family"]
    return stats, statistical, meta


def synthetic_vs_benchmark_matrix(synthetic_meta: pd.DataFrame, benchmark_meta: pd.DataFrame) -> pd.DataFrame:
    """Compute nearest benchmark dataset per synthetic dataset in meta-feature space."""

    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    if synthetic_meta.empty or benchmark_meta.empty:
        return pd.DataFrame()
    numeric_columns = [
        column
        for column in synthetic_meta.columns
        if column in benchmark_meta.columns and pd.api.types.is_numeric_dtype(synthetic_meta[column])
    ]
    numeric_columns = [column for column in numeric_columns if column not in {"dataset_id"}]
    if not numeric_columns:
        return pd.DataFrame()
    syn = synthetic_meta[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    bench = benchmark_meta[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    combined = scaler.fit_transform(pd.concat([syn, bench], axis=0, ignore_index=True))
    syn_scaled = combined[: len(syn)]
    bench_scaled = combined[len(syn) :]
    distances = pairwise_distances(syn_scaled, bench_scaled)
    nearest = np.argmin(distances, axis=1)
    rows = []
    for syn_index, bench_index in enumerate(nearest):
        benchmark_row = benchmark_meta.iloc[int(bench_index)]
        rows.append(
            {
                "synthetic_dataset_id": synthetic_meta.iloc[syn_index]["dataset_id"],
                "nearest_benchmark_dataset_id": benchmark_row["dataset_id"],
                "nearest_benchmark_family": benchmark_row.get("benchmark_family", benchmark_row.get("source", "benchmark")),
                "meta_feature_distance": float(distances[syn_index, bench_index]),
            }
        )
    return pd.DataFrame(rows).sort_values("meta_feature_distance").reset_index(drop=True)

