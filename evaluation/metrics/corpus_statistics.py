"""Dataset-level and global corpus statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.io import DatasetRecord


ID_TOKENS = ("_id", "id", "uuid", "key")
TEMPORAL_TOKENS = ("timestamp", "date", "time", "hour", "day", "week", "month", "quarter")


def is_target_column(column: str) -> bool:
    return column.endswith("_target") or column in {"target", "label", "y"}


def is_identifier_column(column: str, series: pd.Series) -> bool:
    lower = column.lower()
    if any(token in lower for token in ID_TOKENS):
        return True
    if series.dtype == object and series.nunique(dropna=True) > max(20, len(series) * 0.8):
        return True
    return False


def is_temporal_column(column: str, series: pd.Series) -> bool:
    lower = column.lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(token in lower for token in TEMPORAL_TOKENS):
        return True
    if series.dtype == object and ("date" in lower or "time" in lower):
        parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
        return bool(parsed.notna().mean() > 0.8) if len(parsed) else False
    return False


def is_boolean_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    values = set(non_null.astype(str).str.lower().unique())
    return values.issubset({"true", "false", "0", "1", "yes", "no"})


def infer_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer feature groups using stable naming and dtype heuristics."""

    groups = {
        "target": [],
        "identifier": [],
        "temporal": [],
        "boolean": [],
        "numerical": [],
        "categorical": [],
    }
    for column in df.columns:
        series = df[column]
        if is_target_column(column):
            groups["target"].append(column)
        elif is_identifier_column(column, series):
            groups["identifier"].append(column)
        elif is_temporal_column(column, series):
            groups["temporal"].append(column)
        elif is_boolean_column(series):
            groups["boolean"].append(column)
        elif pd.api.types.is_numeric_dtype(series):
            groups["numerical"].append(column)
        else:
            groups["categorical"].append(column)
    return groups


def infer_target_types(df: pd.DataFrame, target_columns: Iterable[str]) -> str:
    types = []
    for column in target_columns:
        series = df[column].dropna()
        lower = column.lower()
        if series.empty:
            types.append("unknown")
        elif "forecast" in lower:
            types.append("forecasting")
        elif "anomaly" in lower:
            types.append("anomaly_detection")
        elif "imputation" in lower:
            types.append("imputation")
        elif "ordinal" in lower:
            types.append("ordinal")
        elif "count" in lower:
            types.append("count")
        elif pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            types.append("regression")
        elif pd.api.types.is_numeric_dtype(series):
            types.append("classification")
        else:
            types.append("classification")
    return ",".join(sorted(set(types))) if types else "none"


def compute_sparsity(df: pd.DataFrame, feature_columns: list[str]) -> float:
    if not feature_columns or df.empty:
        return 0.0
    features = df[feature_columns]
    missing = features.isna().to_numpy()
    numeric = features.select_dtypes(include=[np.number])
    zero_cells = np.zeros(features.shape, dtype=bool)
    if not numeric.empty:
        numeric_positions = [features.columns.get_loc(column) for column in numeric.columns]
        zero_cells[:, numeric_positions] = numeric.fillna(np.nan).to_numpy(dtype=float) == 0
    return float(np.logical_or(missing, zero_cells).sum() / features.size)


def classify_world(record: DatasetRecord, groups: dict[str, list[str]]) -> str:
    """Classify exported table into a world family."""

    frame = record.dataframe
    values = []
    for column in ("world_type", "world_id"):
        if column in frame.columns:
            values.extend(frame[column].dropna().astype(str).str.lower().unique().tolist())
    text = " ".join([record.world_id, record.table_name, *values]).lower()
    columns = " ".join(frame.columns.astype(str)).lower()
    if "advanced_forecasting" in text or "forecast" in columns:
        return "Forecasting Worlds"
    if "scientific" in text or any(token in columns for token in ("sci_", "assay", "experiment", "lab_batch")):
        return "Scientific Worlds"
    if "sparse" in text or compute_sparsity(frame, groups["numerical"] + groups["categorical"]) >= 0.85:
        return "Sparse Worlds"
    if "relational" in text or record.table_name in {"customers", "products", "transactions", "entity_graph_edges"}:
        return "Relational Worlds"
    if "enterprise" in text or any(token in columns for token in ("workflow", "event_type", "session_id")):
        return "Enterprise Worlds"
    if "temporal" in text or groups["temporal"]:
        return "Temporal Worlds"
    return "IID Statistical Worlds"


def dataset_statistics(records: list[DatasetRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        df = record.dataframe
        groups = infer_feature_groups(df)
        feature_columns = [
            column
            for kind in ("numerical", "categorical", "boolean", "identifier", "temporal")
            for column in groups[kind]
        ]
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "world_id": record.world_id,
                "table_name": record.table_name,
                "world_family": classify_world(record, groups),
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "numerical_features": len(groups["numerical"]),
                "categorical_features": len(groups["categorical"]),
                "boolean_features": len(groups["boolean"]),
                "identifier_features": len(groups["identifier"]),
                "temporal_features": len(groups["temporal"]),
                "target_count": len(groups["target"]),
                "target_type": infer_target_types(df, groups["target"]),
                "missing_count": int(df.isna().sum().sum()),
                "missing_percentage": float(df.isna().mean().mean()) if not df.empty else 0.0,
                "sparsity_percentage": compute_sparsity(df, feature_columns),
            }
        )
    return pd.DataFrame(rows)


def shannon_entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total + 1e-12) for count in counts.values()))


def normalized_entropy(values: Iterable[str]) -> float:
    values = list(values)
    if not values:
        return 0.0
    entropy = shannon_entropy(values)
    unique = len(set(values))
    return float(entropy / math.log(unique)) if unique > 1 else 0.0


def global_statistics(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    if dataset_stats.empty:
        return pd.DataFrame()
    numeric_totals = dataset_stats[
        [
            "numerical_features",
            "categorical_features",
            "boolean_features",
            "identifier_features",
            "temporal_features",
        ]
    ].sum()
    return pd.DataFrame(
        [
            {
                "total_datasets": int(len(dataset_stats)),
                "total_rows": int(dataset_stats["rows"].sum()),
                "total_columns": int(dataset_stats["columns"].sum()),
                "average_rows": float(dataset_stats["rows"].mean()),
                "average_columns": float(dataset_stats["columns"].mean()),
                "min_rows": int(dataset_stats["rows"].min()),
                "max_rows": int(dataset_stats["rows"].max()),
                "min_columns": int(dataset_stats["columns"].min()),
                "max_columns": int(dataset_stats["columns"].max()),
                "feature_type_distribution": numeric_totals.to_dict(),
            }
        ]
    )


def world_coverage(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    if dataset_stats.empty:
        return pd.DataFrame(columns=["world_family", "dataset_count", "row_count", "coverage_fraction"])
    grouped = dataset_stats.groupby("world_family", as_index=False).agg(
        dataset_count=("dataset_id", "count"),
        row_count=("rows", "sum"),
    )
    grouped["coverage_fraction"] = grouped["dataset_count"] / grouped["dataset_count"].sum()
    return grouped.sort_values("dataset_count", ascending=False).reset_index(drop=True)


def schema_diversity(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    if dataset_stats.empty:
        return pd.DataFrame()
    type_signatures = dataset_stats[
        [
            "numerical_features",
            "categorical_features",
            "boolean_features",
            "identifier_features",
            "temporal_features",
            "target_count",
        ]
    ].astype(str).agg("|".join, axis=1)
    schema_entropy = normalized_entropy(type_signatures)
    size_cv = float(dataset_stats["columns"].std(ddof=0) / (dataset_stats["columns"].mean() + 1e-9))
    target_entropy = normalized_entropy(
        target
        for targets in dataset_stats["target_type"].fillna("none")
        for target in str(targets).split(",")
    )
    score = float(np.clip(0.45 * schema_entropy + 0.25 * min(size_cv, 1.0) + 0.30 * target_entropy, 0, 1))
    return pd.DataFrame(
        [
            {
                "schema_entropy": schema_entropy,
                "schema_size_cv": size_cv,
                "target_entropy": target_entropy,
                "schema_diversity_score": score,
            }
        ]
    )

