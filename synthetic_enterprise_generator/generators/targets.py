"""Multi-task target generation for TabFM pretraining.

The target priors intentionally lean toward tree-style tabular structure:
threshold rules, gated interactions, categorical effects, and abrupt local
regime changes.  A smaller smooth component remains so the resulting tasks are
hybrid rather than pure decision trees.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _is_generated_target(column: str) -> bool:
    return column.endswith("_target")


def _numeric_matrix(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    excluded = set(exclude)
    numeric = df.select_dtypes(include=[np.number]).copy()
    columns = [
        c
        for c in numeric.columns
        if c not in excluded and not _is_generated_target(c)
    ]
    return numeric[columns]


def _categorical_matrix(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    excluded = set(exclude)
    categorical = df.select_dtypes(include=["object", "category", "bool"]).copy()
    columns = [
        c
        for c in categorical.columns
        if c not in excluded and not _is_generated_target(c)
    ]
    return categorical[columns]


def _robust_numeric_array(df: pd.DataFrame) -> np.ndarray:
    matrix = df.fillna(df.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    if matrix.size == 0:
        return matrix
    center = np.nanmedian(matrix, axis=0)
    spread = np.nanpercentile(matrix, 75, axis=0) - np.nanpercentile(matrix, 25, axis=0)
    spread = np.where(np.abs(spread) < 1e-6, 1.0, spread)
    return np.clip((matrix - center) / spread, -8.0, 8.0)


def _tree_rule_component(
    matrix: np.ndarray,
    rng: np.random.Generator,
    min_rules: int = 5,
    max_rules: int = 14,
) -> np.ndarray:
    """Sample axis-aligned rule conjunctions, the basic shape GBDTs exploit."""

    n_rows, n_features = matrix.shape
    score = np.zeros(n_rows, dtype=float)
    if n_features == 0:
        return score

    n_rules = int(rng.integers(min_rules, max_rules + 1))
    for _ in range(n_rules):
        depth = int(rng.integers(1, min(5, n_features) + 1))
        feature_ids = rng.choice(n_features, size=depth, replace=False)
        active = np.ones(n_rows, dtype=bool)
        for feature_id in feature_ids:
            column = matrix[:, feature_id]
            q = float(rng.uniform(0.15, 0.85))
            threshold = float(np.nanquantile(column, q))
            if rng.random() < 0.5:
                active &= column <= threshold
            else:
                active &= column > threshold
        if not active.any():
            continue
        leaf_value = float(rng.normal(scale=rng.uniform(0.4, 1.6)))
        score[active] += leaf_value

    return score


def _interaction_component(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add implicit feature interactions and gated smooth effects."""

    n_rows, n_features = matrix.shape
    score = np.zeros(n_rows, dtype=float)
    if n_features < 2:
        return score

    n_pairs = int(min(max(2, n_features // 3), 10))
    for _ in range(n_pairs):
        left, right = rng.choice(n_features, size=2, replace=False)
        left_values = matrix[:, left]
        right_values = matrix[:, right]
        threshold = float(np.nanquantile(left_values, rng.uniform(0.2, 0.8)))
        gate = left_values > threshold
        effect = (
            0.35 * left_values * right_values
            + 0.20 * np.abs(left_values - right_values)
            + 0.15 * np.sin(right_values)
        )
        score += rng.normal(scale=0.35) * gate.astype(float) * effect

    return score


def _categorical_effect_component(
    df: pd.DataFrame,
    rng: np.random.Generator,
    max_columns: int = 5,
) -> np.ndarray:
    categorical = _categorical_matrix(df)
    score = np.zeros(len(df), dtype=float)
    if categorical.empty:
        return score

    columns = rng.choice(
        categorical.columns.to_numpy(),
        size=min(max_columns, len(categorical.columns)),
        replace=False,
    )
    for column in columns:
        series = categorical[column].astype("object").where(categorical[column].notna(), "__MISSING__")
        levels = pd.Index(series.unique())
        if levels.empty:
            continue
        values = rng.normal(scale=rng.uniform(0.15, 0.75), size=len(levels))
        mapping = dict(zip(levels, values))
        score += series.map(mapping).fillna(0.0).to_numpy(dtype=float)
    return score


def _hybrid_predictive_score(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_name: str,
) -> np.ndarray:
    """Combine tree-biased rules with smooth and categorical priors."""

    numeric = _numeric_matrix(df, exclude=[target_name])
    if numeric.empty:
        return _categorical_effect_component(df, rng)

    matrix = _robust_numeric_array(numeric)
    tree_score = _tree_rule_component(matrix, rng)
    interaction_score = _interaction_component(matrix, rng)

    weights = rng.normal(scale=0.18, size=matrix.shape[1])
    smooth_score = matrix @ weights
    if matrix.shape[1] >= 1:
        smooth_score += 0.20 * np.sin(matrix[:, 0])
    if matrix.shape[1] >= 3:
        smooth_score += 0.12 * np.tanh(matrix[:, 1] + matrix[:, 2])

    categorical_score = _categorical_effect_component(df, rng)
    score = tree_score + interaction_score + smooth_score + categorical_score
    score += rng.normal(scale=np.std(score) * 0.04 + 1e-6, size=len(df))
    return score


def create_classification_targets(
    df: pd.DataFrame,
    n_classes: int,
    rng: np.random.Generator,
    target_name: str = "classification_target",
) -> pd.DataFrame:
    """Create tree-biased hybrid classification labels from feature structure."""

    out = df.copy()
    score = _hybrid_predictive_score(out, rng, target_name)
    if not np.isfinite(score).any() or np.std(score) < 1e-9:
        out[target_name] = rng.integers(0, n_classes, size=len(out))
        return out
    bins = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    out[target_name] = np.digitize(score, bins).astype(int)
    return out


def create_regression_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_name: str = "regression_target",
) -> pd.DataFrame:
    """Create a regression target with rule, threshold, and smooth structure."""

    out = df.copy()
    y = _hybrid_predictive_score(out, rng, target_name)
    if not np.isfinite(y).any() or np.std(y) < 1e-9:
        out[target_name] = rng.normal(size=len(out))
        return out
    out[target_name] = y.astype(float)
    return out


def create_ordinal_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_levels: int = 5,
    target_name: str = "ordinal_target",
) -> pd.DataFrame:
    """Create ordered bucket labels from the hybrid predictive prior."""

    out = df.copy()
    score = _hybrid_predictive_score(out, rng, target_name)
    if not np.isfinite(score).any() or np.std(score) < 1e-9:
        out[target_name] = rng.integers(0, n_levels, size=len(out))
        return out
    cutpoints = np.quantile(score, np.linspace(0, 1, n_levels + 1)[1:-1])
    out[target_name] = np.digitize(score, cutpoints).astype(int)
    return out


def create_count_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_name: str = "count_target",
) -> pd.DataFrame:
    """Create overdispersed count labels with threshold-driven intensity."""

    out = df.copy()
    score = _hybrid_predictive_score(out, rng, target_name)
    if not np.isfinite(score).any() or np.std(score) < 1e-9:
        out[target_name] = rng.poisson(lam=1.0, size=len(out)).astype(int)
        return out
    normalized = (score - np.median(score)) / (np.std(score) + 1e-6)
    intensity = np.exp(np.clip(0.45 * normalized, -2.0, 2.2))
    gamma_noise = rng.gamma(shape=1.5, scale=1.0 / 1.5, size=len(out))
    out[target_name] = rng.poisson(lam=np.maximum(0.02, intensity * gamma_noise)).astype(int)
    return out


def create_forecasting_targets(
    df: pd.DataFrame,
    value_column: Optional[str] = None,
    group_column: Optional[str] = "customer_id",
    horizon: int = 1,
    target_name: str = "forecast_target",
) -> pd.DataFrame:
    """Create next-horizon value targets for temporal forecasting."""

    out = df.copy()
    numeric = _numeric_matrix(out, exclude=[target_name])
    if value_column is None:
        value_column = numeric.columns[0] if not numeric.empty else None
    if value_column is None:
        out[target_name] = np.nan
        return out
    sort_cols = [c for c in [group_column, "timestamp"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if group_column and group_column in out.columns:
        out[target_name] = out.groupby(group_column)[value_column].shift(-horizon)
    else:
        out[target_name] = out[value_column].shift(-horizon)
    return out


def create_next_event_targets(
    df: pd.DataFrame,
    event_column: str = "event_type",
    group_column: str = "sequence_id",
    target_name: str = "next_event_target",
) -> pd.DataFrame:
    """Create next-event labels for workflow/event-sequence modeling."""

    out = df.copy()
    if event_column not in out.columns:
        out[target_name] = None
        return out
    sort_cols = [c for c in [group_column, "event_index", "timestamp"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if group_column in out.columns:
        out[target_name] = out.groupby(group_column)[event_column].shift(-1)
    else:
        out[target_name] = out[event_column].shift(-1)
    return out


def create_imputation_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
    feature_name_column: str = "imputation_feature_name",
    target_name: str = "imputation_target",
) -> pd.DataFrame:
    """Create row-wise holdout labels for masked-feature reconstruction.

    The actual masking engine runs later; this target records a recoverable
    numeric value and the feature it came from so pretraining can include an
    explicit imputation objective.
    """

    out = df.copy()
    numeric = _numeric_matrix(out).drop(columns=[target_name], errors="ignore")
    if numeric.empty:
        out[feature_name_column] = None
        out[target_name] = np.nan
        return out
    candidate_columns = numeric.columns.tolist()
    selected_indices = rng.integers(0, len(candidate_columns), size=len(out))
    selected_columns = np.asarray(candidate_columns, dtype=object)[selected_indices]
    values = numeric.to_numpy(dtype=float, copy=False)[
        np.arange(len(out)), selected_indices
    ]
    out[feature_name_column] = selected_columns
    out[target_name] = values
    return out


def create_anomaly_targets(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float = 0.03,
    target_name: str = "anomaly_target",
) -> pd.DataFrame:
    """Mark a small fraction of high-distance rows as anomaly labels."""

    out = df.copy()
    X = _numeric_matrix(out).fillna(0.0)
    if X.empty:
        out[target_name] = (rng.random(len(out)) < rate).astype(int)
        return out
    z = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))
    distance = np.sqrt((z**2).sum(axis=1))
    cutoff = np.quantile(distance, max(0.0, 1.0 - rate))
    out[target_name] = (distance >= cutoff).astype(int)
    return out
