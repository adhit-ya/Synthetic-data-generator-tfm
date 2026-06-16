"""Forecasting-specific temporal dynamics for synthetic tabular worlds.

The base temporal augmentation adds timestamps, calendar fields, seasonality,
and simple distribution shifts. This module layers forecasting processes on top
of those tables: autoregression, trend, seasonality, shocks, drift, and regime
changes. It intentionally keeps the output tabular so the same export and
target-generation pipeline can consume it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastingDynamicsSpec:
    """Controls the stochastic process used for advanced forecasting worlds."""

    min_series: int = 4
    max_series: int = 64
    min_autoregressive_order: int = 1
    max_autoregressive_order: int = 5
    shock_probability: float = 0.035
    regime_change_probability: float = 0.45
    concept_drift_strength: float = 0.35
    seasonal_periods: tuple[int, ...] = (7, 30, 365)


def _safe_scale(values: np.ndarray) -> float:
    scale = float(np.nanstd(values))
    return scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def _numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded and not column.endswith("_target")
    ]


def _choose_value_column(df: pd.DataFrame, value_column: Optional[str]) -> Optional[str]:
    if value_column is not None and value_column in df.columns:
        return value_column
    candidates = _numeric_feature_columns(
        df,
        exclude={
            "hour",
            "day_of_week",
            "week_of_year",
            "month",
            "quarter",
            "is_weekend",
            "regime_id",
        },
    )
    return candidates[0] if candidates else None


def add_forecasting_dynamics(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    group_column: str = "series_id",
    value_column: Optional[str] = None,
    spec: ForecastingDynamicsSpec | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Add realistic forecasting dynamics to an already-temporal table.

    Parameters
    ----------
    df:
        A table that preferably already contains ``timestamp`` and a series
        identifier. If either is absent, the function still returns a valid
        table but will only add the dynamics it can infer.
    rng:
        Numpy random generator used for deterministic corpus creation.
    group_column:
        Column containing independent time-series identifiers.
    value_column:
        Optional source numeric column to transform into the observed forecast
        value. When omitted, the first suitable numeric feature is used.
    spec:
        Optional dynamics configuration.

    Returns
    -------
    tuple[pd.DataFrame, str | None]
        The augmented table and the name of the observed value column.
    """

    dynamics = spec or ForecastingDynamicsSpec()
    out = df.copy()
    selected_value_column = _choose_value_column(out, value_column)
    if selected_value_column is None:
        out["forecast_observed_value"] = np.nan
        return out, None

    if group_column not in out.columns:
        n_series = int(rng.integers(dynamics.min_series, dynamics.max_series + 1))
        n_series = max(1, min(n_series, max(1, len(out))))
        out[group_column] = rng.choice([f"SERIES_{i:05d}" for i in range(n_series)], size=len(out))

    sort_columns = [column for column in [group_column, "timestamp"] if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns).reset_index(drop=True)

    observed = np.zeros(len(out), dtype=float)
    trend = np.zeros(len(out), dtype=float)
    seasonal = np.zeros(len(out), dtype=float)
    ar_component = np.zeros(len(out), dtype=float)
    shock = np.zeros(len(out), dtype=float)
    drift = np.zeros(len(out), dtype=float)
    regime = np.zeros(len(out), dtype=int)

    source_values = out[selected_value_column].fillna(out[selected_value_column].median()).fillna(0.0).to_numpy(dtype=float)
    global_scale = _safe_scale(source_values)

    grouped_indices = out.groupby(group_column, sort=False).indices
    for _, index_values in grouped_indices.items():
        idx = np.asarray(index_values, dtype=int)
        n = len(idx)
        if n == 0:
            continue

        base = source_values[idx]
        local_scale = _safe_scale(base)
        time = np.arange(n, dtype=float)
        normalized_time = time / max(n - 1, 1)

        slope = rng.normal(scale=0.55 * local_scale)
        local_trend = slope * normalized_time

        local_seasonal = np.zeros(n, dtype=float)
        for period in rng.choice(
            np.asarray(dynamics.seasonal_periods),
            size=int(rng.integers(1, min(3, len(dynamics.seasonal_periods)) + 1)),
            replace=False,
        ):
            amplitude = rng.uniform(0.05, 0.40) * local_scale
            phase = rng.uniform(0, 2 * np.pi)
            local_seasonal += amplitude * np.sin(2 * np.pi * time / max(float(period), 1.0) + phase)

        local_regime = np.zeros(n, dtype=int)
        regime_effect = np.zeros(n, dtype=float)
        if n >= 8 and rng.random() < dynamics.regime_change_probability:
            change_point = int(rng.integers(max(2, n // 4), max(3, (3 * n) // 4)))
            local_regime[change_point:] = 1
            regime_effect[change_point:] = rng.normal(scale=0.8 * local_scale)

        local_drift = dynamics.concept_drift_strength * local_scale * normalized_time
        if rng.random() < 0.5:
            local_drift *= -1.0

        local_shock = np.zeros(n, dtype=float)
        shock_mask = rng.random(n) < dynamics.shock_probability
        local_shock[shock_mask] = rng.normal(scale=2.5 * local_scale, size=shock_mask.sum())

        ar_order = int(
            rng.integers(
                dynamics.min_autoregressive_order,
                dynamics.max_autoregressive_order + 1,
            )
        )
        ar_order = max(1, min(ar_order, max(1, n - 1)))
        coefficients = rng.uniform(0.08, 0.55, size=ar_order)
        coefficients = coefficients / max(coefficients.sum(), 1e-6) * rng.uniform(0.35, 0.85)

        local_observed = np.zeros(n, dtype=float)
        local_ar = np.zeros(n, dtype=float)
        innovations = rng.normal(scale=0.12 * local_scale + 0.03 * global_scale, size=n)
        for t in range(n):
            previous = 0.0
            for lag, coefficient in enumerate(coefficients, start=1):
                if t - lag >= 0:
                    previous += coefficient * local_observed[t - lag]
            local_ar[t] = previous
            local_observed[t] = (
                0.30 * base[t]
                + previous
                + local_trend[t]
                + local_seasonal[t]
                + local_drift[t]
                + regime_effect[t]
                + local_shock[t]
                + innovations[t]
            )

        observed[idx] = local_observed
        trend[idx] = local_trend
        seasonal[idx] = local_seasonal
        ar_component[idx] = local_ar
        shock[idx] = local_shock
        drift[idx] = local_drift
        regime[idx] = local_regime

    out["forecast_observed_value"] = observed
    out["forecast_trend_component"] = trend
    out["forecast_seasonal_component"] = seasonal
    out["forecast_autoregressive_component"] = ar_component
    out["forecast_shock_component"] = shock
    out["forecast_concept_drift_component"] = drift
    out["forecast_regime_id"] = regime
    out["forecast_domain"] = rng.choice(
        ["demand", "energy", "inventory", "sensor", "finance"],
        size=len(out),
    )
    out["shock_indicator"] = (np.abs(shock) > 1e-9).astype(int)
    return out, "forecast_observed_value"

