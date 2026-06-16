"""Stage 3: temporal feature injection and distribution dynamics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from synthetic_enterprise_generator.config import TemporalConfig


FREQUENCY_MAP = {
    "hourly": "h",
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
}


def add_temporal_features(
    df: pd.DataFrame,
    config: TemporalConfig,
    rng: np.random.Generator,
    entity_column: str = "session_id",
) -> pd.DataFrame:
    """Add timestamps and derived calendar features.

    When sessions exist, timestamps are monotonic within each session so sequence
    and forecasting tasks can use coherent histories.
    """

    out = df.copy()
    freq = FREQUENCY_MAP.get(config.granularity, "D")
    base_range = pd.date_range(
        start=config.start_date,
        periods=max(config.periods, 2),
        freq=freq,
    )
    if entity_column in out.columns:
        timestamp_values = []
        for _, group in out.groupby(entity_column, sort=False):
            start_idx = int(rng.integers(0, len(base_range)))
            offsets = np.arange(len(group))
            indices = np.minimum(start_idx + offsets, len(base_range) - 1)
            jitter_hours = rng.integers(0, 24, size=len(group))
            timestamp_values.extend(base_range[indices] + pd.to_timedelta(jitter_hours, unit="h"))
        out["timestamp"] = timestamp_values
    else:
        out["timestamp"] = rng.choice(base_range, size=len(out))

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["week_of_year"] = out["timestamp"].dt.isocalendar().week.astype(int)
    out["month"] = out["timestamp"].dt.month
    out["quarter"] = out["timestamp"].dt.quarter
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def inject_seasonality(
    df: pd.DataFrame,
    config: TemporalConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject periodic patterns into numeric business measures."""

    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    numeric_columns = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if not c.endswith("_target")
    ]
    if not numeric_columns:
        return out
    day_index = (out["timestamp"] - out["timestamp"].min()).dt.total_seconds() / 86_400
    weekly = np.sin(2 * np.pi * day_index / 7.0)
    yearly = np.sin(2 * np.pi * day_index / 365.25)
    for column in numeric_columns[: max(1, len(numeric_columns) // 3)]:
        scale = out[column].std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[column] = out[column] + config.seasonality_strength * scale * (
            0.65 * weekly + 0.35 * yearly
        )
    out["seasonality_index"] = (0.65 * weekly + 0.35 * yearly).astype(float)
    return out


def inject_distribution_shift(
    df: pd.DataFrame,
    config: TemporalConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create temporal drift and occasional regime shifts."""

    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    numeric_columns = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if not c.endswith("_target")
    ]
    if not numeric_columns:
        return out
    order = out["timestamp"].rank(method="first").to_numpy()
    normalized_time = (order - order.min()) / max(order.max() - order.min(), 1)
    shift_point = rng.uniform(0.35, 0.75)
    out["regime_id"] = (normalized_time >= shift_point).astype(int)
    for column in numeric_columns[: max(1, len(numeric_columns) // 4)]:
        scale = out[column].std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        drift = config.drift_strength * scale * normalized_time
        shock = (
            rng.normal(loc=0.4 * scale, scale=0.1 * scale)
            * (normalized_time >= shift_point)
            * (rng.random() < config.shift_probability)
        )
        out[column] = out[column] + drift + shock
    return out


class TemporalAugmentor:
    """Object-oriented facade for temporal feature and drift augmentation."""

    def __init__(self, config: TemporalConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def add_temporal_features(
        self,
        df: pd.DataFrame,
        entity_column: str = "session_id",
    ) -> pd.DataFrame:
        """Add timestamps and derived calendar columns."""

        return add_temporal_features(df, self.config, self.rng, entity_column)

    def inject_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal signal to numeric columns."""

        return inject_seasonality(df, self.config, self.rng)

    def inject_distribution_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal drift and occasional regime shifts."""

        return inject_distribution_shift(df, self.config, self.rng)

    def augment(
        self,
        df: pd.DataFrame,
        entity_column: str = "session_id",
    ) -> pd.DataFrame:
        """Apply the standard temporal stage order."""

        out = self.add_temporal_features(df, entity_column)
        out = self.inject_seasonality(out)
        return self.inject_distribution_shift(out)
