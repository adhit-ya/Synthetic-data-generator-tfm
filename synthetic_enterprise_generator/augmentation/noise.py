"""Stage 5: realistic enterprise missingness, corruption, and noise."""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

from synthetic_enterprise_generator.config import MissingnessConfig
from synthetic_enterprise_generator.augmentation.temporal import TemporalAugmentor


MissingnessMode = Literal["MCAR", "MAR", "MNAR"]


def inject_missingness(
    df: pd.DataFrame,
    config: MissingnessConfig,
    rng: np.random.Generator,
    mode: MissingnessMode | str = "MCAR",
    protected_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Inject MCAR, MAR, and MNAR null patterns.

    Protected columns prevent primary keys, timestamps, and labels from being
    destroyed unless a caller explicitly allows it.
    """

    out = df.copy()
    protected = set(protected_columns or [])
    protected.update(c for c in out.columns if c.endswith("_target"))
    candidate_columns = [c for c in out.columns if c not in protected]
    boolean_columns = out[candidate_columns].select_dtypes(include=["bool"]).columns.tolist()
    if boolean_columns:
        out[boolean_columns] = out[boolean_columns].astype(object)
    numeric_columns = out[candidate_columns].select_dtypes(include=[np.number]).columns.tolist()
    if not candidate_columns:
        return out

    mode_upper = mode.upper()
    if mode_upper == "MCAR":
        rate = config.mcar_rate
        for column in candidate_columns:
            out.loc[rng.random(len(out)) < rate, column] = np.nan
    elif mode_upper == "MAR":
        rate = config.mar_rate
        driver = numeric_columns[0] if numeric_columns else candidate_columns[0]
        if pd.api.types.is_numeric_dtype(out[driver]):
            driver_score = out[driver].rank(pct=True).fillna(0.5).to_numpy()
        else:
            driver_score = out[driver].astype(str).map(lambda x: hash(x) % 1000 / 1000).to_numpy()
        for column in candidate_columns:
            probability = rate * (0.5 + driver_score)
            out.loc[rng.random(len(out)) < probability, column] = np.nan
    elif mode_upper == "MNAR":
        rate = config.mnar_rate
        for column in numeric_columns:
            score = out[column].rank(pct=True).fillna(0.5).to_numpy()
            probability = rate * (0.25 + 1.5 * score)
            out.loc[rng.random(len(out)) < probability, column] = np.nan
    else:
        raise ValueError(f"Unsupported missingness mode: {mode}")

    # Sparse telemetry columns mimic enterprise systems with rarely populated fields.
    for column in candidate_columns:
        if rng.random() < config.sparse_column_probability:
            out.loc[rng.random(len(out)) < 0.65, column] = np.nan
    return out


def inject_outliers(
    df: pd.DataFrame,
    config: MissingnessConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject row-level outliers into numeric columns."""

    out = df.copy()
    numeric_columns = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if not c.endswith("_target")
    ]
    if not numeric_columns:
        return out
    out[numeric_columns] = out[numeric_columns].astype(float)
    row_mask = rng.random(len(out)) < config.outlier_rate
    for column in numeric_columns:
        scale = out[column].std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        signs = rng.choice([-1, 1], size=row_mask.sum())
        out.loc[row_mask, column] = out.loc[row_mask, column] + signs * scale * rng.uniform(4, 10)
    outlier_indicator = pd.DataFrame(
        {"is_injected_outlier": row_mask.astype(int)},
        index=out.index,
    )
    return pd.concat([out, outlier_indicator], axis=1).copy()


def inject_noise(
    df: pd.DataFrame,
    config: MissingnessConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject numeric jitter and categorical field corruption."""

    out = df.copy()
    numeric_columns = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if not c.endswith("_target")
    ]
    protected_categoricals = {
        "customer_id",
        "session_id",
        "transaction_id",
        "product_id",
        "machine_id",
        "patient_id",
        "sequence_id",
        "event_type",
        "workflow_domain",
        "timestamp",
        "base_generator_source",
    }
    categorical_columns = [
        c
        for c in out.select_dtypes(include=["object", "category"]).columns
        if c not in protected_categoricals and not c.endswith("_target")
    ]
    if numeric_columns:
        out[numeric_columns] = out[numeric_columns].astype(float)
    for column in numeric_columns:
        mask = rng.random(len(out)) < config.noise_rate
        scale = out[column].std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out.loc[mask, column] = out.loc[mask, column] + rng.normal(
            scale=0.2 * scale, size=mask.sum()
        )
    for column in categorical_columns:
        mask = rng.random(len(out)) < config.noise_rate
        corrupt_values = rng.choice(["UNKNOWN", "CORRUPT", "REDACTED", ""], size=mask.sum())
        out.loc[mask, column] = corrupt_values
    return out


class NoiseEngine:
    """Object-oriented facade for missingness, outlier, and corruption stages."""

    def __init__(self, config: MissingnessConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def inject_missingness(
        self,
        df: pd.DataFrame,
        mode: MissingnessMode | str = "MCAR",
        protected_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Inject one missingness mode."""

        return inject_missingness(df, self.config, self.rng, mode, protected_columns)

    def inject_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject row-level numeric outliers."""

        return inject_outliers(df, self.config, self.rng)

    def inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject numeric jitter and categorical corruption."""

        return inject_noise(df, self.config, self.rng)

    def augment(
        self,
        df: pd.DataFrame,
        protected_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Apply MCAR, MAR, MNAR, outlier, and noise stages."""

        out = self.inject_missingness(df, mode="MCAR", protected_columns=protected_columns)
        out = self.inject_missingness(out, mode="MAR", protected_columns=protected_columns)
        out = self.inject_missingness(out, mode="MNAR", protected_columns=protected_columns)
        out = self.inject_outliers(out)
        return self.inject_noise(out)
