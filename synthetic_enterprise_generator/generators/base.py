"""Stage 1: base synthetic table generation.

The functions in this module create mixed-schema tabular data from the TabPFN
ecosystem adapter, then enrich it with categorical columns, noisy columns,
missing values, class imbalance, and nonlinear feature interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from synthetic_enterprise_generator.generators.tabpfn_adapter import (
    TabPFNEcosystemGenerator,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class BaseTableSpec:
    n_rows: int
    n_features: int
    n_classes: int = 3
    missing_rate: float = 0.02
    categorical_fraction: float = 0.25
    noisy_fraction: float = 0.15
    class_imbalance: float = 0.25


class BaseSyntheticTableGenerator:
    """Generate base tables for classification, regression, and mixed tasks."""

    def __init__(
        self,
        rng: np.random.Generator,
        require_tabpfn_ecosystem: bool = False,
        compute_device: str = "auto",
        tabpfn_max_rows: int = 10_000,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self.rng = rng
        self.progress_callback = progress_callback
        self.adapter = TabPFNEcosystemGenerator(
            require_ecosystem=require_tabpfn_ecosystem,
            compute_device=compute_device,
            max_ecosystem_rows=tabpfn_max_rows,
        )

    def generate_classification_table(
        self,
        spec: BaseTableSpec,
        count_progress: bool = True,
    ) -> pd.DataFrame:
        """Generate a classification table with mixed enterprise-like features."""

        result = self.adapter.generate(
            task="classification",
            n_rows=spec.n_rows,
            n_features=spec.n_features,
            n_classes=spec.n_classes,
            rng=self.rng,
            progress_callback=self.progress_callback if count_progress else None,
        )
        df = self._matrix_to_frame(result.X)
        df["classification_target"] = self._rebalance_labels(
            result.y.astype(int), spec.n_classes, spec.class_imbalance
        )
        df["base_generator_source"] = result.source
        return self._postprocess_base_table(df, spec)

    def generate_regression_table(
        self,
        spec: BaseTableSpec,
        count_progress: bool = True,
    ) -> pd.DataFrame:
        """Generate a regression table with mixed feature types."""

        result = self.adapter.generate(
            task="regression",
            n_rows=spec.n_rows,
            n_features=spec.n_features,
            n_classes=spec.n_classes,
            rng=self.rng,
            progress_callback=self.progress_callback if count_progress else None,
        )
        df = self._matrix_to_frame(result.X)
        df["regression_target"] = result.y.astype(float)
        df["base_generator_source"] = result.source
        return self._postprocess_base_table(df, spec)

    def generate_mixed_schema_table(self, spec: BaseTableSpec) -> pd.DataFrame:
        """Generate a table containing classification and regression targets."""

        classification = self.generate_classification_table(spec, count_progress=True)
        regression = self.generate_regression_table(spec, count_progress=False)
        numeric_columns = [c for c in regression.columns if c.startswith("num_")]
        for column in numeric_columns[: max(1, len(numeric_columns) // 3)]:
            classification[f"reg_{column}"] = regression[column].to_numpy()
        classification["regression_target"] = regression["regression_target"].to_numpy()
        return classification

    def _matrix_to_frame(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=[f"num_{i:03d}" for i in range(X.shape[1])])

    def _postprocess_base_table(self, df: pd.DataFrame, spec: BaseTableSpec) -> pd.DataFrame:
        df = self._add_causal_interactions(df)
        df = self._add_categorical_columns(df, spec)
        df = self._add_noisy_features(df, spec)
        df = self._inject_light_missingness(df, spec.missing_rate)
        return df

    def _add_causal_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple nonlinear interactions that downstream targets can exploit."""

        out = df.copy()
        numeric_columns = [c for c in out.columns if c.startswith("num_")]
        if len(numeric_columns) >= 2:
            out["interaction_product_0_1"] = out[numeric_columns[0]] * out[numeric_columns[1]]
            out["interaction_ratio_0_1"] = out[numeric_columns[0]] / (
                out[numeric_columns[1]].abs() + 1e-3
            )
        if numeric_columns:
            out["periodic_signal_0"] = np.sin(out[numeric_columns[0]])
        return out

    def _add_categorical_columns(self, df: pd.DataFrame, spec: BaseTableSpec) -> pd.DataFrame:
        out = df.copy()
        n_categoricals = max(1, int(spec.n_features * spec.categorical_fraction))
        numeric_columns = [c for c in out.columns if c.startswith("num_")]
        for i in range(n_categoricals):
            source = numeric_columns[i % len(numeric_columns)]
            n_levels = int(self.rng.integers(3, 12))
            quantiles = np.linspace(0, 1, n_levels + 1)
            bins = np.unique(np.quantile(out[source], quantiles))
            if len(bins) <= 2:
                labels = self.rng.integers(0, n_levels, size=len(out))
            else:
                labels = np.digitize(out[source], bins[1:-1])
            out[f"cat_{i:03d}"] = (
                pd.Series(labels, index=out.index).astype(str).radd(f"C{i}_")
            )
        return out

    def _add_noisy_features(self, df: pd.DataFrame, spec: BaseTableSpec) -> pd.DataFrame:
        out = df.copy()
        n_noise = max(1, int(spec.n_features * spec.noisy_fraction))
        for i in range(n_noise):
            distribution = self.rng.choice(["normal", "lognormal", "poisson", "binary", "gamma"])
            if distribution == "normal":
                values = self.rng.normal(size=len(out))
            elif distribution == "lognormal":
                values = self.rng.lognormal(mean=0.0, sigma=0.8, size=len(out))
            elif distribution == "poisson":
                values = self.rng.poisson(lam=float(self.rng.uniform(1, 8)), size=len(out))
            elif distribution == "gamma":
                values = stats.gamma(a=float(self.rng.uniform(1.0, 4.0))).rvs(
                    size=len(out),
                    random_state=self.rng,
                )
            else:
                values = self.rng.binomial(1, p=float(self.rng.uniform(0.05, 0.5)), size=len(out))
            out[f"noise_{i:03d}"] = values
        return out

    def _inject_light_missingness(self, df: pd.DataFrame, rate: float) -> pd.DataFrame:
        out = df.copy()
        feature_columns = [
            c
            for c in out.columns
            if not c.endswith("_target") and c != "base_generator_source"
        ]
        for column in feature_columns:
            mask = self.rng.random(len(out)) < rate
            out.loc[mask, column] = np.nan
        return out

    def _rebalance_labels(
        self, y: np.ndarray, n_classes: int, imbalance: float
    ) -> np.ndarray:
        """Push labels toward a configurable long-tail distribution."""

        if imbalance <= 0:
            return y
        probabilities = np.linspace(1.0, max(0.05, imbalance), n_classes)
        probabilities = probabilities / probabilities.sum()
        replacement = self.rng.choice(np.arange(n_classes), size=len(y), p=probabilities)
        keep_original = self.rng.random(len(y)) > imbalance
        return np.where(keep_original, y % n_classes, replacement)


def _default_generator(seed: Optional[int], require_tabpfn_ecosystem: bool) -> BaseSyntheticTableGenerator:
    rng = np.random.default_rng(seed)
    return BaseSyntheticTableGenerator(rng=rng, require_tabpfn_ecosystem=require_tabpfn_ecosystem)


def generate_classification_table(
    n_rows: int = 1_000,
    n_features: int = 16,
    n_classes: int = 3,
    seed: Optional[int] = None,
    require_tabpfn_ecosystem: bool = False,
) -> pd.DataFrame:
    """Convenience function for Stage 1 classification generation."""

    generator = _default_generator(seed, require_tabpfn_ecosystem)
    return generator.generate_classification_table(
        BaseTableSpec(n_rows=n_rows, n_features=n_features, n_classes=n_classes)
    )


def generate_regression_table(
    n_rows: int = 1_000,
    n_features: int = 16,
    seed: Optional[int] = None,
    require_tabpfn_ecosystem: bool = False,
) -> pd.DataFrame:
    """Convenience function for Stage 1 regression generation."""

    generator = _default_generator(seed, require_tabpfn_ecosystem)
    return generator.generate_regression_table(
        BaseTableSpec(n_rows=n_rows, n_features=n_features)
    )


def generate_mixed_schema_table(
    n_rows: int = 1_000,
    n_features: int = 16,
    n_classes: int = 3,
    seed: Optional[int] = None,
    require_tabpfn_ecosystem: bool = False,
) -> pd.DataFrame:
    """Convenience function for Stage 1 mixed-schema generation."""

    generator = _default_generator(seed, require_tabpfn_ecosystem)
    return generator.generate_mixed_schema_table(
        BaseTableSpec(n_rows=n_rows, n_features=n_features, n_classes=n_classes)
    )
