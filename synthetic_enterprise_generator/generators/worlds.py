"""Stage 7 and Stage X: diverse synthetic world orchestration.

This module implements a mixture-of-worlds prior. Enterprise workflow worlds are
one important family, but the pretraining corpus also samples scientific, IID,
temporal, relational, and sparse statistical environments so a TabFM does not
overfit to a single benchmark style.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from synthetic_enterprise_generator.augmentation.entities import (
    add_entity_columns,
    create_session_structure,
)
from synthetic_enterprise_generator.augmentation.forecasting import (
    ForecastingDynamicsSpec,
    add_forecasting_dynamics,
)
from synthetic_enterprise_generator.augmentation.noise import (
    inject_missingness,
    inject_noise,
    inject_outliers,
)
from synthetic_enterprise_generator.augmentation.temporal import (
    add_temporal_features,
    inject_distribution_shift,
    inject_seasonality,
)
from synthetic_enterprise_generator.config import WorldConfig
from synthetic_enterprise_generator.generators.base import (
    BaseSyntheticTableGenerator,
    BaseTableSpec,
)
from synthetic_enterprise_generator.generators.targets import (
    create_anomaly_targets,
    create_classification_targets,
    create_count_targets,
    create_forecasting_targets,
    create_imputation_targets,
    create_next_event_targets,
    create_ordinal_targets,
    create_regression_targets,
)
from synthetic_enterprise_generator.relational.tables import generate_relational_tables
from synthetic_enterprise_generator.workflows.events import simulate_enterprise_workflows

LOGGER = logging.getLogger(__name__)
WORLD_TYPES = (
    "enterprise",
    "scientific",
    "iid",
    "temporal",
    "relational",
    "sparse",
    "advanced_forecasting",
)


def randomize_schema(config: WorldConfig, rng: np.random.Generator) -> WorldConfig:
    """Create a per-world config variant for diversity scaling."""

    variant = copy.deepcopy(config)
    if config.randomize_row_counts:
        variant.rows_per_world = int(
            max(
                100,
                rng.normal(
                    loc=config.rows_per_world,
                    scale=config.rows_per_world * 0.15,
                ),
            )
        )
    else:
        variant.rows_per_world = int(config.rows_per_world)
    variant.min_features = max(4, config.min_features)
    variant.max_features = max(variant.min_features, config.max_features)
    variant.classification_classes = int(rng.integers(2, max(3, config.classification_classes + 2)))
    variant.class_imbalance = float(rng.uniform(0.05, 0.45))
    variant.workflow.domain = str(rng.choice(["retail", "industrial", "healthcare"]))
    variant.workflow.max_sequence_length = int(rng.integers(5, 12))
    variant.workflow.branch_probability = float(rng.uniform(0.15, 0.55))
    variant.temporal.granularity = str(rng.choice(["hourly", "daily", "weekly", "monthly"]))
    variant.temporal.seasonality_strength = float(rng.uniform(0.05, 0.40))
    variant.missingness.mcar_rate = float(rng.uniform(0.01, 0.08))
    variant.missingness.mar_rate = float(rng.uniform(0.01, 0.10))
    variant.missingness.mnar_rate = float(rng.uniform(0.005, 0.05))
    return variant


def sample_world_type(config: WorldConfig, rng: np.random.Generator) -> str:
    """Sample a world category from configured mixture weights.

    The default weights are balanced across all supported world types, which
    gives the final corpus a broad prior over benchmark-like environments.
    """

    weights = np.array([float(config.world_type_weights.get(kind, 0.0)) for kind in WORLD_TYPES])
    if weights.sum() <= 0:
        weights = np.ones(len(WORLD_TYPES))
    probabilities = weights / weights.sum()
    return str(rng.choice(WORLD_TYPES, p=probabilities))


def _random_row_count(config: WorldConfig, rng: np.random.Generator, world_type: str) -> int:
    """Draw small, medium, large, and low-sample regimes around the base size."""

    if not config.randomize_row_counts:
        return int(config.rows_per_world)
    if world_type == "scientific":
        factor = rng.choice([0.15, 0.3, 0.6, 1.0], p=[0.25, 0.35, 0.25, 0.15])
    elif world_type == "sparse":
        factor = rng.choice([0.5, 1.0, 2.0, 4.0], p=[0.20, 0.35, 0.30, 0.15])
    else:
        factor = rng.choice([0.4, 0.8, 1.0, 1.8, 3.0], p=[0.15, 0.25, 0.25, 0.25, 0.10])
    return int(max(40, round(config.rows_per_world * float(factor))))


def _random_feature_count(config: WorldConfig, rng: np.random.Generator, world_type: str) -> int:
    if world_type == "scientific":
        high = max(config.max_features * 8, config.min_features * 16, 192)
        low = max(config.min_features, 20)
    elif world_type == "sparse":
        high = max(config.max_features * 16, config.min_features * 32, 512)
        low = max(config.min_features * 4, 128)
    elif world_type == "iid":
        low, high = config.min_features, max(config.max_features, 12)
    else:
        low, high = config.min_features, max(config.max_features, config.min_features + 4)
    return int(rng.integers(max(4, low), high + 1))


def _feature_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if not column.endswith("_target")
        and column
        not in {
            "hour",
            "day_of_week",
            "week_of_year",
            "month",
            "quarter",
            "is_weekend",
            "regime_id",
            "forecast_regime_id",
            "shock_indicator",
        }
    ]


def _add_world_columns(df: pd.DataFrame, world_id: str, world_type: str) -> pd.DataFrame:
    out = df.copy()
    out["world_id"] = world_id
    out["world_type"] = world_type
    return out


def _protected_columns(df: pd.DataFrame) -> set[str]:
    protected = {
        "customer_id",
        "session_id",
        "transaction_id",
        "product_id",
        "machine_id",
        "patient_id",
        "entity_id",
        "user_id",
        "order_id",
        "visit_id",
        "experiment_id",
        "sample_id",
        "timestamp",
        "sequence_id",
        "event_type",
        "world_id",
        "world_type",
    }
    protected.update(c for c in df.columns if c.endswith("_target"))
    return protected


def _apply_multitask_targets(
    df: pd.DataFrame,
    config: WorldConfig,
    rng: np.random.Generator,
    include_forecast: bool = False,
    include_next_event: bool = False,
) -> pd.DataFrame:
    out = create_classification_targets(df, config.classification_classes, rng)
    out = create_regression_targets(out, rng)
    out = create_ordinal_targets(out, rng, n_levels=max(3, min(7, config.classification_classes + 2)))
    out = create_count_targets(out, rng)
    if include_forecast or "timestamp" in out.columns:
        group_column = "customer_id" if "customer_id" in out.columns else None
        out = create_forecasting_targets(out, group_column=group_column)
    if include_next_event and "event_type" in out.columns:
        out = create_next_event_targets(out)
    out = create_imputation_targets(out, rng)
    out = create_anomaly_targets(out, rng)
    return out


def _apply_noise_and_missingness(
    df: pd.DataFrame,
    config: WorldConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = inject_missingness(df, config.missingness, rng, mode="MCAR", protected_columns=_protected_columns(df))
    out = inject_missingness(out, config.missingness, rng, mode="MAR", protected_columns=_protected_columns(out))
    out = inject_missingness(out, config.missingness, rng, mode="MNAR", protected_columns=_protected_columns(out))
    out = inject_outliers(out, config.missingness, rng)
    out = inject_noise(out, config.missingness, rng)
    return out


def _report_progress(
    progress_callback: Optional[Callable[[str, int], None]],
    stage: str,
    rows: int = 0,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, rows)


def _base_generator(
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> BaseSyntheticTableGenerator:
    return BaseSyntheticTableGenerator(
        rng=rng,
        require_tabpfn_ecosystem=config.require_tabpfn_ecosystem,
        compute_device=config.compute_device,
        tabpfn_max_rows=config.tabpfn_max_rows,
        progress_callback=progress_callback,
    )


def _base_spec(
    config: WorldConfig,
    rng: np.random.Generator,
    world_type: str,
    rows: Optional[int] = None,
    features: Optional[int] = None,
) -> BaseTableSpec:
    selected_rows = int(rows if rows is not None else _random_row_count(config, rng, world_type))
    requested_features = int(
        features if features is not None else _random_feature_count(config, rng, world_type)
    )
    feature_budget = max(4, int(config.max_table_cells) // max(1, selected_rows))
    selected_features = min(requested_features, feature_budget)
    if selected_features < requested_features:
        LOGGER.warning(
            "Capping %s world base features from %s to %s for %s rows "
            "(max_table_cells=%s).",
            world_type,
            requested_features,
            selected_features,
            f"{selected_rows:,}",
            f"{config.max_table_cells:,}",
        )
    return BaseTableSpec(
        n_rows=selected_rows,
        n_features=selected_features,
        n_classes=int(max(2, config.classification_classes)),
        missing_rate=float(rng.uniform(0.0, 0.03)),
        class_imbalance=float(rng.uniform(0.05, 0.55)),
        categorical_fraction=float(rng.uniform(0.08, 0.45)),
        noisy_fraction=float(rng.uniform(0.05, 0.30)),
    )


def _generate_enterprise_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], object]:
    """Generate one complete enterprise workflow world."""

    variant = randomize_schema(config, rng)
    spec = _base_spec(variant, rng, "enterprise", rows=variant.rows_per_world)
    _report_progress(progress_callback, f"{world_label}: base rows")
    fact = _base_generator(variant, rng, progress_callback).generate_mixed_schema_table(spec)
    _report_progress(progress_callback, f"{world_label}: entities and sessions")
    fact = add_entity_columns(fact, variant.entity, rng)
    fact = create_session_structure(fact, variant.entity, rng)
    _report_progress(progress_callback, f"{world_label}: temporal features")
    fact = add_temporal_features(fact, variant.temporal, rng)
    fact = inject_seasonality(fact, variant.temporal, rng)
    fact = inject_distribution_shift(fact, variant.temporal, rng)
    _report_progress(progress_callback, f"{world_label}: workflow events")
    fact, workflow_graph = simulate_enterprise_workflows(fact, variant.workflow, rng)
    _report_progress(progress_callback, f"{world_label}: multitask targets")
    fact = _apply_multitask_targets(fact, variant, rng, include_forecast=True, include_next_event=True)
    _report_progress(progress_callback, f"{world_label}: noise and missingness")
    fact = _apply_noise_and_missingness(fact, variant, rng)
    fact = _add_world_columns(fact, world_label, "enterprise")
    _report_progress(progress_callback, f"{world_label}: relational tables")
    tables = generate_relational_tables(fact, variant.entity, rng)
    _report_progress(progress_callback, f"{world_label}: complete")
    return tables, workflow_graph


def _generate_scientific_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate genomics/biomedical/chemical/physics-style tabular data."""

    variant = randomize_schema(config, rng)
    n_rows = _random_row_count(variant, rng, "scientific")
    n_features = _random_feature_count(variant, rng, "scientific")

    spec = _base_spec(variant, rng, "scientific", rows=n_rows, features=n_features)
    n_features = spec.n_features
    spec.categorical_fraction = float(rng.uniform(0.01, 0.10))
    spec.noisy_fraction = float(rng.uniform(0.08, 0.25))
    spec.missing_rate = float(rng.uniform(0.0, 0.04))
    _report_progress(progress_callback, f"{world_label}: base rows")
    df = _base_generator(variant, rng, progress_callback).generate_regression_table(spec)

    numeric_columns = _feature_numeric_columns(df)
    numeric = df[numeric_columns].fillna(df[numeric_columns].median()).fillna(0.0)
    matrix = numeric.to_numpy(dtype=float)
    centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    scale = np.nanstd(centered, axis=0, keepdims=True)
    normalized = centered / np.where(scale < 1e-8, 1.0, scale)

    latent_dim = int(rng.integers(3, min(14, max(3, len(numeric_columns))) + 1))
    projection = rng.normal(size=(len(numeric_columns), latent_dim))
    latent = normalized @ projection / np.sqrt(max(1, len(numeric_columns)))
    loading = rng.normal(size=(latent_dim, max(8, n_features // 2)))
    assay_matrix = latent @ loading
    assay_matrix += rng.normal(scale=rng.uniform(0.15, 0.75), size=assay_matrix.shape)

    # Lightweight augmentation: correlated assay panels, batch effects, sparse
    # thresholds, and redundant measurements layered over TabPFN-style priors.
    batch_count = max(3, n_rows // 40)
    batch_ids = rng.integers(0, batch_count, size=n_rows)
    batch_effects = rng.normal(scale=rng.uniform(0.10, 0.60), size=(batch_count, assay_matrix.shape[1]))
    assay_matrix += batch_effects[batch_ids]
    threshold_mask = rng.random(assay_matrix.shape) < rng.uniform(0.05, 0.35)
    assay_matrix[threshold_mask] = 0.0

    assay_columns = [f"sci_assay_{i:04d}" for i in range(assay_matrix.shape[1])]
    assay_df = pd.DataFrame(assay_matrix, columns=assay_columns, index=df.index)
    df = pd.concat([df, assay_df], axis=1)
    redundant_columns = {}
    for i in range(max(2, n_features // 10)):
        source = str(rng.choice(assay_columns + numeric_columns))
        source_scale = float(pd.Series(df[source]).std(skipna=True))
        if not np.isfinite(source_scale) or source_scale <= 0:
            source_scale = 1.0
        redundant_columns[f"redundant_measure_{i:03d}"] = (
            df[source].to_numpy(dtype=float) + rng.normal(scale=0.04 * source_scale, size=n_rows)
        )

    scientific_metadata = pd.DataFrame(
        {
            **redundant_columns,
            "experiment_id": rng.choice([f"EXP_{i:04d}" for i in range(max(4, n_rows // 25))], size=n_rows),
            "sample_id": [f"SAMPLE_{i:08d}" for i in range(n_rows)],
            "assay_type": rng.choice(
                ["genomics", "bioinformatics", "biomedical", "chemistry", "materials"],
                size=n_rows,
            ),
            "lab_batch": pd.Series(batch_ids).map(lambda value: f"BATCH_{int(value):03d}").to_numpy(),
            "measurement_quality_flag": rng.random(n_rows) < rng.uniform(0.05, 0.20),
        },
        index=df.index,
    )
    df = pd.concat([df, scientific_metadata], axis=1).copy()

    nonlinear = np.sin(latent[:, 0])
    if latent.shape[1] > 1:
        nonlinear += 0.35 * latent[:, 1] ** 2
    if latent.shape[1] > 2:
        nonlinear += 0.20 * latent[:, 0] * latent[:, 2]
    df["scientific_property_target"] = nonlinear + rng.normal(scale=0.2, size=n_rows)
    df = _apply_multitask_targets(df, variant, rng)
    df = _apply_noise_and_missingness(df, variant, rng)
    df = _add_world_columns(df, world_label, "scientific")
    return {"scientific_measurements": df}


def _generate_iid_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate OpenML/UCI/Kaggle-style IID benchmark tables."""

    variant = randomize_schema(config, rng)
    spec = _base_spec(variant, rng, "iid")
    task = str(rng.choice(["classification", "regression", "mixed"], p=[0.42, 0.28, 0.30]))
    _report_progress(progress_callback, f"{world_label}: base rows")
    generator = _base_generator(variant, rng, progress_callback)
    if task == "classification":
        df = generator.generate_classification_table(spec)
    elif task == "regression":
        df = generator.generate_regression_table(spec)
    else:
        df = generator.generate_mixed_schema_table(spec)
    df["benchmark_style"] = rng.choice(["openml", "uci", "kaggle", "tabarena"], size=len(df))
    df["fold_id"] = rng.integers(0, 5, size=len(df))
    df = _apply_multitask_targets(df, variant, rng)
    df = _apply_noise_and_missingness(df, variant, rng)
    df = _add_world_columns(df, world_label, "iid")
    return {"iid_table": df}


def _generate_temporal_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate forecasting and temporal-tabular benchmark worlds."""

    variant = randomize_schema(config, rng)
    variant.temporal.granularity = str(rng.choice(["hourly", "daily", "weekly", "monthly"]))
    spec = _base_spec(variant, rng, "temporal")
    _report_progress(progress_callback, f"{world_label}: base rows")
    df = _base_generator(variant, rng, progress_callback).generate_regression_table(spec)
    n_entities = max(5, int(np.sqrt(len(df))))
    df["series_id"] = rng.choice([f"SERIES_{i:05d}" for i in range(n_entities)], size=len(df))
    df = add_temporal_features(df, variant.temporal, rng, entity_column="series_id")
    df = inject_seasonality(df, variant.temporal, rng)
    df = inject_distribution_shift(df, variant.temporal, rng)
    value_col = next((c for c in df.columns if c.startswith("num_")), None)
    if value_col is not None:
        grouped = df.sort_values(["series_id", "timestamp"]).groupby("series_id")[value_col]
        df["rolling_mean_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
        df["rolling_std_7"] = grouped.transform(lambda s: s.rolling(7, min_periods=2).std()).fillna(0.0)
        df["lag_1"] = grouped.shift(1)
    df["temporal_domain"] = rng.choice(["demand", "energy", "market", "sensor"], size=len(df))
    df = _apply_multitask_targets(df, variant, rng, include_forecast=True)
    df = create_forecasting_targets(df, value_column=value_col, group_column="series_id", horizon=int(rng.integers(1, 4)))
    df = _apply_noise_and_missingness(df, variant, rng)
    df = _add_world_columns(df, world_label, "temporal")
    return {"temporal_observations": df}


def _generate_advanced_forecasting_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate temporal worlds with explicit forecasting dynamics.

    This reuses the standard temporal stage for timestamps, calendar features,
    seasonality, and distribution shift. The forecasting layer then replaces the
    simple next-value task with a stochastic process containing trend,
    autoregression, shocks, concept drift, and regime changes.
    """

    variant = randomize_schema(config, rng)
    variant.temporal.granularity = str(rng.choice(["hourly", "daily", "weekly", "monthly"]))
    rows = _random_row_count(variant, rng, "temporal")
    features = _random_feature_count(variant, rng, "temporal")
    spec = _base_spec(variant, rng, "temporal", rows=rows, features=features)
    _report_progress(progress_callback, f"{world_label}: base rows")
    df = _base_generator(variant, rng, progress_callback).generate_regression_table(spec)

    n_entities = max(4, min(len(df), int(rng.integers(6, max(7, int(np.sqrt(len(df))) + 8)))))
    df["series_id"] = rng.choice([f"SERIES_{i:05d}" for i in range(n_entities)], size=len(df))
    df = add_temporal_features(df, variant.temporal, rng, entity_column="series_id")
    df = inject_seasonality(df, variant.temporal, rng)
    df = inject_distribution_shift(df, variant.temporal, rng)

    value_col = next((column for column in df.columns if column.startswith("num_")), None)
    dynamics_spec = ForecastingDynamicsSpec(
        min_series=max(4, n_entities // 2),
        max_series=max(6, n_entities * 2),
        shock_probability=float(rng.uniform(0.015, 0.075)),
        regime_change_probability=float(rng.uniform(0.35, 0.80)),
        concept_drift_strength=float(rng.uniform(0.15, 0.70)),
    )
    df, observed_col = add_forecasting_dynamics(
        df,
        rng,
        group_column="series_id",
        value_column=value_col,
        spec=dynamics_spec,
    )

    grouped = df.sort_values(["series_id", "timestamp"]).groupby("series_id")
    if observed_col is not None:
        observed_grouped = grouped[observed_col]
        for lag in (1, 2, 3, 7):
            df[f"forecast_lag_{lag}"] = observed_grouped.shift(lag)
        for window in (3, 7, 14):
            df[f"forecast_rolling_mean_{window}"] = observed_grouped.transform(
                lambda s, window=window: s.rolling(window, min_periods=1).mean()
            )
            df[f"forecast_rolling_std_{window}"] = observed_grouped.transform(
                lambda s, window=window: s.rolling(window, min_periods=2).std()
            ).fillna(0.0)

    df["forecast_horizon"] = rng.choice([1, 2, 3, 7, 14], size=len(df), p=[0.30, 0.20, 0.20, 0.20, 0.10])
    df = _apply_multitask_targets(df, variant, rng, include_forecast=True)
    if observed_col is not None:
        for horizon in (1, 3, 7):
            df = create_forecasting_targets(
                df,
                value_column=observed_col,
                group_column="series_id",
                horizon=horizon,
                target_name=f"forecast_horizon_{horizon}_target",
            )
        df = create_forecasting_targets(
            df,
            value_column=observed_col,
            group_column="series_id",
            horizon=1,
            target_name="forecast_target",
        )
    df = _apply_noise_and_missingness(df, variant, rng)
    df = _add_world_columns(df, world_label, "advanced_forecasting")
    return {"forecasting_observations": df}


def _generate_relational_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate connected multi-table ecosystems with graph-like dependencies."""

    variant = randomize_schema(config, rng)
    spec = _base_spec(variant, rng, "relational")
    _report_progress(progress_callback, f"{world_label}: base rows")
    fact = _base_generator(variant, rng, progress_callback).generate_mixed_schema_table(spec)
    fact = add_entity_columns(fact, variant.entity, rng)
    fact = create_session_structure(fact, variant.entity, rng)
    fact = add_temporal_features(fact, variant.temporal, rng)
    fact["order_id"] = [f"ORDER_{i:09d}" for i in range(len(fact))]
    fact["visit_id"] = rng.choice([f"VISIT_{i:08d}" for i in range(max(10, len(fact) // 4))], size=len(fact))
    fact = _apply_multitask_targets(fact, variant, rng, include_forecast=True)
    fact = _apply_noise_and_missingness(fact, variant, rng)
    fact = _add_world_columns(fact, world_label, "relational")
    tables = generate_relational_tables(fact, variant.entity, rng)

    if not tables["customers"].empty:
        customer_edges = []
        customers = tables["customers"]["customer_id"].tolist()
        for _ in range(max(1, len(customers) // 2)):
            src, dst = rng.choice(customers, size=2, replace=False)
            customer_edges.append(
                {
                    "source_customer_id": src,
                    "target_customer_id": dst,
                    "relationship_type": rng.choice(["same_account", "referral", "household", "supplier"]),
                    "edge_weight": float(rng.uniform(0.1, 1.0)),
                    "world_id": world_label,
                    "world_type": "relational",
                }
            )
        tables["entity_graph_edges"] = pd.DataFrame(customer_edges)
    return tables


def _generate_sparse_world(
    world_label: str,
    config: WorldConfig,
    rng: np.random.Generator,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate sparse high-dimensional ad-click/genomics/recommender worlds."""

    variant = randomize_schema(config, rng)
    n_rows = _random_row_count(variant, rng, "sparse")
    n_features = _random_feature_count(variant, rng, "sparse")

    spec = _base_spec(variant, rng, "sparse", rows=n_rows, features=n_features)
    n_features = spec.n_features
    spec.categorical_fraction = float(rng.uniform(0.005, 0.05))
    spec.noisy_fraction = float(rng.uniform(0.10, 0.35))
    spec.missing_rate = float(rng.uniform(0.0, 0.02))
    _report_progress(progress_callback, f"{world_label}: base rows")
    df = _base_generator(variant, rng, progress_callback).generate_mixed_schema_table(spec)

    numeric_columns = _feature_numeric_columns(df)
    numeric = df[numeric_columns].fillna(df[numeric_columns].median()).fillna(0.0)
    matrix = numeric.to_numpy(dtype=float)
    if matrix.size == 0:
        matrix = rng.normal(size=(n_rows, n_features))
        numeric_columns = [f"sparse_prior_num_{i:05d}" for i in range(n_features)]

    n_signal = max(2, min(len(numeric_columns), n_features // 40))
    signal_positions = rng.choice(np.arange(len(numeric_columns)), size=n_signal, replace=False)
    signal_score = matrix[:, signal_positions].sum(axis=1)
    if n_signal >= 2:
        signal_score += 0.25 * matrix[:, signal_positions[0]] * matrix[:, signal_positions[1]]
    signal_score += rng.normal(scale=np.std(signal_score) * 0.10 + 1e-6, size=n_rows)

    sparse_matrix = matrix.copy()
    for column_index in range(sparse_matrix.shape[1]):
        is_signal = column_index in set(signal_positions.tolist())
        keep_rate = float(rng.uniform(0.01, 0.05) if is_signal else rng.uniform(0.001, 0.018))
        values = sparse_matrix[:, column_index]
        if is_signal:
            threshold = np.nanquantile(np.abs(values), rng.uniform(0.88, 0.97))
            keep_mask = (np.abs(values) >= threshold) | (rng.random(n_rows) < keep_rate)
        else:
            keep_mask = rng.random(n_rows) < keep_rate
        sparse_matrix[:, column_index] = np.where(keep_mask, values, 0.0)

    for column_index, column in enumerate(numeric_columns):
        df[column] = sparse_matrix[:, column_index]

    extra_sparse_count = max(0, n_features - len(numeric_columns))
    activation_rate = float(rng.uniform(0.002, 0.035))
    if extra_sparse_count > 0:
        extra_values = stats.lognorm(s=float(rng.uniform(0.4, 1.2))).rvs(
            size=(n_rows, extra_sparse_count),
            random_state=rng,
        )
        extra_mask = rng.random((n_rows, extra_sparse_count)) < activation_rate
        extra_values = np.where(extra_mask, extra_values, 0.0)
        extra_df = pd.DataFrame(
            extra_values,
            columns=[f"sparse_num_{column_index:05d}" for column_index in range(extra_sparse_count)],
            index=df.index,
        )
        df = pd.concat([df, extra_df], axis=1)

    sparse_cat_columns = {}
    for i in range(max(2, n_features // 24)):
        active = rng.random(n_rows) < activation_rate * rng.uniform(0.5, 5.0)
        sparse_cat_columns[f"sparse_cat_{i:04d}"] = np.where(
            active,
            rng.choice([f"TOKEN_{i}_{j}" for j in range(int(rng.integers(8, 80)))], size=n_rows),
            None,
        )
    sparse_metadata = pd.DataFrame(
        {
            **sparse_cat_columns,
            "user_id": rng.choice([f"USER_{i:08d}" for i in range(max(20, n_rows // 5))], size=n_rows),
            "item_id": rng.choice([f"ITEM_{i:08d}" for i in range(max(20, n_rows // 4))], size=n_rows),
            "sparse_context": rng.choice(
                ["genomics_sparse", "recommendation", "click_prediction", "sparse_assay"],
                size=n_rows,
            ),
        },
        index=df.index,
    )
    rare_cutoff = np.quantile(signal_score, float(rng.uniform(0.93, 0.985)))
    sparse_metadata["rare_event_target"] = (signal_score >= rare_cutoff).astype(int)
    df = pd.concat([df, sparse_metadata], axis=1).copy()

    feature_columns = [
        column
        for column in df.columns
        if column not in _protected_columns(df)
        and column != "base_generator_source"
        and not column.endswith("_target")
    ]
    missing_rate = float(rng.uniform(0.20, 0.55))
    for column in feature_columns:
        if rng.random() < 0.45:
            mask = rng.random(n_rows) < missing_rate
            df.loc[mask, column] = np.nan

    df = _apply_multitask_targets(df, variant, rng)
    df = _apply_noise_and_missingness(df, variant, rng)
    df = _add_world_columns(df, world_label, "sparse")
    return {"sparse_features": df}


def generate_world(
    world_id: int,
    config: WorldConfig,
    seed: int,
    world_type: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Tuple[str, Dict[str, pd.DataFrame], object]:
    """Generate one synthetic world from the configured mixture prior."""

    rng = np.random.default_rng(seed)
    selected_type = world_type or sample_world_type(config, rng)
    if selected_type not in WORLD_TYPES:
        raise ValueError(f"Unsupported world type: {selected_type}")
    world_label = f"WORLD_{world_id:05d}_{selected_type.upper()}"
    if selected_type == "enterprise":
        tables, graph = _generate_enterprise_world(world_label, config, rng, progress_callback)
    elif selected_type == "scientific":
        tables, graph = _generate_scientific_world(world_label, config, rng, progress_callback), None
    elif selected_type == "iid":
        tables, graph = _generate_iid_world(world_label, config, rng, progress_callback), None
    elif selected_type == "temporal":
        tables, graph = _generate_temporal_world(world_label, config, rng, progress_callback), None
    elif selected_type == "relational":
        tables, graph = _generate_relational_world(world_label, config, rng, progress_callback), None
    elif selected_type == "sparse":
        tables, graph = _generate_sparse_world(world_label, config, rng, progress_callback), None
    else:
        tables, graph = _generate_advanced_forecasting_world(world_label, config, rng, progress_callback), None
    return world_label, tables, graph


def _generate_category_worlds(
    config: WorldConfig,
    n_worlds: int,
    seed: int,
    world_type: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    worlds: Dict[str, Dict[str, pd.DataFrame]] = {}
    for index in range(n_worlds):
        world_id, tables, _ = generate_world(index, config, seed + index * 9973, world_type=world_type)
        worlds[world_id] = tables
    return worlds


def generate_enterprise_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "enterprise")


def generate_scientific_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "scientific")


def generate_iid_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "iid")


def generate_temporal_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "temporal")


def generate_relational_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "relational")


def generate_sparse_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "sparse")


def generate_advanced_forecasting_worlds(config: WorldConfig, n_worlds: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _generate_category_worlds(config, n_worlds or config.n_worlds, seed or config.seed, "advanced_forecasting")


def _generate_world_worker(args: Tuple[int, WorldConfig, int]) -> Tuple[str, Dict[str, pd.DataFrame], object]:
    return generate_world(*args)


def generate_multiple_worlds(config: WorldConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generate many diverse worlds, optionally using multiprocessing."""

    seeds = [config.seed + i * 9973 for i in range(config.n_worlds)]
    tasks = [(i, config, seeds[i]) for i in range(config.n_worlds)]
    worlds: Dict[str, Dict[str, pd.DataFrame]] = {}
    if config.multiprocessing_workers and config.multiprocessing_workers > 1:
        with ProcessPoolExecutor(max_workers=config.multiprocessing_workers) as executor:
            futures = [executor.submit(_generate_world_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating worlds"):
                world_id, tables, _ = future.result()
                worlds[world_id] = tables
    else:
        for task in tqdm(tasks, desc="Generating worlds"):
            world_id, tables, _ = generate_world(*task)
            worlds[world_id] = tables
    return worlds
