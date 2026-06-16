"""Statistical diversity and meta-feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from evaluation.io import DatasetRecord
from evaluation.metrics.corpus_statistics import infer_feature_groups


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    groups = infer_feature_groups(df)
    columns = groups["numerical"]
    if not columns:
        return pd.DataFrame(index=df.index)
    return df[columns].apply(pd.to_numeric, errors="coerce")


def _hist_entropy(series: pd.Series, bins: int = 20) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.nanstd(values) < 1e-12:
        return 0.0
    hist, _ = np.histogram(values, bins=min(bins, max(3, int(np.sqrt(len(values))))), density=False)
    probabilities = hist / max(hist.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def _mutual_information_summary(df: pd.DataFrame, numeric: pd.DataFrame) -> dict[str, float]:
    target_columns = [column for column in df.columns if column.endswith("_target")]
    if numeric.empty or not target_columns:
        return {"mutual_information_mean": 0.0, "mutual_information_max": 0.0}
    X = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median()).fillna(0.0)
    if X.shape[1] > 50:
        X = X.iloc[:, :50]
    values = []
    for target in target_columns[:5]:
        y = df[target]
        mask = y.notna()
        if mask.sum() < 20:
            continue
        y_valid = y.loc[mask]
        X_valid = X.loc[mask]
        try:
            if pd.api.types.is_numeric_dtype(y_valid) and y_valid.nunique() > 20:
                mi = mutual_info_regression(X_valid, y_valid.astype(float), random_state=0)
            else:
                codes, _ = pd.factorize(y_valid.astype(str), sort=True)
                mi = mutual_info_classif(X_valid, codes, random_state=0)
            values.extend(np.nan_to_num(mi).tolist())
        except Exception:
            continue
    if not values:
        return {"mutual_information_mean": 0.0, "mutual_information_max": 0.0}
    return {
        "mutual_information_mean": float(np.mean(values)),
        "mutual_information_max": float(np.max(values)),
    }


def statistical_summary(records: list[DatasetRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        df = record.dataframe
        numeric = _numeric_features(df)
        if numeric.empty:
            rows.append(
                {
                    "dataset_id": record.dataset_id,
                    "numeric_feature_count": 0,
                    "skewness_mean": 0.0,
                    "skewness_std": 0.0,
                    "kurtosis_mean": 0.0,
                    "kurtosis_std": 0.0,
                    "entropy_mean": 0.0,
                    "entropy_std": 0.0,
                    "abs_correlation_mean": 0.0,
                    "abs_correlation_max": 0.0,
                    "mutual_information_mean": 0.0,
                    "mutual_information_max": 0.0,
                    "statistical_diversity_score": 0.0,
                }
            )
            continue

        limited = numeric.iloc[:, : min(128, numeric.shape[1])]
        skewness = limited.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
        kurtosis = limited.kurtosis(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
        entropies = pd.Series([_hist_entropy(limited[column]) for column in limited.columns])
        if limited.shape[1] >= 2:
            corr = limited.corr(numeric_only=True).abs().replace([np.inf, -np.inf], np.nan)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            corr_mean = float(upper.mean()) if not upper.empty else 0.0
            corr_max = float(upper.max()) if not upper.empty else 0.0
        else:
            corr_mean = 0.0
            corr_max = 0.0
        mi = _mutual_information_summary(df, limited)
        entropy_norm = float(np.tanh(entropies.mean() / 3.0)) if not entropies.empty else 0.0
        shape_norm = float(np.tanh((skewness.abs().mean() + kurtosis.abs().mean()) / 8.0)) if not skewness.empty else 0.0
        corr_norm = float(np.clip(corr_mean, 0, 1))
        mi_norm = float(np.tanh(mi["mutual_information_mean"]))
        score = float(np.clip(0.30 * entropy_norm + 0.25 * shape_norm + 0.20 * corr_norm + 0.25 * mi_norm, 0, 1))
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "numeric_feature_count": int(numeric.shape[1]),
                "skewness_mean": float(skewness.mean()) if not skewness.empty else 0.0,
                "skewness_std": float(skewness.std(ddof=0)) if not skewness.empty else 0.0,
                "kurtosis_mean": float(kurtosis.mean()) if not kurtosis.empty else 0.0,
                "kurtosis_std": float(kurtosis.std(ddof=0)) if not kurtosis.empty else 0.0,
                "entropy_mean": float(entropies.mean()) if not entropies.empty else 0.0,
                "entropy_std": float(entropies.std(ddof=0)) if len(entropies) else 0.0,
                "abs_correlation_mean": corr_mean,
                "abs_correlation_max": corr_max,
                **mi,
                "statistical_diversity_score": score,
            }
        )
    return pd.DataFrame(rows)


def metafeatures(dataset_stats: pd.DataFrame, statistical_stats: pd.DataFrame, source: str = "synthetic") -> pd.DataFrame:
    """Build comparable dataset-level meta-features for embeddings."""

    if dataset_stats.empty:
        return pd.DataFrame()
    joined = dataset_stats.merge(statistical_stats, on="dataset_id", how="left")
    columns = [
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
    out = joined[["dataset_id", "world_family", *columns]].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["source"] = source
    out["benchmark_family"] = source
    return out

