"""Schema inference and metadata helpers."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def infer_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Infer a compact schema description for metadata export."""

    schema: Dict[str, Dict[str, Any]] = {}
    for column in df.columns:
        series = df[column]
        schema[column] = {
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            schema[column].update(
                {
                    "mean": None if series.dropna().empty else float(series.mean()),
                    "std": None if series.dropna().empty else float(series.std(ddof=0)),
                }
            )
        else:
            examples = series.dropna().astype(str).unique()[:5].tolist()
            schema[column]["examples"] = examples
    return schema


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return table-level summary metadata used by previews and validators."""

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in df.columns if column not in numeric_columns
    ]
    target_columns = [column for column in df.columns if column.endswith("_target")]
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_columns": target_columns,
        "missing_rate": float(df.isna().mean().mean()) if not df.empty else 0.0,
        "schema": infer_schema(df),
    }
