"""Compatibility layer for legacy base-prior imports.

The implementation now lives in :mod:`synthetic_enterprise_generator.generators.base`.
This module keeps older import paths working while the project evolves.
"""

from __future__ import annotations

from synthetic_enterprise_generator.generators.base import (
    BaseSyntheticTableGenerator,
    BaseTableSpec,
    generate_classification_table,
    generate_mixed_schema_table,
    generate_regression_table,
)

BasePriorGenerator = BaseSyntheticTableGenerator

__all__ = [
    "BaseSyntheticTableGenerator",
    "BasePriorGenerator",
    "BaseTableSpec",
    "generate_classification_table",
    "generate_mixed_schema_table",
    "generate_regression_table",
]
