"""Synthetic enterprise dataset generation toolkit for TabFM pretraining.

The package uses the TabPFN ecosystem as the preferred source of base tabular
priors, then adds temporal, relational, workflow, missingness, and multitask
structure around those base tables.
"""

from synthetic_enterprise_generator.config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config"]

