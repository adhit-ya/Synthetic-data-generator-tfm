"""Train/validation/test split helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def split_dataframe(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """Shuffle and split a frame into train/validation/test partitions."""

    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    train_end = int(len(df) * train_fraction)
    valid_end = train_end + int(len(df) * validation_fraction)
    return {
        "train": df.iloc[indices[:train_end]].reset_index(drop=True),
        "validation": df.iloc[indices[train_end:valid_end]].reset_index(drop=True),
        "test": df.iloc[indices[valid_end:]].reset_index(drop=True),
    }

