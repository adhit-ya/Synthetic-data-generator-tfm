"""Reproducibility utilities."""

from __future__ import annotations

import random
import hashlib
from typing import Optional

import numpy as np

from synthetic_enterprise_generator.utils.torch_runtime import torch


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch, returning a NumPy generator."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a local generator without mutating global state."""

    return np.random.default_rng(seed)


def child_seed(parent_seed: int, *parts: object) -> int:
    """Derive a deterministic child seed from a parent seed and optional labels."""

    key = "|".join([str(parent_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32 - 1)
