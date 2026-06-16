"""Input/output utilities for evaluating exported synthetic corpora.

The evaluator reads existing generator outputs only. It never calls the
synthetic generator. A dataset is treated as one exported table reconstructed
from train/validation/test splits when available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetRecord:
    """One reconstructed exported table."""

    dataset_id: str
    world_id: str
    table_name: str
    path: Path
    dataframe: pd.DataFrame


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create the evaluator output tree and return useful subdirectories."""

    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "reports": output_dir / "reports",
        "benchmark": output_dir / "benchmark",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_split_files(table_dir: Path) -> pd.DataFrame | None:
    frames = []
    for split in SPLIT_NAMES:
        parquet_path = table_dir / f"{split}.parquet"
        csv_path = table_dir / f"{split}.csv"
        if parquet_path.exists():
            frames.append(pd.read_parquet(parquet_path))
        elif csv_path.exists():
            frames.append(pd.read_csv(csv_path, low_memory=False))
    if not frames:
        return None
    return pd.concat(frames, axis=0, ignore_index=True)


def _discover_from_metadata(corpus_dir: Path) -> list[tuple[str, str, Path]]:
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    discovered = []
    for world_id, tables in metadata.get("worlds", {}).items():
        for table_name in tables.keys():
            table_dir = corpus_dir / world_id / table_name
            if table_dir.exists():
                discovered.append((world_id, table_name, table_dir))
    return discovered


def _discover_by_filesystem(corpus_dir: Path) -> list[tuple[str, str, Path]]:
    table_dirs = set()
    for pattern in ("train.parquet", "train.csv"):
        for split_path in corpus_dir.rglob(pattern):
            table_dirs.add(split_path.parent)
    discovered = []
    for table_dir in sorted(table_dirs):
        if table_dir.parent == corpus_dir:
            world_id = "UNKNOWN_WORLD"
        else:
            world_id = table_dir.parent.name
        discovered.append((world_id, table_dir.name, table_dir))
    return discovered


def load_corpus(corpus_dir: str | Path, sample_rows: int | None = None) -> list[DatasetRecord]:
    """Load all exported datasets from a corpus directory.

    Parameters
    ----------
    corpus_dir:
        Directory containing generator exports, usually the value passed to
        ``--output-dir`` during generation.
    sample_rows:
        Optional deterministic head sample for fast evaluation smoke tests.
    """

    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Synthetic corpus directory does not exist: {root}")

    discovered = _discover_from_metadata(root) or _discover_by_filesystem(root)
    records: list[DatasetRecord] = []
    seen: set[tuple[str, str]] = set()
    for world_id, table_name, table_dir in discovered:
        key = (world_id, table_name)
        if key in seen:
            continue
        seen.add(key)
        frame = _read_split_files(table_dir)
        if frame is None or frame.empty:
            continue
        if sample_rows is not None and sample_rows > 0 and len(frame) > sample_rows:
            frame = frame.head(sample_rows).copy()
        dataset_id = f"{world_id}::{table_name}"
        records.append(
            DatasetRecord(
                dataset_id=dataset_id,
                world_id=world_id,
                table_name=table_name,
                path=table_dir,
                dataframe=frame,
            )
        )
    return records


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write metric tables as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        table.to_json(output_dir / f"{name}.json", orient="records", indent=2)


def load_optional_benchmark_metafeatures(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load user-provided benchmark meta-feature CSV files."""

    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)

