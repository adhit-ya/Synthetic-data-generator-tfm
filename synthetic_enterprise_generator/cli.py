"""Command-line interface for synthetic enterprise world generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from synthetic_enterprise_generator.config import load_config
from synthetic_enterprise_generator.export.dataset import export_dataset
from synthetic_enterprise_generator.generators.tabpfn_adapter import require_cuda_runtime
from synthetic_enterprise_generator.generators.worlds import WORLD_TYPES, generate_world
from synthetic_enterprise_generator.utils.logging import setup_logging
from synthetic_enterprise_generator.utils.random import set_global_seed
from synthetic_enterprise_generator.workflows.events import transition_examples

LOGGER = logging.getLogger(__name__)


def _schema_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "empty"
    numeric = len(table.select_dtypes(include=["number"]).columns)
    categorical = table.shape[1] - numeric
    missing = table.isna().mean().mean()
    return (
        f"{len(table):,} rows x {table.shape[1]:,} cols | "
        f"numeric={numeric}, categorical={categorical}, avg_missing={missing:.3f}"
    )


def _print_world_summary(world_id: str, tables: Dict[str, pd.DataFrame]) -> None:
    print(f"\n=== {world_id} schema summary ===")
    for table_name, table in tables.items():
        print(f"{table_name:16s} {_schema_summary(table)}")

    fact = tables.get("fact_events", pd.DataFrame())
    if fact.empty:
        return
    target_columns = [c for c in fact.columns if c.endswith("_target")]
    print(f"Targets: {', '.join(target_columns)}")
    missing_stats = fact.isna().mean().sort_values(ascending=False).head(8)
    print("\nTop missing-value rates:")
    for column, rate in missing_stats.items():
        print(f"  {column:28s} {rate:.3f}")

    event_columns = [
        c
        for c in ["sequence_id", "event_index", "event_type", "next_event_target"]
        if c in fact.columns
    ]
    if event_columns:
        example = fact[event_columns].sort_values(event_columns[:2]).head(10)
        print("\nExample workflow rows:")
        print(example.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic enterprise TabFM pretraining worlds."
    )
    parser.add_argument(
        "--config",
        default="synthetic_enterprise_generator/configs/default.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Override rows_per_world.")
    parser.add_argument("--worlds", type=int, default=None, help="Override n_worlds.")
    parser.add_argument(
        "--world-type",
        choices=WORLD_TYPES,
        default=None,
        help="Generate only this world family instead of sampling from mixture weights.",
    )
    parser.add_argument("--output-dir", default=None, help="Override export output directory.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Compute device for compatible base generators. Auto uses CUDA when available.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable row-level progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.rows is not None:
        config.rows_per_world = args.rows
        config.randomize_row_counts = False
    if args.worlds is not None:
        config.n_worlds = args.worlds
    if args.output_dir is not None:
        config.export.output_dir = args.output_dir
    if args.device is not None:
        config.compute_device = args.device
    if args.world_type is not None:
        config.world_type = args.world_type

    if str(config.compute_device).lower() == "cuda":
        require_cuda_runtime()

    set_global_seed(config.seed)
    LOGGER.info(
        "Starting generation: worlds=%s rows_per_world=%s world_type=%s device=%s",
        config.n_worlds,
        config.rows_per_world,
        config.world_type or "mixture",
        config.compute_device,
    )

    worlds: Dict[str, Dict[str, pd.DataFrame]] = {}
    graphs = {}
    expected_rows = None if config.randomize_row_counts else config.rows_per_world * config.n_worlds
    row_progress = None
    if not args.no_progress:
        row_progress = tqdm(
            total=expected_rows,
            desc="Rows generated",
            unit="rows",
            unit_scale=True,
            position=1,
            leave=True,
        )

    def progress_callback(stage: str, rows: int) -> None:
        if row_progress is None:
            return
        if stage:
            row_progress.set_postfix_str(stage[-80:])
        if rows:
            row_progress.update(rows)

    try:
        for world_index in tqdm(
            range(config.n_worlds),
            desc="Generating worlds",
            position=0,
            leave=True,
            disable=args.no_progress,
        ):
            world_id, tables, graph = generate_world(
                world_id=world_index,
                config=config,
                seed=config.seed + world_index * 9973,
                world_type=config.world_type,
                progress_callback=progress_callback,
            )
            worlds[world_id] = tables
            graphs[world_id] = graph
            _print_world_summary(world_id, tables)
    finally:
        if row_progress is not None:
            row_progress.close()

    for world_id, graph in graphs.items():
        if graph is None:
            continue
        print(f"\n=== {world_id} workflow transitions ===")
        for source, target, weight in transition_examples(graph):
            print(f"  {source:20s} -> {target:20s} p~{weight:.2f}")

    metadata = export_dataset(
        worlds,
        config.export,
        seed=config.seed,
        show_progress=not args.no_progress,
    )
    output_dir = Path(config.export.output_dir).resolve()
    print(f"\nExport complete: {output_dir}")
    print(f"Metadata worlds: {', '.join(metadata['worlds'].keys())}")


if __name__ == "__main__":
    main()
