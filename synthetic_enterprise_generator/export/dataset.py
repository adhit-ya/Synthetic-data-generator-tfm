"""Stage 9: export and PyTorch-compatible pretraining formatting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from synthetic_enterprise_generator.config import ExportConfig
from synthetic_enterprise_generator.export.excel import export_excel_preview
from synthetic_enterprise_generator.utils.schema import infer_schema
from synthetic_enterprise_generator.utils.splits import split_dataframe
from synthetic_enterprise_generator.utils.torch_runtime import torch, torch_unavailable_message

Dataset = torch.utils.data.Dataset if torch is not None else object

LOGGER = logging.getLogger(__name__)


class TabFMSyntheticDataset(Dataset):
    """PyTorch dataset wrapper for mixed-type synthetic enterprise tables.

    Categorical columns are factorized into integer IDs, numeric columns are
    float tensors, and missing masks are retained so future TabFM pretraining can
    learn imputation and uncertainty objectives.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_columns: Optional[Iterable[str]] = None,
        categorical_maps: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "torch is required to build TabFMSyntheticDataset. Install torch "
                "or remove 'torch' from export.formats for CSV/Parquet-only export. "
                f"{torch_unavailable_message()}"
            )
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_columns = list(target_columns or [c for c in dataframe.columns if c.endswith("_target")])
        self.feature_columns = [c for c in dataframe.columns if c not in self.target_columns]
        self.numeric_columns = self.dataframe[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [
            c for c in self.feature_columns if c not in self.numeric_columns
        ]
        self.categorical_maps = categorical_maps or self._build_categorical_maps()

        numeric_frame = self.dataframe[self.numeric_columns].astype(float) if self.numeric_columns else pd.DataFrame(index=self.dataframe.index)
        self.numeric_mask = torch.tensor(~numeric_frame.isna().to_numpy(), dtype=torch.bool)
        self.numeric = torch.tensor(numeric_frame.fillna(0.0).to_numpy(), dtype=torch.float32)
        self.categorical, self.categorical_mask = self._encode_categoricals()
        self.targets = self._encode_targets()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric[index],
            "numeric_mask": self.numeric_mask[index],
            "categorical": self.categorical[index],
            "categorical_mask": self.categorical_mask[index],
            "targets": self.targets[index],
        }

    def _build_categorical_maps(self) -> Dict[str, Dict[str, int]]:
        maps: Dict[str, Dict[str, int]] = {}
        for column in self.categorical_columns:
            values = self.dataframe[column].dropna().astype(str).unique().tolist()
            maps[column] = {value: idx + 1 for idx, value in enumerate(sorted(values))}
        return maps

    def _encode_categoricals(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.categorical_columns:
            empty = torch.empty((len(self.dataframe), 0), dtype=torch.long)
            return empty, torch.empty((len(self.dataframe), 0), dtype=torch.bool)
        arrays = []
        masks = []
        for column in self.categorical_columns:
            mapping = self.categorical_maps[column]
            series = self.dataframe[column]
            masks.append(~series.isna().to_numpy())
            encoded = series.astype(str).map(mapping).fillna(0).astype(int).to_numpy()
            encoded[series.isna().to_numpy()] = 0
            arrays.append(encoded)
        return (
            torch.tensor(np.vstack(arrays).T, dtype=torch.long),
            torch.tensor(np.vstack(masks).T, dtype=torch.bool),
        )

    def _encode_targets(self) -> torch.Tensor:
        if not self.target_columns:
            return torch.empty((len(self.dataframe), 0), dtype=torch.float32)
        encoded = []
        for column in self.target_columns:
            series = self.dataframe[column]
            if pd.api.types.is_numeric_dtype(series):
                encoded.append(series.astype(float).fillna(0.0).to_numpy())
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                codes[series.isna().to_numpy()] = -1
                encoded.append(codes.astype(float))
        return torch.tensor(np.vstack(encoded).T, dtype=torch.float32)


def build_pytorch_dataset(
    dataframe: pd.DataFrame,
    target_columns: Optional[Iterable[str]] = None,
) -> TabFMSyntheticDataset:
    """Build an in-memory PyTorch dataset from a pandas DataFrame."""

    return TabFMSyntheticDataset(dataframe=dataframe, target_columns=target_columns)


def export_dataset(
    worlds: Dict[str, Dict[str, pd.DataFrame]],
    config: ExportConfig,
    seed: int,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Export worlds to CSV, Parquet, Torch tensors, and metadata JSON."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {"worlds": {}, "formats": config.formats}
    total_rows = sum(len(table) for tables in worlds.values() for table in tables.values())
    export_progress = tqdm(
        total=total_rows,
        desc="Rows exported",
        unit="rows",
        unit_scale=True,
        disable=not show_progress,
    )

    try:
        for world_id, tables in worlds.items():
            world_dir = output_dir / world_id
            world_dir.mkdir(parents=True, exist_ok=True)
            if "xlsx" in config.formats or "excel" in config.formats:
                export_excel_preview(
                    world_id=world_id,
                    tables=tables,
                    output_path=world_dir / "generated_data_preview.xlsx",
                )
            metadata["worlds"][world_id] = {}
            for table_name, table in tables.items():
                export_progress.set_postfix_str(f"{world_id}/{table_name}"[-80:])
                table_dir = world_dir / table_name
                table_dir.mkdir(parents=True, exist_ok=True)
                metadata["worlds"][world_id][table_name] = {
                    "n_rows": int(len(table)),
                    "n_columns": int(table.shape[1]),
                    "schema": infer_schema(table),
                }
                if table.empty:
                    continue
                splits = split_dataframe(
                    table,
                    train_fraction=config.train_fraction,
                    validation_fraction=config.validation_fraction,
                    test_fraction=config.test_fraction,
                    seed=seed,
                )
                for split_name, split_df in splits.items():
                    if "csv" in config.formats:
                        split_df.to_csv(table_dir / f"{split_name}.csv", index=False)
                    if "parquet" in config.formats:
                        split_df.to_parquet(table_dir / f"{split_name}.parquet", index=False, engine="pyarrow")
                    if "torch" in config.formats:
                        if torch is None:
                            LOGGER.warning("Skipping torch export for %s/%s because torch is not installed.", world_id, table_name)
                        else:
                            dataset = build_pytorch_dataset(split_df)
                            torch.save(
                                {
                                    "numeric": dataset.numeric,
                                    "numeric_mask": dataset.numeric_mask,
                                    "categorical": dataset.categorical,
                                    "categorical_mask": dataset.categorical_mask,
                                    "targets": dataset.targets,
                                    "feature_columns": dataset.feature_columns,
                                    "target_columns": dataset.target_columns,
                                    "numeric_columns": dataset.numeric_columns,
                                    "categorical_columns": dataset.categorical_columns,
                                    "categorical_maps": dataset.categorical_maps,
                                },
                                table_dir / f"{split_name}.pt",
                            )
                    export_progress.update(len(split_df))
    finally:
        export_progress.close()

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    LOGGER.info("Exported synthetic worlds to %s", output_dir)
    return metadata
