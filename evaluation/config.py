"""Configuration helpers for the corpus evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


@dataclass
class OpenMLConfig:
    enabled: bool = True
    dataset_ids: list[int] = field(default_factory=list)
    max_datasets: int = 15


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    openml: OpenMLConfig = field(default_factory=OpenMLConfig)
    local_paths: list[str] = field(default_factory=list)


@dataclass
class VisualizationConfig:
    formats: list[str] = field(default_factory=lambda: ["png", "pdf", "svg"])
    dpi: int = 300
    max_points_for_embedding: int = 500


@dataclass
class ReportConfig:
    title: str = "Mixture-of-Worlds Synthetic Corpus Evaluation"
    generate_pdf: bool = True


@dataclass
class EvaluationConfig:
    corpus_dir: str = "synthetic_enterprise_generator/outputs/smoke_all_world_types"
    output_dir: str = "evaluation/outputs"
    seed: int = 42
    sample_rows_per_dataset: int | None = 5000
    max_numeric_features_for_stats: int = 128
    max_correlation_features: int = 60
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    field_map = {f.name: f for f in fields(instance)}
    for key, value in updates.items():
        if key not in field_map:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, key, _merge_dataclass(current, value))
        else:
            setattr(instance, key, value)
    return instance


def load_evaluation_config(path: str | Path | None = None) -> EvaluationConfig:
    config = EvaluationConfig()
    if path is None:
        default_path = Path("evaluation/configs/default.yaml")
        path = default_path if default_path.exists() else None
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to load evaluation YAML configs.")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation YAML config must contain a mapping at the root.")
    return _merge_dataclass(config, raw)

