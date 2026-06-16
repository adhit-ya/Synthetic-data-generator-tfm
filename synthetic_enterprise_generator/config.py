"""Configuration objects for the synthetic enterprise generator.

The config layer is intentionally lightweight: dataclasses keep defaults close
to the code, while YAML files can override only the fields a run needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs.
    yaml = None


@dataclass
class EntityConfig:
    entity_types: List[str] = field(
        default_factory=lambda: [
            "customer_id",
            "session_id",
            "transaction_id",
            "product_id",
            "machine_id",
            "patient_id",
        ]
    )
    n_customers: int = 250
    n_products: int = 80
    n_machines: int = 40
    n_patients: int = 120
    avg_sessions_per_entity: float = 3.0
    avg_rows_per_session: float = 4.0


@dataclass
class TemporalConfig:
    start_date: str = "2022-01-01"
    periods: int = 365
    granularity: str = "daily"  # hourly, daily, weekly, monthly
    seasonality_strength: float = 0.25
    drift_strength: float = 0.15
    shift_probability: float = 0.25


@dataclass
class MissingnessConfig:
    mcar_rate: float = 0.03
    mar_rate: float = 0.04
    mnar_rate: float = 0.02
    noise_rate: float = 0.03
    outlier_rate: float = 0.01
    sparse_column_probability: float = 0.08


@dataclass
class WorkflowConfig:
    domain: str = "retail"  # retail, industrial, healthcare, auto
    min_sequence_length: int = 3
    max_sequence_length: int = 8
    branch_probability: float = 0.35
    terminal_probability: float = 0.18


@dataclass
class ExportConfig:
    output_dir: str = "synthetic_enterprise_generator/outputs"
    formats: List[str] = field(default_factory=lambda: ["csv", "parquet", "torch", "xlsx"])
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass
class WorldConfig:
    seed: int = 42
    n_worlds: int = 2
    rows_per_world: int = 2_000
    randomize_row_counts: bool = False
    world_type: Optional[str] = None
    min_features: int = 8
    max_features: int = 32
    max_table_cells: int = 10_000_000
    classification_classes: int = 3
    class_imbalance: float = 0.25
    world_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "enterprise": 1.0,
            "scientific": 1.0,
            "iid": 1.0,
            "temporal": 1.0,
            "relational": 1.0,
            "sparse": 1.0,
            "advanced_forecasting": 1.0,
        }
    )
    require_tabpfn_ecosystem: bool = False
    tabpfn_max_rows: int = 10_000
    compute_device: str = "auto"
    multiprocessing_workers: int = 1
    entity: EntityConfig = field(default_factory=EntityConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _merge_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    """Recursively merge a dictionary into a dataclass instance."""

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


def load_config(path: Optional[str | Path] = None) -> WorldConfig:
    """Load a YAML config file into :class:`WorldConfig`.

    Parameters
    ----------
    path:
        Optional YAML path. When omitted, defaults are returned.
    """

    config = WorldConfig()
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            raw = yaml.safe_load(handle) or {}
        else:
            raw = _simple_yaml_load(handle.read())
    if not isinstance(raw, dict):
        raise ValueError("YAML config must contain a mapping at the root.")
    return _merge_dataclass(config, raw)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by the default config.

    This fallback keeps the demo runnable in minimal Python environments. Full
    YAML support is still provided by PyYAML when installed.
    """

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root
