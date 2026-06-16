"""Stage 2: enterprise identifier and entity augmentation."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from synthetic_enterprise_generator.config import EntityConfig


def _make_ids(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{i:08d}" for i in range(count)]


def add_entity_columns(
    df: pd.DataFrame,
    config: EntityConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add repeated enterprise entities with relational consistency.

    IDs are deliberately reused across rows. That reuse gives later modeling
    stages entity history and foreign-key continuity instead of one-off records.
    """

    out = df.copy()
    customers = _make_ids("CUST", config.n_customers)
    products = _make_ids("PROD", config.n_products)
    machines = _make_ids("MACH", config.n_machines)
    patients = _make_ids("PAT", config.n_patients)

    customer_weights = rng.dirichlet(np.ones(config.n_customers) * 0.8)
    product_weights = rng.dirichlet(np.ones(config.n_products) * 0.9)

    out["customer_id"] = rng.choice(customers, size=len(out), p=customer_weights)
    out["product_id"] = rng.choice(products, size=len(out), p=product_weights)
    out["machine_id"] = rng.choice(machines, size=len(out))
    out["patient_id"] = rng.choice(patients, size=len(out))
    transaction_offset = int(rng.integers(0, 2**31 - 1))
    out["transaction_id"] = [
        f"TXN_{transaction_offset + index:012X}" for index in range(len(out))
    ]

    # Stable entity attributes are mapped by ID so repeated entities carry state.
    customer_segment = {
        customer: rng.choice(["smb", "midmarket", "enterprise", "public_sector"])
        for customer in customers
    }
    patient_risk = {
        patient: rng.choice(["low", "medium", "high"], p=[0.55, 0.30, 0.15])
        for patient in patients
    }
    machine_family = {
        machine: rng.choice(["line_a", "line_b", "line_c", "edge_node"])
        for machine in machines
    }
    out["customer_segment"] = out["customer_id"].map(customer_segment)
    out["patient_risk_band"] = out["patient_id"].map(patient_risk)
    out["machine_family"] = out["machine_id"].map(machine_family)
    return out


def create_session_structure(
    df: pd.DataFrame,
    config: EntityConfig,
    rng: np.random.Generator,
    entity_column: str = "customer_id",
) -> pd.DataFrame:
    """Create grouped sessions with ordered row positions per entity."""

    out = df.copy()
    if entity_column not in out.columns:
        out = add_entity_columns(out, config, rng)

    rows = []
    for entity_id, group in out.groupby(entity_column, sort=False):
        group = group.copy()
        n_rows = len(group)
        if n_rows == 0:
            continue
        expected_sessions = max(1, int(np.ceil(n_rows / max(config.avg_rows_per_session, 1.0))))
        n_sessions = max(1, min(n_rows, int(rng.poisson(expected_sessions) + 1)))
        session_ids = [f"SES_{entity_id}_{i:05d}" for i in range(n_sessions)]
        session_assignment = np.repeat(session_ids, np.ceil(n_rows / n_sessions))[:n_rows]
        rng.shuffle(session_assignment)
        group["session_id"] = session_assignment
        group["session_row_index"] = group.groupby("session_id").cumcount()
        group["session_size"] = group.groupby("session_id")["session_id"].transform("size")
        rows.append(group)

    return pd.concat(rows, ignore_index=True)


def create_foreign_key_pool(
    values: Iterable[str],
    rng: np.random.Generator,
    size: int,
    concentration: float = 1.0,
) -> np.ndarray:
    """Sample foreign keys from a long-tail entity distribution."""

    value_list = list(values)
    probabilities = rng.dirichlet(np.ones(len(value_list)) * concentration)
    return rng.choice(value_list, size=size, p=probabilities)


class EntityAugmentor:
    """Object-oriented facade for enterprise entity augmentation."""

    def __init__(self, config: EntityConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def add_entity_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stable enterprise identifiers and entity attributes."""

        return add_entity_columns(df, self.config, self.rng)

    def create_session_structure(
        self,
        df: pd.DataFrame,
        entity_column: str = "customer_id",
    ) -> pd.DataFrame:
        """Create reusable sessions for the configured entity column."""

        return create_session_structure(df, self.config, self.rng, entity_column)

    def augment(
        self,
        df: pd.DataFrame,
        entity_column: str = "customer_id",
    ) -> pd.DataFrame:
        """Apply IDs and session structure in the standard entity stage order."""

        out = self.add_entity_columns(df)
        return self.create_session_structure(out, entity_column)
