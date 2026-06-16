"""Stage 6: relational multi-table generation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
try:
    from faker import Faker
except ModuleNotFoundError:  # pragma: no cover - lean local smoke environments.
    Faker = None

from synthetic_enterprise_generator.augmentation.entities import create_foreign_key_pool
from synthetic_enterprise_generator.config import EntityConfig


def create_foreign_keys(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    key_column: str,
    rng: np.random.Generator,
    concentration: float = 0.8,
) -> pd.DataFrame:
    """Assign a parent key to every child row using a long-tail distribution."""

    if key_column not in parent.columns:
        raise KeyError(f"Parent table does not contain key column {key_column}")
    out = child.copy()
    out[key_column] = create_foreign_key_pool(
        parent[key_column].dropna().astype(str).unique(),
        rng=rng,
        size=len(out),
        concentration=concentration,
    )
    return out


def generate_relational_tables(
    fact_table: pd.DataFrame,
    config: EntityConfig,
    rng: np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    """Create related dimension and event tables from a fact table."""

    fake = Faker() if Faker is not None else None
    if Faker is not None:
        Faker.seed(int(rng.integers(0, 2**31 - 1)))
    fact = fact_table.copy()

    customers = _customers_table(fact, fake, rng)
    products = _products_table(fact, rng)
    sessions = _sessions_table(fact)
    transactions = _transactions_table(fact, rng)
    machine_logs = _machine_logs_table(fact, rng)

    # Reapply FK values where needed so child tables remain connected even if a
    # previous augmentation stage produced partial IDs.
    if not transactions.empty and not customers.empty:
        transactions = create_foreign_keys(transactions, customers, "customer_id", rng)
    if not transactions.empty and not products.empty:
        transactions = create_foreign_keys(transactions, products, "product_id", rng)

    return {
        "fact_events": fact,
        "customers": customers,
        "products": products,
        "sessions": sessions,
        "transactions": transactions,
        "machine_logs": machine_logs,
    }


def _customers_table(fact: pd.DataFrame, fake: object, rng: np.random.Generator) -> pd.DataFrame:
    if "customer_id" not in fact.columns:
        return pd.DataFrame()
    customers = fact[["customer_id"]].dropna().drop_duplicates().reset_index(drop=True)
    customers["customer_name"] = [
        fake.company() if fake is not None else f"Synthetic Company {i:05d}"
        for i in range(len(customers))
    ]
    customers["industry"] = rng.choice(
        ["retail", "healthcare", "manufacturing", "finance", "logistics"],
        size=len(customers),
    )
    customers["region"] = rng.choice(["NA", "EU", "APAC", "LATAM"], size=len(customers))
    customers["contract_value"] = rng.lognormal(mean=10.2, sigma=0.9, size=len(customers))
    if "customer_segment" in fact.columns:
        segment = fact.groupby("customer_id")["customer_segment"].first()
        customers["customer_segment"] = customers["customer_id"].map(segment)
    return customers


def _products_table(fact: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if "product_id" not in fact.columns:
        return pd.DataFrame()
    products = fact[["product_id"]].dropna().drop_duplicates().reset_index(drop=True)
    products["category"] = rng.choice(
        ["software", "hardware", "service", "consumable", "subscription"],
        size=len(products),
    )
    products["unit_price"] = rng.lognormal(mean=4.5, sigma=0.8, size=len(products))
    products["margin_band"] = rng.choice(["low", "medium", "high"], p=[0.25, 0.50, 0.25], size=len(products))
    return products


def _sessions_table(fact: pd.DataFrame) -> pd.DataFrame:
    required = {"session_id", "customer_id"}
    if not required.issubset(fact.columns):
        return pd.DataFrame()
    aggregations = {"customer_id": "first"}
    if "timestamp" in fact.columns:
        aggregations["timestamp"] = ["min", "max"]
    if "event_type" in fact.columns:
        aggregations["event_type"] = "count"
    sessions = fact.groupby("session_id").agg(aggregations)
    sessions.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in sessions.columns
    ]
    sessions = sessions.reset_index()
    if "event_type_count" in sessions.columns:
        sessions = sessions.rename(columns={"event_type_count": "event_count"})
    return sessions


def _transactions_table(fact: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    columns = [c for c in ["transaction_id", "customer_id", "product_id", "session_id", "timestamp"] if c in fact.columns]
    if "transaction_id" not in columns:
        return pd.DataFrame()
    transactions = fact[columns].dropna(subset=["transaction_id"]).drop_duplicates("transaction_id")
    transactions = transactions.reset_index(drop=True)
    transactions["quantity"] = rng.poisson(lam=2.0, size=len(transactions)) + 1
    transactions["gross_amount"] = transactions["quantity"] * rng.lognormal(mean=4.2, sigma=0.7, size=len(transactions))
    transactions["payment_status"] = rng.choice(
        ["authorized", "captured", "failed", "refunded"],
        p=[0.15, 0.70, 0.10, 0.05],
        size=len(transactions),
    )
    return transactions


def _machine_logs_table(fact: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    columns = [c for c in ["machine_id", "timestamp", "event_type", "event_duration_minutes"] if c in fact.columns]
    if "machine_id" not in columns:
        return pd.DataFrame()
    logs = fact[columns].dropna(subset=["machine_id"]).copy().reset_index(drop=True)
    logs["log_id"] = [f"LOG_{i:010d}" for i in range(len(logs))]
    logs["temperature_c"] = rng.normal(loc=65, scale=12, size=len(logs))
    logs["vibration_score"] = rng.lognormal(mean=0.0, sigma=0.6, size=len(logs))
    logs["failure_flag"] = (
        (logs["temperature_c"] > 85) | (logs["vibration_score"] > 3.0)
    ).astype(int)
    return logs


class RelationalGenerator:
    """Object-oriented facade for connected multi-table generation."""

    def __init__(self, config: EntityConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def create_foreign_keys(
        self,
        child: pd.DataFrame,
        parent: pd.DataFrame,
        key_column: str,
        concentration: float = 0.8,
    ) -> pd.DataFrame:
        """Assign parent keys to a child table."""

        return create_foreign_keys(child, parent, key_column, self.rng, concentration)

    def generate_tables(self, fact_table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create the standard relational table set from a fact table."""

        return generate_relational_tables(fact_table, self.config, self.rng)

    def generate(self, fact_table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Alias for generate_tables for callers that expect generator APIs."""

        return self.generate_tables(fact_table)
