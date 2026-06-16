"""Relational multi-table generation."""

from synthetic_enterprise_generator.relational.tables import (
    RelationalGenerator,
    create_foreign_keys,
    generate_relational_tables,
)

__all__ = ["RelationalGenerator", "create_foreign_keys", "generate_relational_tables"]
