"""Workflow graph and event-sequence simulation."""

from synthetic_enterprise_generator.workflows.events import (
    build_workflow_graph,
    generate_event_sequences,
    simulate_enterprise_workflows,
    transition_examples,
)

__all__ = [
    "build_workflow_graph",
    "generate_event_sequences",
    "simulate_enterprise_workflows",
    "transition_examples",
]
