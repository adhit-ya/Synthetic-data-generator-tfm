"""Stage 4: sequential workflow and event modeling."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from synthetic_enterprise_generator.config import WorkflowConfig


DOMAIN_EVENTS: Dict[str, List[str]] = {
    "retail": ["login", "browse", "search", "cart", "coupon", "payment", "refund", "support"],
    "industrial": [
        "sensor_start",
        "warmup",
        "heating",
        "nominal_run",
        "overload",
        "cooldown",
        "failure",
        "maintenance",
    ],
    "healthcare": [
        "patient_admission",
        "triage",
        "lab_order",
        "scan",
        "medication",
        "ICU",
        "discharge",
        "readmission",
    ],
}


def build_workflow_graph(
    config: WorkflowConfig,
    rng: np.random.Generator,
    event_vocab: Optional[Iterable[str]] = None,
) -> nx.DiGraph:
    """Build a weighted directed workflow graph."""

    events = list(event_vocab or DOMAIN_EVENTS.get(config.domain, DOMAIN_EVENTS["retail"]))
    graph = nx.DiGraph(domain=config.domain)
    for index, event in enumerate(events):
        graph.add_node(event, terminal=index >= len(events) - 2)

    for source, target in zip(events[:-1], events[1:]):
        graph.add_edge(source, target, weight=float(rng.uniform(0.55, 0.95)))
    for i, source in enumerate(events[:-2]):
        for target in events[i + 2 : min(len(events), i + 5)]:
            if rng.random() < config.branch_probability:
                graph.add_edge(source, target, weight=float(rng.uniform(0.05, 0.35)))
    if len(events) > 3:
        graph.add_edge(events[-2], events[1], weight=float(rng.uniform(0.02, 0.12)))
    return graph


def _sample_next_event(
    graph: nx.DiGraph,
    current_event: str,
    rng: np.random.Generator,
) -> Optional[str]:
    successors = list(graph.successors(current_event))
    if not successors:
        return None
    weights = np.array([graph[current_event][event].get("weight", 1.0) for event in successors])
    probabilities = weights / weights.sum()
    return str(rng.choice(successors, p=probabilities))


def generate_event_sequences(
    graph: nx.DiGraph,
    n_sequences: int,
    config: WorkflowConfig,
    rng: np.random.Generator,
    entity_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Generate ordered event chains from a workflow graph."""

    nodes = list(graph.nodes)
    starts = nodes[: max(1, min(3, len(nodes)))]
    entity_pool = list(entity_ids) if entity_ids is not None else [f"ENTITY_{i:08d}" for i in range(n_sequences)]
    rows = []
    for sequence_index in range(n_sequences):
        sequence_id = f"SEQ_{sequence_index:09d}"
        entity_id = str(rng.choice(entity_pool))
        current = str(rng.choice(starts))
        max_length = int(rng.integers(config.min_sequence_length, config.max_sequence_length + 1))
        for event_index in range(max_length):
            event_duration = float(rng.lognormal(mean=1.5, sigma=0.8))
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "entity_id": entity_id,
                    "event_index": event_index,
                    "event_type": current,
                    "event_duration_minutes": event_duration,
                    "event_cost": float(event_duration * rng.uniform(0.5, 8.0)),
                    "workflow_domain": graph.graph.get("domain", "unknown"),
                }
            )
            if rng.random() < config.terminal_probability and event_index >= config.min_sequence_length - 1:
                break
            next_event = _sample_next_event(graph, current, rng)
            if next_event is None:
                break
            current = next_event
    return pd.DataFrame(rows)


def simulate_enterprise_workflows(
    df: pd.DataFrame,
    config: WorkflowConfig,
    rng: np.random.Generator,
    entity_column: str = "customer_id",
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Attach simulated workflow events to an existing enterprise table."""

    out = df.copy()
    graph = build_workflow_graph(config, rng)
    if entity_column in out.columns:
        entity_ids = out[entity_column].dropna().astype(str).unique().tolist()
    else:
        entity_ids = [f"ENTITY_{i:08d}" for i in range(max(1, len(out) // 4))]
    n_sequences = max(1, min(len(out), int(np.ceil(len(out) / 3))))
    events = generate_event_sequences(graph, n_sequences, config, rng, entity_ids)
    if events.empty:
        out["sequence_id"] = None
        out["event_type"] = None
        return out, graph

    repeated_events = events.sample(
        n=len(out),
        replace=True,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)
    event_columns = [
        "sequence_id",
        "event_index",
        "event_type",
        "event_duration_minutes",
        "event_cost",
        "workflow_domain",
    ]
    for column in event_columns:
        out[column] = repeated_events[column].to_numpy()
    return out, graph


def transition_examples(graph: nx.DiGraph, limit: int = 12) -> List[Tuple[str, str, float]]:
    """Return human-readable transition examples for demos and logs."""

    examples = []
    for source, target, data in graph.edges(data=True):
        examples.append((source, target, float(data.get("weight", 1.0))))
        if len(examples) >= limit:
            break
    return examples

