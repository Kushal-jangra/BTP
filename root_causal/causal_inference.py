"""Topology-based causal influence scoring for post-hoc log analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .utils import as_float, normalise_scores


@dataclass(frozen=True)
class CausalEdge:
    """A directed event dependency and its post-hoc confidence."""

    source: Any
    target: Any
    confidence: float
    frequency: float


def _temporal_precedence(source_attributes: Mapping[str, Any], target_attributes: Mapping[str, Any]) -> float:
    """Reward a valid timestamp order when timestamps are available.

    Directed log transitions already encode order, so missing timestamps are neutral.
    """
    source_time = source_attributes.get("timestamp", source_attributes.get("time"))
    target_time = target_attributes.get("timestamp", target_attributes.get("time"))
    if source_time is None or target_time is None:
        return 1.0
    try:
        return 1.0 if float(source_time) <= float(target_time) else 0.05
    except (TypeError, ValueError):
        return 1.0


def estimate_causal_influence(
    graph: nx.DiGraph,
    node_importance: Mapping[Any, float],
) -> list[CausalEdge]:
    """Score observed event transitions using frequency, order, and importance.

    Confidence is a transparent heuristic rather than a causal-discovery claim:
    55% normalised transition frequency, 25% temporal precedence, and 20% importance
    of the downstream event. It is suitable for ranking observed dependencies after an
    anomaly has been detected.
    """
    if graph.number_of_edges() == 0:
        return []
    raw_frequencies = [max(0.0, as_float(attributes.get("weight", attributes.get("frequency", 1.0))))
                       for _, _, attributes in graph.edges(data=True)]
    max_frequency = max(raw_frequencies) or 1.0
    importance = normalise_scores(node_importance, list(graph.nodes))
    edges: list[CausalEdge] = []
    for source, target, attributes in graph.edges(data=True):
        frequency = max(0.0, as_float(attributes.get("weight", attributes.get("frequency", 1.0))))
        frequency_score = frequency / max_frequency
        precedence = _temporal_precedence(graph.nodes[source], graph.nodes[target])
        confidence = 0.55 * frequency_score + 0.25 * precedence + 0.20 * importance[target]
        edges.append(CausalEdge(source, target, float(np.clip(confidence, 0.0, 1.0)), frequency))
    return sorted(edges, key=lambda edge: edge.confidence, reverse=True)
