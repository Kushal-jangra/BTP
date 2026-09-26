"""Input conversion and small shared helpers for root-causal analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np


def as_float(value: Any, default: float = 1.0) -> float:
    """Convert tensor/scalar values to finite floats, falling back to ``default``."""
    try:
        result = float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def normalise_scores(scores: Mapping[Any, float], nodes: Sequence[Any]) -> dict[Any, float]:
    """Min-max normalise scores while retaining a useful value for constant scores."""
    values = np.asarray([max(0.0, as_float(scores.get(node, 0.0), 0.0)) for node in nodes])
    if values.size == 0:
        return {}
    maximum = values.max()
    if maximum == 0:
        return {node: 0.0 for node in nodes}
    return {node: float(value / maximum) for node, value in zip(nodes, values, strict=True)}


def node_label(graph: nx.DiGraph, node: Any) -> str:
    """Return a readable event-template label without imposing a schema."""
    attributes = graph.nodes[node]
    for key in ("event_template", "label", "name", "event"):
        if key in attributes:
            return str(attributes[key])
    return str(node)


def to_networkx_digraph(graph: Any) -> nx.DiGraph:
    """Convert a NetworkX directed graph or one PyG ``Data`` graph to NetworkX.

    PyG's ``edge_attr`` is interpreted as a transition weight. The first component is
    used when it is multi-dimensional, matching the repository's edge-frequency
    convention. Node embeddings are copied from ``x`` as an ``embedding`` attribute.
    """
    if isinstance(graph, nx.DiGraph):
        return graph.copy()
    if isinstance(graph, nx.MultiDiGraph):
        converted = nx.DiGraph()
        converted.add_nodes_from(graph.nodes(data=True))
        for source, target, attributes in graph.edges(data=True):
            weight = as_float(attributes.get("weight", attributes.get("frequency", 1.0)))
            if converted.has_edge(source, target):
                converted[source][target]["weight"] += weight
            else:
                converted.add_edge(source, target, **attributes, weight=weight)
        return converted
    if not hasattr(graph, "edge_index"):
        raise TypeError("graph must be a networkx directed graph or a PyG Data-like object")

    edge_index = graph.edge_index.detach().cpu().numpy() if hasattr(graph.edge_index, "detach") else np.asarray(graph.edge_index)
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, number_of_edges]")
    converted = nx.DiGraph()
    node_count = int(graph.num_nodes)
    features = getattr(graph, "x", None)
    for node in range(node_count):
        attributes: dict[str, Any] = {"original_node": node}
        if features is not None:
            row = features[node].detach().cpu().numpy() if hasattr(features[node], "detach") else np.asarray(features[node])
            attributes["embedding"] = row.copy()
        converted.add_node(node, **attributes)

    edge_attributes = getattr(graph, "edge_attr", None)
    for index, (source, target) in enumerate(edge_index.T):
        weight = 1.0
        if edge_attributes is not None:
            value = edge_attributes[index]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value).reshape(-1)
            if value.size:
                weight = as_float(value[0])
        source, target = int(source), int(target)
        if converted.has_edge(source, target):
            converted[source][target]["weight"] += weight
        else:
            converted.add_edge(source, target, weight=weight, frequency=weight)
    return converted


def coerce_importance(
    node_importance: Mapping[Any, float] | Sequence[float] | np.ndarray,
    nodes: Sequence[Any],
) -> dict[Any, float]:
    """Accept a node-keyed mapping or values aligned to graph node order."""
    if isinstance(node_importance, Mapping):
        return normalise_scores(node_importance, nodes)
    values = list(node_importance)
    if len(values) != len(nodes):
        raise ValueError("sequence node_importance must have one value per graph node")
    return normalise_scores(dict(zip(nodes, values, strict=True)), nodes)
