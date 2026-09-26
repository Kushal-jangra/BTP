"""High-level post-anomaly Root-Causal Graph analysis API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .causal_inference import CausalEdge, estimate_causal_influence
from .graph_rank import CausalChain, extract_causal_chains, rank_root_causes
from .utils import coerce_importance, to_networkx_digraph


@dataclass(frozen=True)
class RootCausalResult:
    """All outputs of a Root-Causal Graph analysis."""

    graph: nx.DiGraph
    node_importance: dict[Any, float]
    causal_edges: list[CausalEdge]
    ranked_roots: list[tuple[Any, float]]
    causal_chains: list[CausalChain]


def extract_influential_subgraph(
    graph: Any,
    node_importance: Mapping[Any, float] | Sequence[float] | np.ndarray,
    top_k: int = 10,
) -> tuple[nx.DiGraph, dict[Any, float]]:
    """Keep top-K influential nodes and their original directed weighted edges."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    directed_graph = to_networkx_digraph(graph)
    importance = coerce_importance(node_importance, list(directed_graph.nodes))
    selected = sorted(directed_graph.nodes, key=lambda node: (-importance[node], str(node)))[:top_k]
    subgraph = directed_graph.subgraph(selected).copy()
    for node in subgraph.nodes:
        subgraph.nodes[node]["importance"] = importance[node]
    return subgraph, {node: importance[node] for node in selected}


def analyze_root_causes(
    graph: Any,
    node_importance: Mapping[Any, float] | Sequence[float] | np.ndarray,
    top_k: int = 10,
    max_hops: int = 4,
    chain_limit: int = 5,
) -> RootCausalResult:
    """Build and rank a Root-Causal Graph from one already-anomalous log graph."""
    subgraph, importance = extract_influential_subgraph(graph, node_importance, top_k)
    causal_edges = estimate_causal_influence(subgraph, importance)
    roots = rank_root_causes(subgraph, importance, causal_edges, max_hops=max_hops)
    chains = extract_causal_chains(
        subgraph, roots, causal_edges, importance, max_hops=max_hops, limit=chain_limit
    )
    return RootCausalResult(subgraph, importance, causal_edges, roots, chains)
