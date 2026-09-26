"""Rank likely upstream roots and extract bounded causal chains."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .causal_inference import CausalEdge
from .utils import normalise_scores


@dataclass(frozen=True)
class CausalChain:
    """A likely directed event chain terminating at an influential event."""

    nodes: tuple[Any, ...]
    confidence: float


def _edge_confidence(edges: Iterable[CausalEdge]) -> dict[tuple[Any, Any], float]:
    return {(edge.source, edge.target): edge.confidence for edge in edges}


def rank_root_causes(
    graph: nx.DiGraph,
    node_importance: Mapping[Any, float],
    causal_edges: Iterable[CausalEdge],
    max_hops: int = 4,
) -> list[tuple[Any, float]]:
    """Rank roots by their bounded, directed contribution to important descendants.

    For each candidate node, all simple paths up to ``max_hops`` are examined. A path
    contributes its edge-confidence product times the terminal node importance. The
    score also contains the node's own importance and an upstream bonus
    ``1 / (1 + in_degree)``; this makes otherwise similar causes favour earlier events.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be at least one")
    nodes = list(graph.nodes)
    importance = normalise_scores(node_importance, nodes)
    confidence = _edge_confidence(causal_edges)
    raw_scores: dict[Any, float] = {}
    for root in nodes:
        propagated = 0.0
        for target in nodes:
            if target == root or importance[target] == 0.0:
                continue
            for path in nx.all_simple_paths(graph, root, target, cutoff=max_hops):
                path_confidence = 1.0
                for source, destination in zip(path, path[1:]):
                    path_confidence *= confidence.get((source, destination), 0.0)
                propagated += path_confidence * importance[target]
        upstream_bonus = 1.0 / (1.0 + graph.in_degree(root))
        raw_scores[root] = upstream_bonus * (importance[root] + propagated)
    scores = normalise_scores(raw_scores, nodes)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))


def extract_causal_chains(
    graph: nx.DiGraph,
    ranked_roots: Iterable[tuple[Any, float]],
    causal_edges: Iterable[CausalEdge],
    node_importance: Mapping[Any, float],
    max_hops: int = 4,
    limit: int = 5,
) -> list[CausalChain]:
    """Return the highest-confidence bounded paths from the ranked root candidates."""
    importance = normalise_scores(node_importance, list(graph.nodes))
    confidence = _edge_confidence(causal_edges)
    candidates: list[CausalChain] = []
    for root, root_score in ranked_roots:
        for target in graph.nodes:
            if target == root or importance[target] == 0.0:
                continue
            for path in nx.all_simple_paths(graph, root, target, cutoff=max_hops):
                path_confidence = root_score * importance[target]
                for source, destination in zip(path, path[1:]):
                    path_confidence *= confidence.get((source, destination), 0.0)
                if path_confidence > 0:
                    candidates.append(CausalChain(tuple(path), path_confidence))
    candidates.sort(key=lambda chain: chain.confidence, reverse=True)
    unique: list[CausalChain] = []
    seen: set[tuple[Any, ...]] = set()
    for chain in candidates:
        if chain.nodes not in seen:
            unique.append(chain)
            seen.add(chain.nodes)
        if len(unique) == limit:
            break
    return unique
