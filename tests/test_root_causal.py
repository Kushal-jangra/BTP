"""Unit tests for the independent Root-Causal Graph module."""

from __future__ import annotations

import networkx as nx

from root_causal import analyze_root_causes, draw_root_causal_graph, extract_influential_subgraph


def _event_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("AllocateBlock", event_template="AllocateBlock")
    graph.add_node("WriteBlock", event_template="WriteBlock")
    graph.add_node("PacketResponder", event_template="PacketResponder")
    graph.add_node("Unrelated", event_template="Unrelated")
    graph.add_edge("AllocateBlock", "WriteBlock", weight=9.0)
    graph.add_edge("WriteBlock", "PacketResponder", weight=7.0)
    graph.add_edge("Unrelated", "PacketResponder", weight=1.0)
    return graph


def test_subgraph_preserves_directed_weighted_edges_and_attributes() -> None:
    graph = _event_graph()
    subgraph, importance = extract_influential_subgraph(
        graph,
        {"AllocateBlock": 0.3, "WriteBlock": 0.7, "PacketResponder": 1.0, "Unrelated": 0.1},
        top_k=3,
    )
    assert set(subgraph) == {"AllocateBlock", "WriteBlock", "PacketResponder"}
    assert subgraph.has_edge("AllocateBlock", "WriteBlock")
    assert subgraph["AllocateBlock"]["WriteBlock"]["weight"] == 9.0
    assert subgraph.nodes["WriteBlock"]["event_template"] == "WriteBlock"
    assert importance["PacketResponder"] == 1.0


def test_upstream_event_is_ranked_as_root_and_chain_is_directed() -> None:
    result = analyze_root_causes(
        _event_graph(),
        {"AllocateBlock": 0.3, "WriteBlock": 0.7, "PacketResponder": 1.0, "Unrelated": 0.0},
        top_k=3,
        max_hops=3,
    )
    assert result.ranked_roots[0][0] == "AllocateBlock"
    assert any(chain.nodes == ("AllocateBlock", "WriteBlock", "PacketResponder") for chain in result.causal_chains)
    edge = next(edge for edge in result.causal_edges if edge.source == "AllocateBlock")
    assert 0.0 < edge.confidence <= 1.0


def test_visualization_exports_a_png(tmp_path) -> None:
    result = analyze_root_causes(_event_graph(), [0.3, 0.7, 1.0, 0.0], top_k=3)
    output = draw_root_causal_graph(result, tmp_path / "root-causal.png")
    assert output.suffix == ".png"
    assert output.exists()
    assert output.stat().st_size > 0
