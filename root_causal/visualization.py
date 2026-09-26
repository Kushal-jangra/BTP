"""PNG visualization for Root-Causal Graph results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# PNG export must also work on headless experiment machines and in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .causal_graph import RootCausalResult
from .utils import node_label


def draw_root_causal_graph(
    result: RootCausalResult,
    output_path: str | Path,
    title: str = "Root-Causal Graph",
) -> Path:
    """Render a directed causal graph to PNG and return its output path.

    Darker blue nodes indicate stronger causal importance, while the top-ranked root
    is highlighted in orange. Edge widths preserve relative transition frequency.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = result.graph
    figure, axis = plt.subplots(figsize=(10, 7))
    if graph.number_of_nodes() == 0:
        axis.set_title(title)
        axis.axis("off")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return path

    positions = nx.spring_layout(graph, seed=42, weight="weight")
    root = result.ranked_roots[0][0] if result.ranked_roots else None
    causal_importance = dict(result.ranked_roots)
    node_colours = [
        "#f97316" if node == root else plt.cm.Blues(0.25 + 0.7 * causal_importance.get(node, 0.0))
        for node in graph.nodes
    ]
    frequencies = [float(attributes.get("weight", attributes.get("frequency", 1.0)))
                   for _, _, attributes in graph.edges(data=True)]
    maximum = max(frequencies) if frequencies else 1.0
    widths = [1.0 + 4.0 * frequency / maximum for frequency in frequencies]
    nx.draw_networkx_nodes(graph, positions, node_color=node_colours, node_size=1500, ax=axis)
    nx.draw_networkx_edges(
        graph, positions, width=widths, edge_color="#475569", arrows=True,
        arrowsize=22, connectionstyle="arc3,rad=0.05", ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, labels={node: node_label(graph, node) for node in graph.nodes},
                            font_size=8, ax=axis)
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path
