"""Post-hoc root-causal analysis for anomalous Logs2Graphs samples.

The package is deliberately model-agnostic: it consumes an already scored graph and
node-importance values, and never retrains or changes OCDiGCN.
"""

from .causal_graph import RootCausalResult, analyze_root_causes, extract_influential_subgraph
from .graph_rank import rank_root_causes
from .visualization import draw_root_causal_graph

__all__ = [
    "RootCausalResult",
    "analyze_root_causes",
    "draw_root_causal_graph",
    "extract_influential_subgraph",
    "rank_root_causes",
]
