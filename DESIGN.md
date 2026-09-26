# Root-Causal Graph Design

## Why this module is needed

OCDiGCN tells an operator that a log graph is anomalous and its node embeddings make
event-level explanations possible. That does not answer the operational question:
which earlier event chain most plausibly led to the anomalous event? The Root-Causal
Graph is a post-hoc, graph-constrained explanation layer. It turns important events
into directed candidate chains while preserving the observed transitions and their
frequencies.

It does not claim to discover causal effects from observational logs. Its output is a
ranked, evidence-backed hypothesis for investigation after anomaly detection.

## Algorithm

1. Accept one anomalous PyTorch Geometric `Data` graph (or a NetworkX directed graph),
   node importance scores, directed edges, edge weights, and node attributes/embeddings.
2. Keep the top-K important nodes. Induce a directed subgraph, retaining original node
   attributes and edge weights unchanged.
3. Score every retained transition using a transparent heuristic:

   `0.55 × normalized transition frequency + 0.25 × temporal-precedence + 0.20 × downstream importance`.

   Directed graph order is the default temporal signal. If numeric node timestamps are
   present, a backwards edge receives a strong penalty.
4. For each node, enumerate simple directed paths up to a small bounded hop count
   (default four). Each path contributes the product of transition confidences times
   the importance of its terminal event. The node's own importance is added and the
   result receives an upstream factor `1 / (1 + in-degree)`. This favors explanatory
   events that can reach important downstream symptoms and are less likely to be
   effects themselves.
5. Rank nodes by normalized score; emit the strongest chains and draw the induced graph.

The result includes `(event, score)` root rankings, `CausalChain` objects, and
per-edge confidences. The PNG uses darker blue for higher causal importance, orange for
the top root, directed arrows, and widths proportional to stored transition frequency.

## Complexity

Subgraph extraction and edge scoring are `O(V + E)` for the selected subgraph. Ranking
enumerates simple paths to depth `H`, which is `O(V × d^H)` in the bounded worst case,
where `d` is maximum out-degree. The default small `top_k=10` and `max_hops=4` make
this appropriate for interactive explanation; it avoids unbounded path enumeration.
Visualization is `O(V + E)` aside from the spring-layout iterations.

## OCDiGCN integration

The module is deliberately independent of the training path. The integration sequence
is:

```
OCDiGCN inference → SVDD anomaly score → node importance → analyze_root_causes()
```

`DataLoader.MeanTrainer.test()` already computes graph-level SVDD scores. The baseline
model returns node embeddings; a caller can supply its existing explanation scores, or
derive a default score from each node embedding's contribution to the graph embedding.
The Root-Causal package never modifies `DataLoader.py`, `DIGCNConv.py`, preprocessing,
or train/test splits.

## Limitations and future improvements

- Frequency and temporal order are evidence of dependency, not proof of causality.
- The original TU graph data lacks per-node timestamps, so temporal precedence is
  neutral unless timestamps are supplied by a caller.
- Top-K filtering can omit a low-importance but causally necessary intermediary.
- Cycles and highly branching graphs are controlled with a hop bound, which can omit
  long chains.

Future work can incorporate true log timestamps, counterfactual re-scoring through a
frozen OCDiGCN, learned edge attribution, domain constraints, and cross-graph recurring
cause aggregation. A persisted model checkpoint and a formal node-explanation API would
also let the demonstration notebook run end-to-end on every checkout.
