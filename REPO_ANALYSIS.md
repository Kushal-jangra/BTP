# Repository Analysis

## Repository structure

```
.
├── DIGCNConv.py                         # Directed graph convolution
├── TempDIGCNConv.py                     # Temporal directed convolution experiment
├── DataLoader.py                         # PyG dataset loading, splitting, trainer, models
├── GetAdjacencyMatrix.py                 # Directed/PPR adjacency transforms
├── GraphGeneration_BGL.py                # BGL log-to-graph generation
├── GraphGeneration_HDFS.py               # HDFS log-to-graph generation
├── GraphGeneration_Hadoop.py             # Hadoop log-to-graph generation
├── GraphGeneration_Spirit.py             # Spirit log-to-graph generation
├── GraphGeneration_Thunderbird.py        # Thunderbird log-to-graph generation
├── main_BGL.py                           # Baseline BGL runner
├── main_HDFS.py                          # Baseline HDFS runner
├── main_Hadoop.py                        # Baseline Hadoop runner
├── main_Spirit.py                        # Baseline Spirit runner
├── main_Thunderbird.py                   # Baseline Thunderbird runner
├── prepare_bgl_dataset.py                # BGL tensor preparation extension
├── prepare_bgl_paper_dataset.py          # Canonical BGL graph tensor preparation
├── prepare_bgl_temporal_features.py      # BGL temporal feature preparation
├── generate_downsampled_protocol_a_splits.py
├── run_bgl_baseline_csv.py
├── run_bgl_paper_exact.py                # Paper-aligned BGL DiGCN runner
├── run_bgl_temporal_exact.py
├── run_bgl_temporal_final.py
├── run_bgl_temporal_ablation.py
├── run_bgl_edge_temporal.py
├── README.md
├── RESEARCH_REPORT.md
├── TEMP_OCDIGCN_EXECUTION.md
├── GITHUB_HANDOFF.md
├── requirements.txt
└── bgl_*_results.csv                     # Committed experiment results
```

The `Data/` directory and model/data tensor artifacts are intentionally not committed.

## Entry points and training flow

There is no package-wide executable entry point. The baseline entry points are the five
`main_<DATASET>.py` scripts. Each follows the same flow:

```
Data/<dataset>/Graph/Raw → copied to Data/<dataset>/Raw
  → DataLoader.create_loaders()
  → DataLoader.load_data() / ParseDataset
  → PyG DataLoader
  → DiGCN + MeanTrainer.train()
  → MeanTrainer.test()
  → SVDD distances, AP, ROC-AUC
```

The later BGL research scripts (`run_bgl_paper_exact.py`,
`run_bgl_edge_temporal.py`, and related `run_bgl_temporal_*.py` scripts) are
self-contained experiment entry points that load locally generated graph tensors,
construct a DiGCN-based model, train it, and report ROC-AUC/PRC-AUC.

## Graph creation and available graph inputs

`GraphGeneration_{BGL,HDFS,Hadoop,Spirit,Thunderbird}.py` create a
`networkx.MultiDiGraph` for each log group. Nodes are unique event templates; directed
edges link consecutive events; repeated transitions become edge weights. Node attributes
are semantic event-template embeddings. The scripts write TU-style raw files:

- `*_A.txt` — directed edges;
- `*_edge_attributes.txt` — transition weights;
- `*_node_attributes.txt` — semantic embeddings;
- `*_graph_indicator.txt` and `*_graph_labels.txt` — graph boundaries and labels.

They then use `GetAdjacencyMatrix.py` to produce directed/PPR adjacency variants. At
runtime `DataLoader.read_tu_data()` exposes each graph as a PyTorch Geometric `Data`
object with `x`, `edge_index`, `edge_attr`, `edge_index2`, `edge_attr2`, and `y`.

## OCDiGCN definition, inference, and anomaly scores

The baseline model class is `DataLoader.DiGCN`, which uses two `DIGCNConv` layers from
`DIGCNConv.py`. `DiGCN.forward(data)` returns per-graph node embeddings. Its existing
one-class wrapper is `DataLoader.MeanTrainer`:

- `train()` first establishes an SVDD center from normal graph mean embeddings, then
  minimizes squared distance to that center;
- `test()` runs inference with `model.eval()`, mean-pools each graph's node embeddings,
  and computes `sum((F_test - center) ** 2)` as its anomaly score;
- `test()` returns AP, ROC-AUC, graph distances (`dists`), and labels.

The paper-aligned BGL script implements the equivalent explicitly:
`run_bgl_paper_exact.py:graph_embeddings()` performs inference and `run_seed()` computes
the norm of each graph embedding from the SVDD center as the anomaly score.

## Existing explanation status

The repository provides the node embeddings from which explanations can be derived but
does not contain a reusable node-importance/explanation API. The bottom of the baseline
main scripts converts an example test graph to NetworkX for visualization, but it does
not calculate or export node importance scores. A Root-Causal extension therefore needs
an adapter that accepts externally supplied node importances and a small helper that can
derive a transparent default importance from node embedding distance to the SVDD center.

## Files that must remain untouched

The following implement the already-completed preprocessing, data splitting, baseline,
or temporal research paths and will not be changed:

- `GraphGeneration_*.py`, `GetAdjacencyMatrix.py`, and `prepare_bgl_*.py`;
- `DataLoader.py`, `DIGCNConv.py`, `TempDIGCNConv.py`;
- all `main_*.py`, `run_bgl_*.py`, and `generate_downsampled_protocol_a_splits.py`;
- all result CSV files and all external/generated data artifacts.

## Root-Causal Graph extension points

The clean extension point is immediately after anomaly scoring and node explanation:

```
PyG anomalous Data graph + node importance scores + x + edge_index + edge_attr
  → root_causal.analyze_root_causes(...)
  → influential directed subgraph, causal-chain confidences, ranked roots, PNG
```

The module can consume a single unbatched PyG `Data` object directly, and can also
accept NetworkX directed graphs for standalone use. It must treat `edge_attr` as the
stored transition frequency, preserve it unchanged, and never alter model/training or
dataset artifacts. A notebook can load a trained model checkpoint when one is available;
because the baseline runners do not persist checkpoints, it must also make that absence
an explicit, actionable condition rather than silently retraining or regenerating data.
