# Logs2Graphs: Graph Neural Networks based Log Anomaly Detection and Explanation

[![Conference](https://img.shields.io/badge/ICSE--Companion-2024-blue.svg)](https://conf.researchr.org/home/icse-2024)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3639478.3643084-b31b1b.svg)](https://doi.org/10.1145/3639478.3643084)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-1682d4.svg)](https://doi.org/10.5281/zenodo.7771548)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-requirements.txt-blue.svg)](https://www.python.org/)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=ZhongLIFR%2FLogs2Graph&countColor=%23263759&style=flat)](https://visitorbadge.io/status?path=ZhongLIFR%2FLogs2Graph)

This is the official repository for the paper:

> **Graph Neural Networks based Log Anomaly Detection and Explanation**  
> Accepted by the **ICSE 2024 Companion Proceedings** poster track as a short paper.  
> [ICSE 2024](https://conf.researchr.org/home/icse-2024) | [DOI](https://doi.org/10.1145/3639478.3643084) | [Dataset](https://doi.org/10.5281/zenodo.7771548)

Logs2Graphs is a graph-based framework for **unsupervised log anomaly detection and explanation**. Instead of representing logs only as event-count vectors or linear event sequences, Logs2Graphs converts grouped log messages into **attributed, directed, and edge-weighted graphs**, and then performs graph-level anomaly detection with a one-class graph neural network.

The core model is **OCDiGCN**: One-Class Digraph Inception Convolutional Networks. It couples graph representation learning and anomaly detection, and provides node-level explanations by identifying log-event nodes that contribute most to the anomaly score.

---

## Highlights

- **Graph-based log representation**: log groups are represented as attributed, directed, edge-weighted graphs.
- **Structure-aware anomaly detection**: directed edges capture ordering and transition patterns between log events.
- **Semantic node attributes**: log events can be encoded as node attributes, allowing the model to use event semantics.
- **End-to-end graph-level detection**: OCDiGCN integrates graph representation learning with a one-class anomaly detection objective.
- **Anomaly explanation**: detected anomalies are explained by highlighting important log-event nodes.
- **Five benchmark datasets**: HDFS, Hadoop, Spirit, BGL, and Thunderbird.

---

## Method Overview

Logs2Graphs follows a graph-first view of log anomaly detection:

1. **Log parsing**: parse raw logs into log event templates and parameters.
2. **Log grouping**: divide parsed logs into meaningful log groups or sessions.
3. **Graph construction**: convert each log group into an attributed, directed, edge-weighted graph.
4. **Graph-level anomaly detection**: train OCDiGCN with a one-class objective to score anomalous graphs.
5. **Explanation**: decompose the anomaly score to identify influential nodes/log events.

This representation captures quantitative, sequential, semantic, and structural information in one graph object.

---

## Repository Structure

```text
Logs2Graph/
├── DIGCNConv.py                  # Directed graph convolution components
├── DataLoader.py                 # Data loading utilities
├── GetAdjacencyMatrix.py          # Adjacency matrix construction utilities
├── GraphGeneration_BGL.py         # Graph generation for BGL
├── GraphGeneration_HDFS.py        # Graph generation for HDFS
├── GraphGeneration_Hadoop.py      # Graph generation for Hadoop
├── GraphGeneration_Spirit.py      # Graph generation for Spirit
├── GraphGeneration_Thunderbird.py # Graph generation for Thunderbird
├── main_BGL.py                    # Detection on BGL
├── main_HDFS.py                   # Detection on HDFS
├── main_Hadoop.py                 # Detection on Hadoop
├── main_Spirit.py                 # Detection on Spirit
├── main_Thunderbird.py            # Detection on Thunderbird
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

The dataset is not stored directly in this repository. Please download it from Zenodo as described below.

---

## Setup

### Step 0: Check Requirements

For a Windows/Python 3.12 workspace, install the compatible environment with:

```bash
.\setup_windows.ps1
```

The original pinned `requirements.txt` remains for the legacy paper setup.

---

## Dataset

Download `Data.zip` from Zenodo:

[https://doi.org/10.5281/zenodo.7771548](https://doi.org/10.5281/zenodo.7771548)

After downloading, unzip it and place the resulting folder under the repository root:

```text
Logs2Graph/
├── Data/
├── GraphGeneration_HDFS.py
├── main_HDFS.py
└── ...
```

Please make sure the folder is named exactly:

```text
Data
```

If the unzipped folder has a different name, rename it to `Data`.

---

## Configure `root_path`

Before running experiments, replace the variable `root_path` at the beginning of each Python script with your local repository path.

Example:

```python
root_path = r"/Users/YourName/Desktop/Logs2Graph"
```

---

## Running Experiments

For each dataset, first generate graphs, then run the corresponding main script.

| Dataset | Graph Generation | Detection |
| --- | --- | --- |
| HDFS | `python GraphGeneration_HDFS.py` | `python main_HDFS.py` |
| Hadoop | `python GraphGeneration_Hadoop.py` | `python main_Hadoop.py` |
| Spirit | `python GraphGeneration_Spirit.py` | `python main_Spirit.py` |
| BGL | `python GraphGeneration_BGL.py` | `python main_BGL.py` |
| Thunderbird | `python GraphGeneration_Thunderbird.py` | `python main_Thunderbird.py` |

---

## BTP Research Extension: BGL Temporal Anomaly Detection

### Enhanced node feature builder

`enhanced_node_builder.py` provides an optional `EnhancedNodeBuilder` for
parsed log groups. Each unique event template (in first-occurrence order) gets
a normalized MiniLM contextual embedding, fixed seven-way severity encoding,
stable 32-bin hashed component encoding, within-group frequency and relative
first/last positions, and four signed-log-scaled numerical-parameter statistics
(mean, standard deviation, minimum, maximum). Missing parameters are zero-filled. Features are
standardized within each graph by default and returned as a `torch.float32`
matrix, ready for `torch_geometric.data.Data.x`.

Install the additional model dependency with `pip install -r requirements.txt`.
The first builder initialization downloads the configured Hugging Face model;
you can select another compatible model with `EnhancedNodeBuilder(model_name=...)`.
Records should be chronological dictionaries containing an event template
(`EventTemplate`, `event_template`, or `template`) and, when available, raw
message/content, severity/level, component/module, and parameter fields. For
example:

```python
from enhanced_node_builder import EnhancedNodeBuilder
from torch_geometric.data import Data

builder = EnhancedNodeBuilder()
x = builder.build_node_features(parsed_log_group)
graph = Data(x=x, edge_index=edge_index)
```

The existing BGL graph-generation script can opt into this feature path while
keeping the official GloVe path as its default:

```bash
python prepare_bgl_paper_dataset.py --enhanced-node-features \
  --output bgl_enhanced_graphs.pt
```

This builder is an opt-in feature path; the paper-aligned experiments below
continue using the existing official 200-dimensional embedding artifact.

### Run the enhanced model with the supplied local BGL dataset (Windows)

From the BTP workspace, install the isolated Python 3.12 environment:

```powershell
.\setup_windows.ps1
```

Then build enhanced graphs directly from the processed BGL CSV in Downloads.
The builder scans the source in chunks, creates a deterministic sample of up
to 10,000 groups, and writes graph tensors and split files under the workspace:

```powershell
.\.venv\Scripts\python.exe prepare_local_enhanced_bgl.py `
  --input 'C:\path\to\processed_log_data.csv'
```

Train and evaluate the OCDiGCN model on that feature matrix (example uses one
seed and 5 epochs to make the first run practical). The runner saves ROC and
Precision–Recall plots and threshold coordinates alongside a reloadable model
checkpoint:

```powershell
.\.venv\Scripts\python.exe run_bgl_paper_exact.py `
  --dataset bgl_enhanced_graphs.pt `
  --splits-dir splits_enhanced `
  --normalized-dataset bgl_enhanced_digcn_graphs.pt `
  --epochs 5 --seeds 42 --hidden-dim 128 --batch-size 64
```

The checkpoint is written to
`artifacts/enhanced_bgl/ocdigcn_seed_42.pt`; the combined curves are written to
`artifacts/enhanced_bgl/ocdigcn_seed_42_roc_pr_curves.png`. To load the model in
VS Code, open the BTP folder, select `.venv\Scripts\python.exe` as the Python
interpreter, then use:

```python
from load_enhanced_model import load_checkpoint, score_graphs
import torch

model, center, metadata, device = load_checkpoint(
    "artifacts/enhanced_bgl/ocdigcn_seed_42.pt"
)
graphs = torch.load("bgl_enhanced_digcn_graphs.pt", map_location="cpu", weights_only=False)
scores = score_graphs(model, center, graphs, device)
```

Or score the saved graph artifact from a terminal:

```powershell
.\.venv\Scripts\python.exe load_enhanced_model.py
```

The workspace includes VS Code launch profiles under `.vscode/launch.json`.
Open the BTP folder in VS Code, select **Run and Debug** on the sidebar, then
choose **Train and evaluate enhanced BGL** and press **F5**. The launch profile
points directly at this workspace's `.venv` interpreter and the prepared graph
files. To regenerate graphs first, choose **Build enhanced BGL graphs**.
To recreate the figure in a plot window from the saved CSV curve coordinates,
choose **Show ROC and PR curves** and press **F5**. Do not run the curve CSV
files with Code Runner; they contain coordinates, not Python code. The saved
PNG can also be opened by double-clicking it in the Explorer.

`setup_windows.ps1` uses the official CPU PyTorch wheel for a smaller,
repeatable installation; this default environment does not use the NVIDIA GPU.
For longer research runs, increase the epochs and pass more seeds after the
first end-to-end run completes.

The BTP extension preserves the original OCDiGCN baseline and adds a separate
temporal research branch for BGL Protocol A. The baseline remains the reference
model for every comparison.

### What was present originally

The original repository provides log-to-graph conversion, directed weighted
graphs, semantic event-template node features, the DiGCN/OCDiGCN one-class
anomaly detector, and explanation-oriented model components for five datasets.
The paper-aligned BGL implementation uses directed PPR-normalized adjacency,
200-dimensional template embeddings, hidden dimension 300, SGD optimization,
and a one-class hypersphere objective.

### What was added

The BGL research extension uses official structured templates, official
200-dimensional embeddings, and the original Protocol A splits. It extracts
transition counts plus inter-event timing statistics, reproduces the paper's
directed PPR normalization, standardizes temporal features using normal
training graphs only, and uses staged SVDD training.

Two temporal branches were evaluated:

- **Graph-level temporal residual**: summarizes timing statistics per graph and
  adds a bounded residual after DiGCN message passing.
- **Edge-level temporal fusion**: injects log-scaled mean and variance of
  inter-event time directly into edge messages.

### Results

All results use seeds `42`, `100`, `2024`, `777`, and `1213`.

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std |
| --- | ---: | ---: | ---: |
| Exact paper-aligned baseline | 91.91% ± 5.31% | 96.11% | 95.62% ± 2.67% |
| Graph-level temporal residual | 92.06% ± 0.53% | 92.56% | 95.57% ± 0.29% |
| **Edge-level temporal fusion** | **92.34% ± 0.36%** | **92.86%** | 95.04% ± 0.27% |

The edge-level branch improves mean ROC-AUC by approximately 0.43 percentage
points over the exact baseline and greatly reduces seed variance. Its PRC-AUC
is lower, so temporal edge fusion currently improves ROC ranking and robustness
but is not yet a universal improvement across every metric.

### How the result was achieved

The current edge-level configuration uses log-scaled mean and variance edge
features, PPR adjacency with `alpha=0.1`, temporal residual strength `0.05`,
10 epochs of temporal-only warm-up, base learning rate `0.0001`, temporal
learning rate `0.001`, hidden dimension `300`, batch size `128`, weight decay
`1e-4`, and 100 epochs. A bounded `tanh` residual prevents timing features
from erasing the semantic node representation.

### Reproducing the BGL experiments

```bash
./.venv/bin/python prepare_bgl_paper_dataset.py
./.venv/bin/python prepare_bgl_temporal_features.py
./.venv/bin/python run_bgl_paper_exact.py
./.venv/bin/python run_bgl_temporal_final.py
./.venv/bin/python run_bgl_edge_temporal.py
./.venv/bin/python run_bgl_temporal_ablation.py
```

Generated tensors, downloaded datasets, caches, and local paper files are
excluded from Git by `.gitignore` and should be regenerated locally.

### Research status and next direction

The baseline comparison is reproducible and the temporal branch is isolated.
The next direction is to improve the edge-level model's PRC-AUC without losing
its ROC-AUC gain, using multi-scale timing features and validation-only tuning
of residual strength and SVDD warm-up. Labels, anomaly ratios, and test splits
must remain unchanged for comparison with the base paper.

## References and Acknowledgements

This codebase builds on ideas and components from:

- [GLAM](https://github.com/sawlani/GLAM)
- [DiGCN](https://github.com/flyingtango/DiGCN)

We thank the authors of these repositories for making their code available.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{li2024graph,
  title={Graph Neural Networks based Log Anomaly Detection and Explanation},
  author={Zhong Li and Jiayang Shi and Matthijs van Leeuwen},
  booktitle={2024 IEEE/ACM 46th International Conference on Software Engineering: Companion Proceedings (ICSE-Companion)},
  pages={306--307},
  year={2024},
  doi={10.1145/3639478.3643084},
  url={https://doi.org/10.1145/3639478.3643084}
}
```

---

## License

The ICSE-Companion paper is published under the **CC BY 4.0** license. Please refer to this repository for code-specific licensing information.

---

## Contact

For questions, bug reports, or suggestions, please open an issue in this repository.

# BTP
