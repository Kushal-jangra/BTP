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

Install the required packages:

```bash
pip install -r requirements.txt
```

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
