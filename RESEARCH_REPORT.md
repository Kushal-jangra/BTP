# Research Report: Logs2Graphs Baseline Optimization & Novel Temp-OCDiGCN (T-Logs2Graphs) Architecture for Log Anomaly Detection

## Abstract
Event log anomaly detection in large-scale supercomputers and cloud environments is vital for ensuring system reliability. The **Logs2Graphs (OCDiGCN)** framework models system log sequences as attributed, directed, and weighted graphs and performs unsupervised anomaly detection using a One-Class Deep SVDD objective. This repository evaluates the **Optimized Baseline OCDiGCN** and **Temp-OCDiGCN (T-Logs2Graphs)** on a 10,000-graph BlueGene/L Protocol A split. The completed five-seed run gives the corrected baseline **84.38% ± 11.15% ROC-AUC** (95.30% peak) and the temporal model **78.86% ± 2.76% ROC-AUC** (83.50% peak). The earlier 82.44%/89.14% single-run values remain historical reference results.

---

## 1. Introduction & Protocol A Dataset Breakdown

### 1.1 Log Anomaly Detection & Graph Representation
Modern supercomputing clusters like BlueGene/L (BGL) produce millions of semi-structured log events. While traditional quantitative (e.g., PCA, OCSVM) and sequential (e.g., DeepLog, LogAnomaly) methods analyze event counts or linear sequences, graph-based methods capture structural relationships between log messages. 

**Logs2Graphs** converts groups of log messages into directed graphs $\mathcal{G} = (V, E, X, W)$:
- **Nodes ($V$):** Unique log event templates in a session/block.
- **Node Attributes ($X \in \mathbb{R}^{|V| \times 300}$):** Semantic embeddings of log template text.
- **Directed Edges ($E$):** Sequential transitions between consecutive log messages ($v_i \to v_j$).
- **Edge Weights ($W$):** Transition frequency counts ($Y_{ij}$).

### 1.2 Base Paper Protocol A Setup (10,000 Graphs Pool)
Following the base paper benchmark specification (**Protocol A**):
1. **Stratified Downsampling:** The total 69,251 BGL graph population is downsampled to **10,000 graphs** while preserving the original ~45.3% anomaly ratio (5,470 Normal graphs, 4,530 Anomalous graphs).
2. **70/5/25 Split Ratio on Normal Graphs:**
   - **Training Set (Train_Normal):** 70% of normal graphs (**3,829** normal graphs, 0 anomalous graphs).
   - **Validation Set (Val_Normal):** 5% of normal graphs (**274** normal graphs).
   - **Testing Set (Test_Normal):** 25% of normal graphs (**1,367** normal graphs).
3. **Anomalous Graph Distribution:**
   - **Train Set:** **0** anomalous graphs (strict One-Class Deep SVDD setting).
   - **Validation Set (Val_Anomaly):** Includes an equal number of anomalous graphs as normal validation graphs (**274** anomalous graphs) for hyperparameter tuning (50.0% validation anomaly ratio).
   - **Testing Set (Test_Anomaly):** The remaining **4,256** anomalous graphs are assigned to the test set alongside `Test_Normal` (75.69% test anomaly ratio).

| Metric / Split Stage | Protocol A Benchmark File Manifest | Total Graphs | Normal Graphs | Anomaly Graphs | Anomaly Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | [splits/train_graph_ids.csv](file:///Users/kushal/log2graph/splits/train_graph_ids.csv) | 3,829 | 3,829 | 0 | **0.00%** (Pure Normal) |
| **Validation Set** | [splits/validation_graph_ids.csv](file:///Users/kushal/log2graph/splits/validation_graph_ids.csv) | 548 | 274 | 274 | **50.00%** |
| **Test Set** | [splits/test_graph_ids.csv](file:///Users/kushal/log2graph/splits/test_graph_ids.csv) | 5,623 | 1,367 | 4,256 | **75.69%** |
| **Total Stratified Pool** | [splits/BGL_graph_split_manifest.csv](file:///Users/kushal/log2graph/splits/BGL_graph_split_manifest.csv) | **10,000** | **5,470** | **4,530** | **45.30%** |

---

## 2. Methodology & Architectural Enhancements

The implementation includes a reproducible five-seed evaluation path. Historical single-run scores in this report are retained as reference values and should not be treated as the final multi-seed mean until the benchmark command has been run.

### 2.1 Optimized Baseline OCDiGCN ($L=1$, 100 Epochs)
- **Single-Layer Directed Convolutions ($L=1$):** Single directed convolution layer (`DIGCNConv`) to prevent feature oversmoothing.
- **100-Epoch SVDD Convergence:** Extended training for 100 epochs minimizing the One-Class SVDD distance to centroid $o$.
- **Regex-cleaned TF-IDF Weighted 300-d Node Features:** Non-alphabetic characters are removed before TF-IDF/SVD, followed by strict row-wise L2 normalization.
- **Log-scaled transitions:** The baseline consumes $\log(1 + Y_{ij})$ directly from `edge_attr[:, 0]`; the runner no longer reverses this transformation.
- **Multi-seed initialization:** Seeds `[42, 100, 2024, 777, 1213]` control Python, NumPy, PyTorch, and CUDA initialization. The runner reports per-seed, mean ± standard deviation, and peak values and can write CSV output.

### 2.2 Novel Research Architecture — `Temp-OCDiGCN` (T-Logs2Graphs)
Incorporates inter-event time deltas $\Delta t = \max(0, t_{k+1} - t_k)$ alongside transition counts $Y_{ij}$ in a 3D temporal edge attribute vector $E_{ij} = [Y_{ij}, \bar{\Delta t}_{ij}, \text{Var}(\Delta t)_{ij}]$.

Using `TempDIGCNConv`, multi-dimensional edge features modulate node message passing via a dedicated edge MLP encoder:
$$\mathbf{h}_e = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot E_{ij} + b_1) + b_2\right)$$
$$\mathbf{m}_{i \to j} = \mathbf{h}_e \odot (W_{\text{node}} \cdot \mathbf{x}_j)$$

---

## 3. Experimental Results under Protocol A Benchmark

Reproducible multi-seed command:

```bash
.venv/bin/python run_bgl_baseline_csv.py --dataset bgl_processed_graphs.pt --splits-dir splits --seeds 42 100 2024 777 1213 --output-csv bgl_multiseed_results.csv
```

The checked-in dataset already contains regex-cleaned/L2-normalized features and log-scaled first-column edge weights. Regenerate it from the raw CSV when reproducing the artifact from source.

### 3.1 Completed Five-Seed Run

The benchmark was run with 100 epochs, learning rate `0.01`, weight decay `1e-4`, batch size `128`, and the five requested seeds. These are the measured results after correcting the baseline to consume log-scaled edge weights directly:

| Seed | Baseline ROC-AUC | Baseline PRC-AUC | Temp ROC-AUC | Temp PRC-AUC |
|---:|---:|---:|---:|---:|
| 42 | 95.30% | 97.90% | 76.25% | 90.81% |
| 100 | 75.60% | 89.35% | 80.41% | 92.70% |
| 2024 | 90.93% | 94.53% | 77.75% | 91.43% |
| 777 | 66.91% | 85.01% | 76.41% | 90.94% |
| 1213 | 93.17% | 95.33% | 83.50% | 93.28% |
| **Mean ± std** | **84.38% ± 11.15%** | **92.42% ± 4.63%** | **78.86% ± 2.76%** | **91.83% ± 0.99%** |
| **Peak** | **95.30%** | **97.90%** | **83.50%** | **93.28%** |

The corrected baseline reaches the requested 90–93% target on three of five seeds and exceeds it at peak, but its variance is high. The first temporal stabilization pass below reduced temporal variance but did not yet close the ROC-AUC gap.

### 3.2 Temporal Stabilization Result

After regenerating the dataset with `log1p` applied to transition count, mean delta, and delta variance, and adding LayerNorm to the temporal edge MLP:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std | PRC-AUC peak |
|---|---:|---:|---:|---:|
| Temp-OCDiGCN before stabilization | 78.86% ± 2.76% | 83.50% | 91.83% ± 0.99% | 93.28% |
| Temp-OCDiGCN after stabilization | **79.96% ± 1.55%** | **81.67%** | **91.83% ± 0.51%** | **92.34%** |

The remaining gap is now more stable rather than a time-feature scale failure. Further work should focus on temporal encoder capacity/aggregation and SVDD optimization.

### 3.1 Main Benchmark Results (10,000 Graph Pool, 100 Epochs)

| Model Architecture | Training Epochs | Layer Depth ($L$) | Edge Features | ROC-AUC Score | PRC-AUC Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical single-run baseline** | 100 | $L=1$ | Frequency $Y_{ij}$ | **82.44%** (0.8244) | **88.92%** (0.8892) |
| **Historical single-run Temp-OCDiGCN** | 100 | $L=1$ | $E_{ij} = [Y_{ij}, \bar{\Delta t}, \text{Var}(\Delta t)]$ | **89.14%** (0.8914) | **95.12%** (0.9512) |
| **Completed five-seed baseline mean ± std** | 100 | $L=1$ | Log-scaled $\log(1+Y_{ij})$ | **84.38% ± 11.15%** | **92.42% ± 4.63%** |
| **Completed five-seed Temp mean ± std (before stabilization)** | 100 | $L=1$ | Log-scaled temporal $E_{ij}$ | **78.86% ± 2.76%** | **91.83% ± 0.99%** |
| **Completed five-seed Temp mean ± std (after stabilization)** | 100 | $L=1$ | Log-scaled temporal $E_{ij}$ + LayerNorm | **79.96% ± 1.55%** | **91.83% ± 0.51%** |

---

## 4. Summary & Contributions

1. **Protocol A Downsampled Benchmark**: Successfully implemented stratified downsampling to 10,000 graphs with strict 70/5/25 normal graph splitting and balanced validation anomaly ratio.
2. **Corrected multi-seed baseline**: The baseline achieved **84.38% ± 11.15% ROC-AUC** and **92.42% ± 4.63% PRC-AUC**, with a **95.30% ROC-AUC peak**.
3. **Temporal stabilization**: Log-scaling all temporal edge channels and adding LayerNorm improved the temporal model to **79.96% ± 1.55% ROC-AUC** and **91.83% ± 0.51% PRC-AUC**, but additional architecture/optimization work is still required for the 90% ROC-AUC target.

---

## 5. Code Directory Reference
- [generate_downsampled_protocol_a_splits.py](file:///Users/kushal/log2graph/generate_downsampled_protocol_a_splits.py): Stratified downsampling generator for Protocol A 10k splits.
- [prepare_bgl_dataset.py](file:///Users/kushal/log2graph/prepare_bgl_dataset.py): Dataset builder constructing 300-d node embeddings and 3D temporal edge attributes.
- [TempDIGCNConv.py](file:///Users/kushal/log2graph/TempDIGCNConv.py): PyTorch Geometric multi-dimensional edge MLP convolution layer.
- [run_bgl_baseline_csv.py](file:///Users/kushal/log2graph/run_bgl_baseline_csv.py): Benchmark training and evaluation script under Protocol A splits.
