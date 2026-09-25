# Temp-OCDiGCN Execution Record

## Purpose

This document records the implementation changes and experiments being executed to improve the proposed **Temp-OCDiGCN** model. The objective is to determine whether temporal edge information improves anomaly detection over the corrected OCDiGCN baseline under Protocol A.

The target is ROC-AUC above 90% across the requested five random seeds. That target has not yet been reached, so the current results are treated as experimental evidence rather than a final research claim.

## Dataset and benchmark

- Dataset: BlueGene/L (BGL)
- Protocol: Protocol A
- Graphs: 10,000
- Training: 3,829 normal graphs only
- Validation: 548 graphs, balanced normal/anomaly
- Testing: 5,623 graphs
- Seeds: `42, 100, 2024, 777, 1213`
- Epochs: 100
- Learning rate: `0.01`
- Weight decay: `1e-4`
- Batch size: `128`
- Objective: one-class SVDD distance to a fixed hypersphere center

## What was changed

### 1. Regex-cleaned node features

In `prepare_bgl_dataset.py`, log-template text is cleaned using:

```python
re.sub(r'[^a-zA-Z\\s]', '', message).lower()
```

The cleaned templates are transformed with TF-IDF, projected to 300 dimensions using TruncatedSVD, and row-wise L2-normalized. Zero vectors remain zero because their norm is replaced with `1.0` before division.

### 2. Log-scaled transition counts

Transition counts are stored as:

```python
log1p(count)
```

The baseline runner now consumes this value directly. Previously, the runner incorrectly applied `expm1`, converting the values back to raw counts. That created a mismatch between the dataset preparation and model execution.

### 3. Log-scaled temporal features

All three temporal edge channels are now scaled:

```python
edge_attr = [
    log1p(transition_count),
    log1p(mean_inter_event_delta),
    log1p(variance_inter_event_delta),
]
```

This prevents large time-delta values and variances from dominating the temporal edge encoder.

### 4. Layer-normalized temporal edge encoder

`TempDIGCNConv.py` now uses:

```python
Linear(edge_dim, out_channels, bias=False)
LayerNorm(out_channels)
ReLU()
Linear(out_channels, out_channels, bias=False)
```

The edge encoder output multiplicatively modulates messages during directed graph convolution.

### 5. Reproducible multi-seed execution

The runner now controls Python, NumPy, PyTorch, and CUDA seeds. It also supports command-line parameters and optional CSV output:

```bash
.venv/bin/python run_bgl_baseline_csv.py \\
  --dataset bgl_processed_graphs_temp.pt \\
  --splits-dir splits \\
  --seeds 42 100 2024 777 1213 \\
  --epochs 100 \\
  --batch-size 128 \\
  --output-csv bgl_multiseed_results_temp.csv
```

## Measured results

### Corrected baseline

| Metric | Result |
|---|---:|
| ROC-AUC mean ± std | 84.38% ± 11.15% |
| ROC-AUC peak | 95.30% |
| PRC-AUC mean ± std | 92.42% ± 4.63% |
| PRC-AUC peak | 97.90% |

### Temp-OCDiGCN before temporal stabilization

| Metric | Result |
|---|---:|
| ROC-AUC mean ± std | 78.86% ± 2.76% |
| ROC-AUC peak | 83.50% |
| PRC-AUC mean ± std | 91.83% ± 0.99% |
| PRC-AUC peak | 93.28% |

### Temp-OCDiGCN after temporal stabilization

| Metric | Result |
|---|---:|
| ROC-AUC mean ± std | 79.96% ± 1.55% |
| ROC-AUC peak | 81.67% |
| PRC-AUC mean ± std | 91.83% ± 0.51% |
| PRC-AUC peak | 92.34% |

## Interpretation

The temporal changes improved the mean ROC-AUC by approximately 1.10 percentage points and reduced the ROC-AUC standard deviation from 2.76% to 1.55%. They did not reproduce the earlier 89.14% single-run result.

This means the novelty is implemented, but its performance benefit is not yet demonstrated. The current evidence supports the claim that the model introduces temporal edge-aware message passing; it does not yet support claiming that the temporal architecture improves ROC-AUC over the corrected baseline.

The baseline's large seed variance also shows that the current SVDD setup is sensitive to initialization. The temporal branch may be suppressing useful temporal information through its message modulation, or the fixed-center SVDD objective may be favoring the simpler baseline representation.

## Next experiments required to recover the novelty result

The next execution phase should isolate the source of the temporal underperformance:

1. Compare temporal encoders with and without message gating. Test additive fusion and residual fusion in addition to multiplicative modulation.
2. Normalize the temporal edge channels using training-set statistics and compare that against `log1p` scaling.
3. Tune the temporal edge MLP width, activation, and learning rate independently from the baseline.
4. Recompute the SVDD center after a short warm-up instead of using only the untrained representation.
5. Evaluate the temporal model with the same initialization and loader order as the baseline for paired seed comparisons.
6. Record every ablation in a separate CSV before selecting the final model.

No final novelty claim should be made until the selected temporal configuration beats the corrected baseline on the predefined five-seed evaluation, not only on one favorable seed.

## Latest additive-fusion and warm-up experiment

The next proposed experiment was executed with three additional changes:

- Temporal edge messages were changed from multiplicative gating to additive fusion: `x_j + edge_emb`.
- Mean log-delta and variance log-delta channels were standardized using statistics fitted only on normal training edges.
- The SVDD center was recomputed after a five-epoch warm-up, followed by the remaining 95 epochs.

Measured results:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std | PRC-AUC peak |
|---|---:|---:|---:|---:|
| Baseline + warm-up | 87.95% ± 10.05% | 96.92% | 94.52% ± 4.30% | 98.40% |
| Temp additive + standardization + warm-up | 76.34% ± 4.74% | 82.35% | 89.89% ± 1.47% | 91.58% |

This result rejects the simple additive-fusion hypothesis under the current training settings. The warm-up improved the baseline, but the temporal branch degraded. The likely issue is that directly adding an unconstrained edge projection to every node message changes the scale and direction of the graph representation too aggressively. The next temporal ablation should use a bounded residual coefficient or concatenation followed by a learned fusion layer, while keeping the standardized edge channels and warm-up fixed.

## Hybrid baseline-preserving residual experiment

The next architecture preserved the baseline transition-weighted message path and added a bounded temporal residual:

```python
baseline_message = log_count_weight * x_j
temporal_residual = alpha * tanh(edge_mlp(temporal_features))
message = baseline_message + temporal_residual
```

The temporal coefficient was initialized at `alpha=0.1`. This experiment used the regenerated dataset with training-normal temporal z-score statistics and disabled warm-up (`--warmup-epochs 0`) to isolate the architecture.

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std | PRC-AUC peak |
|---|---:|---:|---:|---:|
| Corrected baseline | 84.38% ± 11.15% | 95.30% | 92.42% ± 4.63% | 97.90% |
| Hybrid temporal residual | 78.18% ± 4.66% | 83.24% | 90.69% ± 1.76% | 92.65% |

The hybrid design improved the best temporal seed relative to earlier temporal runs, but it did not improve the mean or exceed the baseline. The next implementation should sweep `alpha` values including `0.0`, `0.01`, `0.05`, and `0.1`, and test graph-level fusion rather than injecting the residual into every edge message.

### Residual-strength control: `alpha=0.01`

The residual coefficient was made configurable in `run_bgl_baseline_csv.py` and evaluated at `alpha=0.01`, with all other settings unchanged and warm-up disabled:

| Alpha | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std |
|---:|---:|---:|---:|
| 0.10 | 78.18% ± 4.66% | 83.24% | 90.69% ± 1.76% |
| 0.01 | 77.16% ± 3.85% | 80.94% | 90.37% ± 0.97% |

Reducing the residual strength did not recover performance. This suggests that the problem is not simply temporal residual magnitude. The next implementation should move temporal fusion to the graph representation level, where temporal statistics can be learned without perturbing every directed node message.

## Graph-level temporal fusion experiment

The temporal branch was moved after graph pooling. The node path uses the baseline directed convolution with transition-count weights. Temporal mean and max statistics were pooled per graph, encoded by a separate MLP, and added as a bounded graph-level residual:

```python
node_graph = baseline_node_encoder(graph)
temporal_graph = temporal_encoder(mean_and_max_edge_time_features(graph))
z_graph = node_graph + alpha * tanh(temporal_graph)
```

Measured with `alpha=0.1`, no warm-up, and the standard five seeds:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std |
|---|---:|---:|---:|
| Baseline | 84.38% ± 11.15% | 95.30% | 92.42% ± 4.63% |
| Graph-level temporal residual | 58.77% ± 23.65% | 80.17% | 80.41% ± 10.56% |

This experiment was unsuccessful and exposed severe seed instability. The fixed-center SVDD objective still allows the temporal branch to distort the representation after pooling. The next step should be a controlled experiment with the temporal branch frozen or detached, followed by joint training with an explicit representation-preservation regularizer.

## Current artifacts

- `prepare_bgl_dataset.py`: dataset regeneration and feature scaling
- `TempDIGCNConv.py`: temporal edge encoder and message modulation
- `run_bgl_baseline_csv.py`: multi-seed benchmark runner
- `bgl_processed_graphs_temp.pt`: regenerated temporal dataset
- `bgl_multiseed_results_temp.csv`: current five-seed temporal results
- `RESEARCH_REPORT.md`: consolidated research report

## Paper-aligned data baseline

The official package was added under `Data/` and contains the structured BGL
templates and official embedding dictionary. A new artifact was built with:

- `EventTemplate` as the graph node identity;
- official 200-dimensional GloVe/TF-IDF embeddings;
- raw transition counts as baseline edge weights;
- temporal mean/variance retained only as auxiliary edge channels.

The resulting graphs contain a mean of 10.09 nodes per graph, matching the
paper's reported scale and correcting the previous raw-message construction
(25.60 nodes per graph).

The current runner, using the paper-aligned data, 300 hidden dimensions, five
seeds, 100 epochs, and no warm-up, produced:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std |
|---|---:|---:|---:|
| Baseline on paper-aligned data | 79.41% ± 1.72% | 81.28% | 87.72% ± 0.77% |

This is not yet a faithful reproduction of the paper's 0.93 BGL result because
the runner still uses the simplified one-hop `DIGCNConv` path and Adam. The
paper uses normalized directed DiGCN/PPR preprocessing, its inception path,
and the official training implementation. The canonical data mismatch is now
fixed; reproducing the paper's convolution and optimizer is the next baseline
task before evaluating temporal novelty.

## Official DiGCN baseline reproduction

The repository's original `DiGCN` model and training behavior were then used
with the paper-aligned graphs. The run used PPR-normalized directed adjacency
(`alpha=0.1`), official 200-d embeddings, hidden dimension 300, SGD with
learning rate `0.01`, weight decay `1e-4`, 100 epochs, batch size 128, and the
five Protocol A seeds.

| Metric | Result |
|---|---:|
| ROC-AUC mean ± std | **91.91% ± 5.31%** |
| ROC-AUC peak | **96.11%** |
| PRC-AUC mean ± std | **95.62% ± 2.67%** |
| PRC-AUC peak | **97.42%** |

This reproduces the paper's performance range much more closely than the
simplified runner. The remaining variance is seed-related; the next novelty
experiment must add temporal features to this exact normalized DiGCN baseline,
not to the earlier simplified convolution.

## Temporal extension on the exact paper baseline

The first faithful temporal extension now uses the official normalized DiGCN
representation as its node path. Temporal features are derived from the same
canonical graph edges, summarized per graph using mean/max/std of log-scaled
inter-event mean and variance, standardized using normal training graphs only,
and added through a bounded graph-level residual.

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std | PRC-AUC peak |
|---|---:|---:|---:|---:|
| Exact paper DiGCN baseline | **91.91% ± 5.31%** | **96.11%** | **95.62% ± 2.67%** | **97.42%** |
| Exact DiGCN + temporal residual | 86.62% ± 7.44% | 95.60% | 92.93% ± 3.75% | 97.20% |

This is a valid same-data temporal ablation, but it does not yet improve the
baseline mean. The next temporal work should focus on temporal feature
selection and fusion strength while preserving the exact DiGCN path.

## Separate high-resolution temporal branch

The baseline artifact and runner were left unchanged. A separate temporal
feature artifact was created from the structured BGL `Time` field rather than
the coarse integer `Timestamp`. It contains six graph-level features derived
from inter-event deltas: mean, median, 95th percentile, standard deviation,
maximum, and zero-gap fraction. Standardization uses normal training graphs
only.

The separate temporal runner reused the exact normalized DiGCN path and added
only a bounded temporal residual:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std |
|---|---:|---:|---:|
| Exact baseline, unchanged | 91.91% ± 5.31% | 96.11% | 95.62% ± 2.67% |
| Separate high-resolution temporal branch | 85.95% ± 8.94% | 95.81% | 92.92% ± 3.93% |

This protects the baseline comparison, but high-resolution time features alone
do not improve the mean score. The next temporal experiment should freeze the
DiGCN encoder, train only the temporal head, and only then attempt joint
fine-tuning.

## Staged temporal training result

The temporal runner was updated to follow the staged procedure while leaving
the baseline runner unchanged:

1. Compute the SVDD center using the exact DiGCN path.
2. Freeze the DiGCN encoder for 20 epochs and train only the temporal head at
   learning rate `0.001`.
3. Unfreeze the DiGCN encoder and jointly fine-tune it at learning rate
   `0.0001`, while retaining `0.001` for the temporal head.

Results on the same five seeds:

| Model | ROC-AUC mean ± std | ROC-AUC peak | PRC-AUC mean ± std | PRC-AUC peak |
|---|---:|---:|---:|---:|
| Exact baseline, unchanged | 91.91% ± 5.31% | 96.11% | 95.62% ± 2.67% | 97.42% |
| Staged temporal extension | **92.03% ± 0.43%** | 92.54% | **95.51% ± 0.22%** | 95.90% |

The temporal extension is now slightly above the baseline mean and has much
lower seed variance. The improvement is modest, so the next research step is
to tune the temporal feature set and freeze/joint-training schedule using the
validation split rather than increasing model complexity blindly.

## Validation-driven temporal ablation

The ablation runner `run_bgl_temporal_ablation.py` tested four temporal
configurations. Configuration selection used validation ROC-AUC only; the test
split was reserved for the final comparison.

| Configuration | Validation ROC-AUC mean ± std | Test ROC-AUC mean ± std | Test PRC-AUC mean ± std |
|---|---:|---:|---:|
| All 6 features, alpha .10, freeze 20 | 91.62% ± 0.52% | 92.03% ± 0.48% | 95.51% ± 0.25% |
| No zero-gap, alpha .10, freeze 20 | 91.62% ± 0.44% | 92.03% ± 0.46% | 95.50% ± 0.27% |
| Core features, alpha .05, freeze 20 | 91.66% ± 0.40% | 92.03% ± 0.37% | 95.56% ± 0.22% |
| No zero-gap, alpha .05, freeze 10 | **91.79% ± 0.65%** | **92.06% ± 0.53%** | **95.57% ± 0.29%** |

Selecting the best configuration per seed from validation produced **92.15% ±
0.43% ROC-AUC** and **95.60% ± 0.24% PRC-AUC** on the held-out test values.
The selected configurations were the no-zero-gap, alpha-.05, 10-epoch-freeze
variant for three seeds, the core-feature variant for one seed, and the full
six-feature variant for one seed.

### What the ablation shows

- Staged training is the most reliable improvement: it prevents the temporal
  branch from destabilizing the exact paper encoder.
- Lower residual strength (`alpha=.05`) is slightly better than `.10`.
- A 10-epoch freeze is slightly better than a 20-epoch freeze in the current
  validation study.
- The zero-gap feature adds little and is removed in the best configuration.
- The gain over the unchanged baseline mean is modest (about 0.24 percentage
  points), but the temporal model reduces seed variance substantially. This is
  a stability improvement, not evidence of a large performance breakthrough.

The next result to report should use one configuration fixed from validation,
preferably `no_zero_gap_a005_f10`, and run it once on the test split without
further test-driven tuning. The baseline remains unchanged in
`run_bgl_paper_exact.py`; all temporal changes are isolated in the temporal
runner and ablation script.

## Edge-level temporal fusion experiment

The next experiment moved temporal information from a graph-level residual into
the message-passing path. `run_bgl_edge_temporal.py` retains the exact PPR
normalized DiGCN adjacency and attaches log-scaled mean/variance time features
to normalized edges that correspond to canonical BGL transitions. PPR edges
without a direct canonical transition receive neutral temporal features.

The baseline node projection is frozen for the first 10 SVDD epochs while the
temporal edge encoders learn. The node path is then jointly fine-tuned with a
learning rate of `0.0001`, while the temporal encoders use `0.001` and the
temporal residual starts at `alpha=0.05`.

| Model | Test ROC-AUC mean ± std | Test ROC-AUC peak | Test PRC-AUC mean ± std |
|---|---:|---:|---:|
| Exact baseline, unchanged | 91.91% ± 5.31% | 96.11% | 95.62% ± 2.67% |
| Graph-level temporal residual | 92.06% ± 0.53% | 92.56% | 95.57% ± 0.29% |
| **Edge-level temporal fusion** | **92.34% ± 0.36%** | **92.86%** | 95.04% ± 0.27% |

Edge-level fusion is the strongest ROC-AUC result so far: it improves the
baseline mean by approximately 0.43 percentage points and improves the
graph-level temporal branch by approximately 0.28 points, while retaining low
seed variance. Its PRC-AUC is lower, so the method improves ROC ranking but
does not improve precision-recall ranking yet. This trade-off must be included
in the paper rather than hidden.
