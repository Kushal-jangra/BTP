# GitHub Handoff: Paper Baseline and Temporal Extension

This document records the reproducible implementation state after the locked
five-seed temporal benchmark.

## Final locked experiment

The selected temporal configuration was fixed from the validation ablation and
then evaluated once across Protocol A seeds `[42, 100, 2024, 777, 1213]`:

- canonical paper-aligned BGL graphs;
- official 200-dimensional BGL template embeddings;
- exact directed DiGCN/PPR baseline path;
- temporal features: mean, median, p95, standard deviation, and maximum of
  log-scaled inter-event time deltas;
- zero-gap feature removed;
- residual strength `alpha=0.05`;
- temporal-only warm-up for 10 epochs;
- joint fine-tuning with temporal learning rate `0.001` and base learning rate
  `0.0001`;
- 100 epochs, hidden dimension 300, batch size 128, weight decay `1e-4`.

Final result:

| Metric | Mean ± std | Peak |
|---|---:|---:|
| Test ROC-AUC | **92.06% ± 0.53%** | **92.56%** |
| Test PRC-AUC | **95.57% ± 0.29%** | **95.96%** |

The unchanged exact baseline remains **91.91% ± 5.31% ROC-AUC**. The temporal
model therefore improves the mean modestly and substantially reduces seed
variance. This should be described as a stability improvement, not as a claim
of a large accuracy jump.

## Edge-level temporal experiment

The stronger experimental branch is implemented in
`run_bgl_edge_temporal.py`. It attaches temporal edge attributes to the exact
PPR-normalized adjacency and injects them during message passing. Its current
five-seed result is **92.34% ± 0.36% ROC-AUC**, with **95.04% ± 0.27% PRC-AUC**.
This branch is the current ROC-AUC candidate, but its lower PRC-AUC means it
should remain documented as an experiment until the precision-recall trade-off
is investigated.

## Files to push to GitHub

### Core implementation

- `run_bgl_paper_exact.py` — exact paper-aligned baseline and shared temporal
  model implementation.
- `run_bgl_temporal_final.py` — locked final temporal benchmark.
- `run_bgl_temporal_ablation.py` — validation-driven ablation study.
- `run_bgl_temporal_exact.py` — general temporal benchmark runner.
- `run_bgl_edge_temporal.py` — edge-level temporal message-passing experiment.
- `prepare_bgl_paper_dataset.py` — canonical paper-aligned graph creation.
- `prepare_bgl_temporal_features.py` — temporal feature creation.
- `DIGCNConv.py` — baseline DiGCN implementation used by the project.
- `TempDIGCNConv.py` — temporal convolution implementation, if retained as part
  of the documented experimental branch.

### Documentation and results

- `README.md` — update with the commands and final metrics if needed.
- `TEMP_OCDIGCN_EXECUTION.md` — chronological implementation and ablation
  record.
- `GITHUB_HANDOFF.md` — this reproducibility and push checklist.
- `bgl_temporal_final_results.csv` — final five-seed result table.
- `bgl_edge_temporal_results.csv` — edge-level temporal five-seed result table.
- `bgl_temporal_ablation_results.csv` — validation ablation results.
- `bgl_paper_baseline_results.csv` — exact baseline result table.
- `RESEARCH_REPORT.md` — push only after checking that its reported metrics
  match the final artifacts.

## Files that should normally not be pushed

These are generated, large, machine-specific, or redundant:

- `*.pt` generated graph and feature tensors, including
  `bgl_paper_digcn_graphs.pt`, `bgl_paper_graphs.pt`, and
  `bgl_temporal_features.pt`;
- `__pycache__/` and `.DS_Store`;
- old exploratory CSVs such as `bgl_multiseed_results_*.csv` unless they are
  explicitly needed for the paper's ablation appendix;
- local virtual environments such as `.venv/`;
- downloaded or locally generated split/data directories if the repository
  already provides a reproducible download/preparation procedure;
- the local PDF copy of the base paper unless redistribution is permitted.

The official BGL data and embedding files under `Data/` should not be pushed
without checking the source repository's license and redistribution terms. If
they cannot be redistributed, push a download/preparation instruction instead.

## Reproduction commands

```bash
# Build the canonical paper-aligned graph artifact
./.venv/bin/python prepare_bgl_paper_dataset.py

# Build temporal features
./.venv/bin/python prepare_bgl_temporal_features.py

# Run the unchanged baseline
./.venv/bin/python run_bgl_paper_exact.py \
  --dataset bgl_paper_graphs.pt \
  --normalized-dataset bgl_paper_digcn_graphs.pt

# Run the locked temporal benchmark
./.venv/bin/python run_bgl_temporal_final.py
```

The final runner writes `bgl_temporal_final_results.csv`. Dataset tensors are
intentionally treated as local build artifacts unless the project decides to
publish them through a suitable data-release mechanism.

## Before pushing

1. Confirm that the baseline source was not changed by temporal experiments.
2. Confirm that the final CSV contains all five seeds.
3. Remove generated tensors, caches, and unrelated exploratory outputs from the
   commit.
4. Check data and paper-PDF licensing.
5. Ensure the README points to the canonical preparation and benchmark commands.
