#!/usr/bin/env python3
"""Run the temporal extension separately from the unchanged paper baseline."""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from run_bgl_paper_exact import run_seed, standardize_temporal_summaries


def main(args):
    graphs = torch.load(args.dataset, weights_only=False)
    temporal = torch.load(args.temporal_features, weights_only=False)
    missing = [node for node in graphs if node not in temporal]
    if missing:
        raise RuntimeError(f"Missing temporal features for {len(missing)} graphs")
    for node, graph in graphs.items():
        graph.temporal_summary = temporal[node]
    graphs = standardize_temporal_summaries(graphs, args.splits_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in args.seeds:
        roc, prc = run_seed(
            graphs, args.splits_dir, seed, args.epochs, args.batch_size,
            args.hidden_dim, args.lr, args.weight_decay, device,
            temporal=True, temporal_alpha=args.temporal_alpha,
            temporal_freeze_epochs=args.freeze_epochs,
            temporal_lr=args.temporal_lr,
            joint_lr=args.joint_lr,
        )
        results.append((roc, prc))
        print(f"Seed {seed}: ROC-AUC={roc:.4f}, PRC-AUC={prc:.4f}")
    values = np.asarray(results)
    print(
        f"Mean ± std: ROC-AUC={values[:,0].mean():.4f} ± {values[:,0].std():.4f}, "
        f"PRC-AUC={values[:,1].mean():.4f} ± {values[:,1].std():.4f}"
    )
    print(f"Peak: ROC-AUC={values[:,0].max():.4f}, PRC-AUC={values[:,1].max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bgl_paper_digcn_graphs.pt")
    parser.add_argument("--temporal-features", dest="temporal_features", default="bgl_temporal_features.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-alpha", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--temporal-lr", type=float, default=0.001)
    parser.add_argument("--joint-lr", type=float, default=0.0001)
    main(parser.parse_args())
