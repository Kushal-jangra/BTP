#!/usr/bin/env python3
"""Locked final benchmark for the selected temporal configuration."""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch

from run_bgl_temporal_ablation import train_config


def main(args):
    graphs = torch.load(args.dataset, weights_only=False)
    temporal = torch.load(args.temporal_features, weights_only=False)
    for node, graph in graphs.items():
        graph.temporal_summary = temporal[node]

    train_ids = pd.read_csv(os.path.join(args.splits_dir, "train_graph_ids.csv"))["node"].tolist()
    fit = torch.cat([
        graphs[n].temporal_summary for n in train_ids
        if n in graphs and int(graphs[n].y.item()) == 0
    ])
    mean = fit.mean(0)
    std = fit.std(0, unbiased=False).clamp_min(1e-6)
    for graph in graphs.values():
        graph.temporal_summary = (graph.temporal_summary - mean) / std

    # Locked from validation ablations: remove zero-gap, alpha=.05, freeze 10.
    features = [0, 1, 2, 3, 4]
    config = {
        "name": "no_zero_gap_a005_f10",
        "features": features,
        "alpha": 0.05,
        "freeze_epochs": 10,
        "temporal_lr": 0.001,
        "joint_lr": 0.0001,
    }
    for graph in graphs.values():
        graph.temporal_summary = graph.temporal_summary[:, features]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        val_roc, val_prc, test_roc, test_prc = train_config(
            graphs, args.splits_dir, seed, config, args.epochs,
            args.batch_size, args.hidden_dim, args.weight_decay, device,
        )
        row = {
            "config": config["name"], "seed": seed,
            "val_roc": val_roc, "val_prc": val_prc,
            "test_roc": test_roc, "test_prc": test_prc,
        }
        rows.append(row)
        print(row)

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for metric in ("val_roc", "test_roc", "test_prc"):
        values = np.array([row[metric] for row in rows])
        print(f"{metric}: mean={values.mean():.6f}, std={values.std(ddof=1):.6f}, peak={values.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bgl_paper_digcn_graphs.pt")
    parser.add_argument("--temporal-features", default="bgl_temporal_features.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-csv", default="bgl_temporal_final_results.csv")
    main(parser.parse_args())
