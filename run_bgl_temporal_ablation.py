#!/usr/bin/env python3
"""Validation-driven ablations for the separate exact temporal branch."""

import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from run_bgl_paper_exact import TempPaperDiGCN


FEATURE_NAMES = ["mean", "median", "p95", "std", "max", "zero_gap"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_vectors(model, batch):
    return model.graph_embeddings(batch)


def scores(model, loader, device, center):
    model.eval()
    values, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = graph_vectors(model, batch)
            values.append(torch.linalg.vector_norm(z - center, dim=1).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    values = np.concatenate(values)
    labels = np.concatenate(labels)
    return roc_auc_score(labels, values), average_precision_score(labels, values)


def train_config(graphs, splits_dir, seed, config, epochs, batch_size,
                 hidden_dim, weight_decay, device):
    seed_all(seed)
    split_ids = {
        name: pd.read_csv(os.path.join(splits_dir, f"{name}_graph_ids.csv"))["node"].tolist()
        for name in ("train", "validation", "test")
    }
    train = [graphs[n] for n in split_ids["train"] if n in graphs and int(graphs[n].y.item()) == 0]
    validation = [graphs[n] for n in split_ids["validation"] if n in graphs]
    test = [graphs[n] for n in split_ids["test"] if n in graphs]
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    model = TempPaperDiGCN(
        nfeat=train[0].x.size(1),
        hidden_dim=hidden_dim,
        temporal_dim=len(config["features"]),
        alpha=config["alpha"],
    ).to(device)
    center = None
    optimizer = None
    for epoch in range(epochs + 1):
        if center is not None and epoch == 1:
            for p in model.base.parameters():
                p.requires_grad = False
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=config["temporal_lr"], weight_decay=weight_decay,
            )
        if center is not None and epoch == config["freeze_epochs"] + 1:
            for p in model.base.parameters():
                p.requires_grad = True
            optimizer = torch.optim.SGD([
                {"params": model.base.parameters(), "lr": config["joint_lr"]},
                {"params": [p for n, p in model.named_parameters() if not n.startswith("base.")], "lr": config["temporal_lr"]},
            ], weight_decay=weight_decay)

        model.train()
        collected = []
        for batch in train_loader:
            batch = batch.to(device)
            z = graph_vectors(model, batch)
            if center is None:
                collected.append(z.detach())
            else:
                loss = ((z - center) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if center is None:
            center = torch.cat(collected).mean(dim=0).detach()

    val_roc, val_prc = scores(model, val_loader, device, center)
    test_roc, test_prc = scores(model, test_loader, device, center)
    return val_roc, val_prc, test_roc, test_prc


def main(args):
    graphs = torch.load(args.dataset, weights_only=False)
    features = torch.load(args.temporal_features, weights_only=False)
    for node, graph in graphs.items():
        graph.temporal_summary = features[node]

    train_ids = pd.read_csv(os.path.join(args.splits_dir, "train_graph_ids.csv"))["node"].tolist()
    fit = torch.cat([graphs[n].temporal_summary for n in train_ids if n in graphs and int(graphs[n].y.item()) == 0])
    mean, std = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-6)
    for graph in graphs.values():
        graph.temporal_summary = (graph.temporal_summary - mean) / std
    base_summaries = {node: graph.temporal_summary.clone() for node, graph in graphs.items()}

    configs = [
        {"name": "all6_a010_f20", "features": [0, 1, 2, 3, 4, 5], "alpha": 0.10, "freeze_epochs": 20, "temporal_lr": 0.001, "joint_lr": 0.0001},
        {"name": "no_zero_gap_a010_f20", "features": [0, 1, 2, 3, 4], "alpha": 0.10, "freeze_epochs": 20, "temporal_lr": 0.001, "joint_lr": 0.0001},
        {"name": "core_a005_f20", "features": [0, 1, 2, 4], "alpha": 0.05, "freeze_epochs": 20, "temporal_lr": 0.001, "joint_lr": 0.0001},
        {"name": "no_zero_gap_a005_f10", "features": [0, 1, 2, 3, 4], "alpha": 0.05, "freeze_epochs": 10, "temporal_lr": 0.001, "joint_lr": 0.0001},
    ]
    # Each configuration gets its own feature view without mutating the baseline artifact.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for config in configs:
        for seed in args.seeds:
            view = {n: g for n, g in graphs.items()}
            for graph in view.values():
                graph.temporal_summary = base_summaries[graph.node_id][:, config["features"]]
            vroc, vprc, troc, tprc = train_config(
                view, args.splits_dir, seed, config, args.epochs,
                args.batch_size, args.hidden_dim, args.weight_decay, device,
            )
            row = {"config": config["name"], "seed": seed, "val_roc": vroc, "val_prc": vprc, "test_roc": troc, "test_prc": tprc}
            rows.append(row)
            print(row)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    selected = []
    for seed in args.seeds:
        candidates = [r for r in rows if r["seed"] == seed]
        selected.append(max(candidates, key=lambda r: r["val_roc"]))
    print("Selected by validation:", selected)
    print("Selected test ROC mean/std:", np.mean([r["test_roc"] for r in selected]), np.std([r["test_roc"] for r in selected]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bgl_paper_digcn_graphs.pt")
    parser.add_argument("--temporal-features", dest="temporal_features", default="bgl_temporal_features.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-csv", default="bgl_temporal_ablation_results.csv")
    main(parser.parse_args())
