#!/usr/bin/env python3
"""Experimental edge-level temporal fusion on the exact paper graph path."""

import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from TempDIGCNConv import TempDIGCNConv
from run_bgl_paper_exact import get_appr_directed_adj


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EdgeTemporalDiGCN(nn.Module):
    def __init__(self, nfeat, hidden_dim, alpha=0.05):
        super().__init__()
        self.conv1 = TempDIGCNConv(nfeat, hidden_dim, edge_dim=3,
                                   temporal_alpha_init=alpha, bias=False)
        self.conv2 = TempDIGCNConv(hidden_dim, hidden_dim, edge_dim=3,
                                   temporal_alpha_init=alpha, bias=False)

    def forward(self, batch):
        x = torch.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        return [x[batch.batch == i] for i in range(batch.num_graphs)]

    def graph_embeddings(self, batch):
        return torch.stack([value.mean(dim=0) for value in self(batch)])


def build_graphs(raw_graphs, splits_dir):
    train_ids = pd.read_csv(os.path.join(splits_dir, "train_graph_ids.csv"))["node"].tolist()
    normal_ids = [n for n in train_ids if n in raw_graphs and int(raw_graphs[n].y.item()) == 0]
    temporal_fit = torch.cat([
        torch.log1p(raw_graphs[n].edge_attr[:, 1:3].clamp_min(0))
        for n in normal_ids
    ])
    mean = temporal_fit.mean(0)
    std = temporal_fit.std(0, unbiased=False).clamp_min(1e-6)

    output = {}
    for node, graph in raw_graphs.items():
        edge_index, norm = get_appr_directed_adj(
            alpha=0.1, edge_index=graph.edge_index,
            num_nodes=graph.num_nodes, edge_weight=graph.edge_attr[:, 0],
        )
        lookup = {
            tuple(edge.tolist()): value
            for edge, value in zip(graph.edge_index.t(), graph.edge_attr[:, 1:3])
        }
        temporal = []
        for edge in edge_index.t():
            value = lookup.get(tuple(edge.tolist()))
            if value is None:
                value = torch.zeros(2)
            else:
                value = (torch.log1p(value.clamp_min(0)) - mean) / std
            temporal.append(value)
        temporal = torch.stack(temporal)
        output[node] = Data(
            x=graph.x, edge_index=edge_index,
            edge_attr=torch.cat([norm.view(-1, 1), temporal], dim=1),
            y=graph.y, node_id=node,
        )
    return output


def score(model, loader, device, center):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.graph_embeddings(batch)
            scores.extend(torch.linalg.vector_norm(z - center, dim=1).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def run_seed(graphs, splits_dir, seed, args, device):
    seed_all(seed)
    ids = {
        name: pd.read_csv(os.path.join(splits_dir, f"{name}_graph_ids.csv"))["node"].tolist()
        for name in ("train", "validation", "test")
    }
    train = [graphs[n] for n in ids["train"] if n in graphs and int(graphs[n].y.item()) == 0]
    validation = [graphs[n] for n in ids["validation"] if n in graphs]
    test = [graphs[n] for n in ids["test"] if n in graphs]
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    model = EdgeTemporalDiGCN(train[0].x.size(1), args.hidden_dim, args.alpha).to(device)
    center = None
    optimizer = None
    for epoch in range(args.epochs + 1):
        if center is not None and epoch == 1:
            for name, parameter in model.named_parameters():
                if name in {"conv1.weight", "conv2.weight"}:
                    parameter.requires_grad = False
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.temporal_lr, weight_decay=args.weight_decay,
            )
        elif center is not None and epoch == args.freeze_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.SGD([
                {"params": [model.conv1.weight, model.conv2.weight], "lr": args.joint_lr},
                {"params": [p for n, p in model.named_parameters() if n not in {"conv1.weight", "conv2.weight"}], "lr": args.temporal_lr},
            ], weight_decay=args.weight_decay)

        model.train()
        collected = []
        for batch in train_loader:
            batch = batch.to(device)
            z = model.graph_embeddings(batch)
            if center is None:
                collected.append(z.detach())
            else:
                loss = ((z - center) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if center is None:
            center = torch.cat(collected).mean(0).detach()

    val = score(model, val_loader, device, center)
    test = score(model, test_loader, device, center)
    return val[0], val[1], test[0], test[1]


def main(args):
    raw = torch.load(args.dataset, weights_only=False)
    graphs = build_graphs(raw, args.splits_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        values = run_seed(graphs, args.splits_dir, seed, args, device)
        row = dict(zip(("val_roc", "val_prc", "test_roc", "test_prc"), values))
        row.update(seed=seed, model="edge_temporal_a005_f10")
        rows.append(row)
        print(row)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for key in ("val_roc", "test_roc", "test_prc"):
        values = np.array([row[key] for row in rows])
        print(key, "mean", values.mean(), "std", values.std(ddof=1), "peak", values.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bgl_paper_graphs.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--temporal-lr", type=float, default=0.001)
    parser.add_argument("--joint-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-csv", default="bgl_edge_temporal_results.csv")
    main(parser.parse_args())
