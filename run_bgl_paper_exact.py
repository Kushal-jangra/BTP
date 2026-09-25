#!/usr/bin/env python3
"""Run the repository's original OCDiGCN baseline on Protocol A BGL graphs."""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from DataLoader import DiGCN
from scipy.linalg import eig


def get_appr_directed_adj(alpha, edge_index, num_nodes, edge_weight):
    """Equivalent of the repository's PPR directed adjacency helper."""
    edges = edge_index.cpu().numpy()
    weights = edge_weight.detach().cpu().numpy().astype(np.float64)
    self_edges = np.arange(num_nodes)
    edges = np.concatenate([edges, np.vstack([self_edges, self_edges])], axis=1)
    weights = np.concatenate([weights, np.ones(num_nodes)])
    row = edges[0]
    degree = np.zeros(num_nodes)
    np.add.at(degree, row, weights)
    p = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    np.add.at(p, (edges[0], edges[1]), weights / np.maximum(degree[row], 1e-12))
    p_v = np.zeros((num_nodes + 1, num_nodes + 1), dtype=np.float64)
    p_v[:num_nodes, :num_nodes] = (1 - alpha) * p
    p_v[num_nodes, :num_nodes] = 1.0 / num_nodes
    p_v[:num_nodes, num_nodes] = alpha
    values, vectors = eig(p_v.T)
    idx = np.argmax(values.real)
    pi = vectors[:, idx].real[:num_nodes]
    if pi.sum() < 0:
        pi = -pi
    pi = np.maximum(pi / max(pi.sum(), 1e-12), 1e-12)
    pi_sqrt = np.diag(np.sqrt(pi))
    pi_inv_sqrt = np.diag(1.0 / np.sqrt(pi))
    L = (pi_sqrt @ p @ pi_inv_sqrt + pi_inv_sqrt @ p.T @ pi_sqrt) / 2.0
    L[~np.isfinite(L)] = 0
    idx = np.nonzero(L)
    values = L[idx]
    degree = np.zeros(num_nodes)
    np.add.at(degree, idx[0], values)
    norm = values / np.sqrt(np.maximum(degree[idx[0]], 1e-12) * np.maximum(degree[idx[1]], 1e-12))
    return torch.tensor(np.vstack(idx), dtype=torch.long), torch.tensor(norm, dtype=torch.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_graphs(graphs, alpha=0.1):
    normalized = {}
    for node, graph in graphs.items():
        edge_index, edge_attr = get_appr_directed_adj(
            alpha=alpha,
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            edge_weight=graph.edge_attr[:, 0],
        )
        temporal = torch.log1p(graph.edge_attr[:, 1:3].clamp_min(0))
        temporal_summary = torch.cat([
            temporal.mean(dim=0),
            temporal.amax(dim=0),
            temporal.std(dim=0, unbiased=False),
        ]).view(1, -1)
        normalized[node] = Data(
            x=graph.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=graph.y,
            node_id=node,
            temporal_summary=temporal_summary,
        )
    return normalized


class TempPaperDiGCN(nn.Module):
    """Official DiGCN plus a bounded graph-level temporal residual."""

    def __init__(self, nfeat, hidden_dim, temporal_dim=6, alpha=0.1):
        super().__init__()
        self.base = DiGCN(nfeat=nfeat, nhid=hidden_dim, nlayer=1, bias=False)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def graph_embeddings(self, batch):
        node_embeddings = self.base(batch)
        node_graph = torch.stack([embedding.mean(dim=0) for embedding in node_embeddings])
        temporal = self.temporal_encoder(batch.temporal_summary)
        return node_graph + self.alpha * torch.tanh(temporal)


def graph_embeddings(model, loader, device):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if hasattr(model, "graph_embeddings"):
                output.append(model.graph_embeddings(batch))
            else:
                output.extend([emb.mean(dim=0) for emb in model(batch)])
    return torch.cat(output, dim=0)


def standardize_temporal_summaries(graphs, splits_dir):
    train_ids = pd.read_csv(os.path.join(splits_dir, "train_graph_ids.csv"))["node"].tolist()
    train = torch.cat([graphs[n].temporal_summary for n in train_ids if n in graphs], dim=0)
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    for graph in graphs.values():
        graph.temporal_summary = (graph.temporal_summary - mean) / std
    return graphs


def run_seed(graphs, splits_dir, seed, epochs, batch_size, hidden_dim, lr,
             weight_decay, device, temporal=False, temporal_alpha=0.1,
             temporal_freeze_epochs=20, temporal_lr=0.001, joint_lr=0.0001):
    set_seed(seed)
    train_ids = pd.read_csv(os.path.join(splits_dir, "train_graph_ids.csv"))["node"].tolist()
    test_ids = pd.read_csv(os.path.join(splits_dir, "test_graph_ids.csv"))["node"].tolist()
    train = [graphs[n] for n in train_ids if n in graphs]
    test = [graphs[n] for n in test_ids if n in graphs]
    train = [g for g in train if int(g.y.item()) == 0]

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    if temporal:
        model = TempPaperDiGCN(
            nfeat=train[0].x.size(1), hidden_dim=hidden_dim,
            temporal_dim=train[0].temporal_summary.size(1), alpha=temporal_alpha,
        ).to(device)
    else:
        model = DiGCN(nfeat=train[0].x.size(1), nhid=hidden_dim, nlayer=1, bias=False).to(device)
    optimizer = None

    # Match the original repository's MeanTrainer behavior: compute the
    # center during the first epoch, then optimize distances for later epochs.
    center = None
    for epoch in range(epochs + 1):
        if temporal and center is not None and epoch == 1:
            for parameter in model.base.parameters():
                parameter.requires_grad = False
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=temporal_lr, weight_decay=weight_decay,
            )
        elif temporal and center is not None and epoch == temporal_freeze_epochs + 1:
            for parameter in model.base.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.SGD([
                {"params": model.base.parameters(), "lr": joint_lr},
                {"params": [p for name, p in model.named_parameters() if not name.startswith("base.")], "lr": temporal_lr},
            ], weight_decay=weight_decay)
        elif not temporal and center is not None and optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        all_train = []
        for batch in train_loader:
            batch = batch.to(device)
            if hasattr(model, "graph_embeddings"):
                graph_embeddings_batch = model.graph_embeddings(batch)
            else:
                embeddings = model(batch)
                graph_embeddings_batch = torch.stack([e.mean(dim=0) for e in embeddings])
            if center is None:
                all_train.append(graph_embeddings_batch.detach())
            else:
                loss = ((graph_embeddings_batch - center) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if center is None:
            center = torch.cat(all_train).mean(dim=0).detach()

    test_embeddings = graph_embeddings(model, test_loader, device)
    scores = torch.linalg.vector_norm(test_embeddings - center, dim=1).cpu().numpy()
    labels = np.concatenate([g.y.cpu().numpy() for g in test])
    from sklearn.metrics import average_precision_score, roc_auc_score
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bgl_paper_graphs.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--normalized-dataset", default="bgl_paper_digcn_graphs.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--temporal-alpha", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--temporal-lr", type=float, default=0.001)
    parser.add_argument("--joint-lr", type=float, default=0.0001)
    args = parser.parse_args()

    if os.path.exists(args.normalized_dataset) and not args.temporal:
        graphs = torch.load(args.normalized_dataset, weights_only=False)
    else:
        graphs = torch.load(args.dataset, weights_only=False)
        graphs = normalize_graphs(graphs)
        torch.save(graphs, args.normalized_dataset)
        print(f"Saved normalized DiGCN graphs to {args.normalized_dataset}")

    if args.temporal:
        graphs = standardize_temporal_summaries(graphs, args.splits_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in args.seeds:
        roc, prc = run_seed(
            graphs, args.splits_dir, seed, args.epochs, args.batch_size,
            args.hidden_dim, args.lr, args.weight_decay, device,
            temporal=args.temporal, temporal_alpha=args.temporal_alpha,
            temporal_freeze_epochs=args.freeze_epochs,
            temporal_lr=args.temporal_lr,
            joint_lr=args.joint_lr,
        )
        results.append((roc, prc))
        print(f"Seed {seed}: ROC-AUC={roc:.4f}, PRC-AUC={prc:.4f}")
    rocs, prcs = np.array(results).T
    print(f"Mean ± std: ROC-AUC={rocs.mean():.4f} ± {rocs.std():.4f}, PRC-AUC={prcs.mean():.4f} ± {prcs.std():.4f}")
    print(f"Peak: ROC-AUC={rocs.max():.4f}, PRC-AUC={prcs.max():.4f}")
