#!/usr/bin/env python3
"""
run_bgl_baseline_csv.py

Trains and evaluates:
1. Optimized Baseline OCDiGCN (L=1, 100 epochs, 300-d node embeddings, raw transition counts)
2. Novel Temp-OCDiGCN (L=1, 100 epochs, 3D Temporal Edge Encoding)
"""

import os
import sys
import argparse
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.metrics import roc_auc_score, average_precision_score

from DIGCNConv import DIGCNConv
from TempDIGCNConv import TempDIGCNConv

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class OptimizedBaselineOCDiGCN(nn.Module):
    """
    Optimized Baseline OCDiGCN with L=1 directed layer and 300-d GloVe+TF-IDF features.
    """
    def __init__(self, in_dim=300, hidden_dim=128):
        super(OptimizedBaselineOCDiGCN, self).__init__()
        self.conv1 = DIGCNConv(in_channels=in_dim, out_channels=hidden_dim)
        # Bias-free linear projection layer
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if data.edge_attr is not None and data.edge_attr.dim() > 1:
            # The dataset stores log1p(Y_ij); keep the scaled transition weight.
            edge_weight = data.edge_attr[:, 0]
        else:
            edge_weight = data.edge_attr

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        z_graph = global_mean_pool(x, data.batch)
        z_graph = self.proj(z_graph)
        return z_graph


class NovelTempOCDiGCN(nn.Module):
    """
    Graph-level temporal fusion model.

    The node path is the validated baseline directed convolution using the
    transition-count channel. Temporal edge features are pooled per graph and
    fused after message passing, so temporal context cannot suppress semantic
    node messages edge-by-edge.
    """
    def __init__(self, in_dim=300, hidden_dim=128, edge_dim=3,
                 temporal_alpha_init=0.1):
        super(NovelTempOCDiGCN, self).__init__()
        self.conv1 = DIGCNConv(in_channels=in_dim, out_channels=hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Mean and max of the two standardized temporal channels.
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2 * (edge_dim - 1), hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.temporal_alpha = nn.Parameter(torch.tensor(float(temporal_alpha_init)))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        count_weight = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr

        x = self.conv1(x, edge_index, count_weight)
        x = F.relu(x)

        node_graph = self.node_proj(global_mean_pool(x, data.batch))

        # Map each source node back to its graph to pool edge features.
        edge_batch = data.batch[edge_index[0]]
        temporal_edge_attr = edge_attr[:, 1:]
        temporal_mean = global_mean_pool(temporal_edge_attr, edge_batch)
        temporal_max = global_max_pool(temporal_edge_attr, edge_batch)
        temporal_graph = self.temporal_encoder(
            torch.cat([temporal_mean, temporal_max], dim=1)
        )
        temporal_graph = self.temporal_alpha * torch.tanh(temporal_graph)

        # True graph-level residual: alpha=0 exactly preserves the node-path
        # representation, while temporal context is added in a bounded way.
        z_graph = node_graph + temporal_graph
        return z_graph


def init_hypersphere_center(model, train_loader, device):
    """
    Computes pure initial centroid o on normal training representations.
    """
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            z = model(batch)
            all_z.append(z)
            
    center = torch.mean(torch.cat(all_z, dim=0), dim=0).detach()
    return center


def train_model(model, train_loader, epochs, lr, weight_decay, device,
                model_name="Model", verbose=False, warmup_epochs=5):
    model = model.to(device)
    center = init_hypersphere_center(model, train_loader, device)
    center.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def optimize_epochs(start_epoch, end_epoch, current_center):
        model.train()
        for epoch in range(start_epoch, end_epoch + 1):
            total_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                z_graph = model(batch)
                dist_sq = torch.sum((z_graph - current_center) ** 2, dim=1)
                loss = torch.mean(dist_sq)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == warmup_epochs):
                print(f"[{model_name}] Epoch {epoch:03d}/{epochs:03d} | SVDD Loss: {avg_loss:.6f}")

    warmup_epochs = min(max(int(warmup_epochs), 0), max(epochs - 1, 0))
    if warmup_epochs:
        optimize_epochs(1, warmup_epochs, center)
        # Recalculate and freeze the hypersphere center from the warmed model.
        center = init_hypersphere_center(model, train_loader, device)
        center.requires_grad = False
        print(f"[{model_name}] Recomputed SVDD center after {warmup_epochs}-epoch warm-up") if verbose else None
    if warmup_epochs < epochs:
        optimize_epochs(warmup_epochs + 1, epochs, center)

    return model, center


def evaluate_model(model, center, test_loader, device):
    model.eval()
    y_true = []
    anomaly_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            z_graph = model(batch)
            
            scores = torch.norm(z_graph - center, p=2, dim=1)
            anomaly_scores.extend(scores.cpu().numpy().tolist())
            y_true.extend(batch.y.cpu().numpy().tolist())

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    prc_auc = average_precision_score(y_true, anomaly_scores)
    return roc_auc, prc_auc


def run_experiments(pt_path, splits_dir, seeds=None, epochs=100, lr=0.01,
                    weight_decay=1e-4, batch_size=128, output_csv=None,
                    warmup_epochs=5, temporal_alpha_init=0.1,
                    hidden_dim=300):
    if seeds is None:
        seeds = [42, 100, 2024, 777, 1213]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    print(f"Loading dataset dictionary from {pt_path}...")
    graphs_dict = torch.load(pt_path, weights_only=False)
    print(f"Loaded {len(graphs_dict)} graph Data objects.")

    train_csv = pd.read_csv(os.path.join(splits_dir, "train_graph_ids.csv"))
    val_csv = pd.read_csv(os.path.join(splits_dir, "validation_graph_ids.csv"))
    test_csv = pd.read_csv(os.path.join(splits_dir, "test_graph_ids.csv"))

    train_nodes = train_csv['node'].tolist()
    val_nodes = val_csv['node'].tolist()
    test_nodes = test_csv['node'].tolist()

    train_graphs = [graphs_dict[n] for n in train_nodes if n in graphs_dict]
    val_graphs = [graphs_dict[n] for n in val_nodes if n in graphs_dict]
    test_graphs = [graphs_dict[n] for n in test_nodes if n in graphs_dict]

    train_graphs_normal = [g for g in train_graphs if int(g.y.item()) == 0]

    input_dim = int(train_graphs_normal[0].x.size(-1))
    print(f"Normal Training Graphs: {len(train_graphs_normal)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    print(f"Model dimensions: input={input_dim}, hidden={hidden_dim}")

    base_results = []
    temp_results = []

    print(f"\nEvaluating models across {len(seeds)} random seeds: {seeds} (lr={lr}, weight_decay={weight_decay})")

    for seed in seeds:
        print(f"\n>>> RUNNING BENCHMARK WITH SEED: {seed} <<<")
        set_seed(seed)

        train_loader = DataLoader(train_graphs_normal, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        # 1. Baseline OCDiGCN
        base_model = OptimizedBaselineOCDiGCN(in_dim=input_dim, hidden_dim=hidden_dim)
        base_model, base_center = train_model(base_model, train_loader, epochs, lr, weight_decay, device, f"Baseline (seed={seed})", verbose=False, warmup_epochs=warmup_epochs)
        base_roc, base_prc = evaluate_model(base_model, base_center, test_loader, device)
        base_results.append((seed, base_roc, base_prc))

        # 2. Temp-OCDiGCN
        set_seed(seed)
        temp_model = NovelTempOCDiGCN(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=3,
            temporal_alpha_init=temporal_alpha_init,
        )
        temp_model, temp_center = train_model(temp_model, train_loader, epochs, lr, weight_decay, device, f"Temp-OCDiGCN (seed={seed})", verbose=False, warmup_epochs=warmup_epochs)
        temp_roc, temp_prc = evaluate_model(temp_model, temp_center, test_loader, device)
        temp_results.append((seed, temp_roc, temp_prc))

        print(f"Seed {seed:4d} | Baseline: ROC={base_roc*100:6.2f}%, PRC={base_prc*100:6.2f}% | Temp-OCDiGCN: ROC={temp_roc*100:6.2f}%, PRC={temp_prc*100:6.2f}%")

    base_rocs = [r[1] for r in base_results]
    base_prcs = [r[2] for r in base_results]
    temp_rocs = [r[1] for r in temp_results]
    temp_prcs = [r[2] for r in temp_results]

    print("\n" + "=" * 75)
    print("      MULTI-SEED EXPERIMENTAL BENCHMARK RESULTS (PROTOCOL A)")
    print("=" * 75)
    print(f"{'Seed':<8} | {'Baseline ROC':<14} | {'Baseline PRC':<14} | {'Temp-OCDiGCN ROC':<16} | {'Temp-OCDiGCN PRC':<16}")
    print("-" * 75)
    for i in range(len(seeds)):
        s = seeds[i]
        b_r, b_p = base_rocs[i], base_prcs[i]
        t_r, t_p = temp_rocs[i], temp_prcs[i]
        print(f"{s:<8} | {b_r*100:>6.2f}% ({b_r:.4f}) | {b_p*100:>6.2f}% ({b_p:.4f}) | {t_r*100:>8.2f}% ({t_r:.4f}) | {t_p*100:>8.2f}% ({t_p:.4f})")
    print("-" * 75)

    print(f"{'Mean ± Std':<8} | {np.mean(base_rocs)*100:>5.2f}% ± {np.std(base_rocs)*100:.2f}% | {np.mean(base_prcs)*100:>5.2f}% ± {np.std(base_prcs)*100:.2f}% | {np.mean(temp_rocs)*100:>7.2f}% ± {np.std(temp_rocs)*100:.2f}% | {np.mean(temp_prcs)*100:>7.2f}% ± {np.std(temp_prcs)*100:.2f}%")
    print(f"{'Peak Score':<8} | {np.max(base_rocs)*100:>6.2f}% ({np.max(base_rocs):.4f}) | {np.max(base_prcs)*100:>6.2f}% ({np.max(base_prcs):.4f}) | {np.max(temp_rocs)*100:>8.2f}% ({np.max(temp_rocs):.4f}) | {np.max(temp_prcs)*100:>8.2f}% ({np.max(temp_prcs):.4f})")
    print("=" * 75)

    results = [{
        "seed": seed,
        "baseline_roc_auc": base_results[i][1],
        "baseline_prc_auc": base_results[i][2],
        "temp_roc_auc": temp_results[i][1],
        "temp_prc_auc": temp_results[i][2],
    } for i, seed in enumerate(seeds)]
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Per-seed results written to: {output_csv}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Protocol A BGL OCDiGCN benchmarks.")
    parser.add_argument("--dataset", default="bgl_processed_graphs.pt")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 777, 1213])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--temporal-alpha", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=300)
    args = parser.parse_args()
    run_experiments(args.dataset, args.splits_dir, seeds=args.seeds,
                    epochs=args.epochs, lr=args.lr,
                    weight_decay=args.weight_decay, batch_size=args.batch_size,
                    output_csv=args.output_csv, warmup_epochs=args.warmup_epochs,
                    temporal_alpha_init=args.temporal_alpha, hidden_dim=args.hidden_dim)
