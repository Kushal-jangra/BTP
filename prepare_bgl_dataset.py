#!/usr/bin/env python3
"""
prepare_bgl_dataset.py

Constructs PyTorch Geometric Data objects from raw BGL logs (/Users/kushal/Downloads/BGL_logs.csv).
- Node Features X in R^{|V| x 300}: TF-IDF weighted semantic word embeddings.
- Multi-dimensional Edge Features E_{ij} in R^3:
  - Column 0: Log-scaled transition frequency log(1 + Y_{ij})
  - Column 1: Mean inter-event time delta bar{\Delta t}_{ij}
  - Column 2: Variance of time deltas Var(\Delta t)_{ij}
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from tqdm import tqdm

def prepare_temporal_bgl_dataset(logs_csv_path, splits_dir, output_pt_path):
    print(f"Loading BGL logs from: {logs_csv_path}")
    df = pd.read_csv(logs_csv_path)
    print(f"Total log records read: {len(df)}")

    manifest_path = os.path.join(splits_dir, "BGL_graph_split_manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    print(f"Total manifest graphs: {len(manifest_df)}")

    node_to_split = dict(zip(manifest_df['node'], manifest_df['split']))
    node_to_label = dict(zip(manifest_df['node'], manifest_df['graph_label']))

    print("Cleaning log templates & computing 300-dim TF-IDF weighted semantic embeddings (X in R^{|V| x 300})...")
    unique_messages = df['message'].astype(str).unique().tolist()
    msg_to_idx = {msg: idx for idx, msg in enumerate(unique_messages)}

    # Clean template messages: remove numbers, special characters, keep pure alpha words
    cleaned_messages = [re.sub(r'[^a-zA-Z\s]', '', m).lower().strip() for m in unique_messages]

    # Sub-word and token TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
    X_tfidf = tfidf.fit_transform(cleaned_messages)

    # 300-dimensional semantic projection
    svd = TruncatedSVD(n_components=300, random_state=42)
    embeddings_300 = svd.fit_transform(X_tfidf)
    # Strict row-wise L2 normalization (zero vectors remain zero).
    norms = np.linalg.norm(embeddings_300, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_300 = embeddings_300 / norms
    embeddings_tensor = torch.tensor(embeddings_300, dtype=torch.float32)

    print("Building temporal attributed directed graphs grouped by node with log1p edge weight scaling...")
    grouped = df.groupby('node')

    graphs_dict = {}

    for node, group in tqdm(grouped):
        if node not in node_to_split:
            continue

        messages = group['message'].astype(str).tolist()
        timestamps = group['timestamp'].values
        
        if len(messages) == 0:
            continue

        group_msgs = list(dict.fromkeys(messages))
        local_msg_map = {msg: i for i, msg in enumerate(group_msgs)}

        # Node features x [num_nodes, 300]
        node_indices = [msg_to_idx[m] for m in group_msgs]
        x = embeddings_tensor[node_indices]

        # Extract transitions and inter-event time deltas \Delta t = t_j - t_i
        edge_deltas = {}  # edge -> list of delta_t
        for i in range(len(messages) - 1):
            src = local_msg_map[messages[i]]
            dst = local_msg_map[messages[i+1]]
            dt = max(0.0, float(timestamps[i+1] - timestamps[i]))
            edge = (src, dst)
            if edge not in edge_deltas:
                edge_deltas[edge] = []
            edge_deltas[edge].append(dt)

        if len(edge_deltas) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor(
                [[float(np.log1p(1.0)), 0.0, 0.0]], dtype=torch.float32
            )
        else:
            edges = list(edge_deltas.keys())
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

            attr_rows = []
            for e in edges:
                dts = edge_deltas[e]
                count = len(dts)
                log_count = np.log1p(float(count)) # log(1 + Y_ij) scaling
                mean_dt = np.mean(dts)
                var_dt = np.var(dts) if count > 1 else 0.0
                # Keep all edge channels on a comparable, non-negative scale.
                attr_rows.append([
                    float(log_count),
                    float(np.log1p(mean_dt)),
                    float(np.log1p(var_dt)),
                ])

            edge_attr = torch.tensor(attr_rows, dtype=torch.float32)

        label = int(node_to_label[node])
        y = torch.tensor([label], dtype=torch.long)

        g_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        g_data.edge_weight_transform = "log1p"
        g_data.node_feature_transform = "regex_clean_tfidf_svd_l2"
        g_data.node_id = node

        graphs_dict[node] = g_data

    print(f"Total graph Data objects constructed: {len(graphs_dict)}")

    # Fit temporal-feature statistics on normal training graphs only, then use
    # those fixed statistics for every split. The transition-count channel is
    # already log-scaled and is intentionally left unchanged.
    train_normal_nodes = set(
        manifest_df.loc[
            (manifest_df['split'] == 'train') & (manifest_df['graph_label'] == 0),
            'node'
        ]
    )
    train_edges = [
        graphs_dict[node].edge_attr[:, 1:3]
        for node in train_normal_nodes
        if node in graphs_dict
    ]
    if not train_edges:
        raise RuntimeError("No normal training edge attributes available for scaling.")
    train_edge_stats = torch.cat(train_edges, dim=0)
    temporal_mean = train_edge_stats.mean(dim=0)
    temporal_std = train_edge_stats.std(dim=0, unbiased=False).clamp_min(1e-6)
    for graph in graphs_dict.values():
        graph.edge_attr[:, 1:3] = (
            graph.edge_attr[:, 1:3] - temporal_mean
        ) / temporal_std
        graph.edge_attr_temporal_standardization = {
            "fit_split": "train_normal",
            "mean": temporal_mean.tolist(),
            "std": temporal_std.tolist(),
        }
    print(
        "Temporal edge z-score stats (train normal): "
        f"mean={temporal_mean.tolist()}, std={temporal_std.tolist()}"
    )

    torch.save(graphs_dict, output_pt_path)
    print(f"Saved temporal graph dataset to: {output_pt_path}")
    return graphs_dict

if __name__ == "__main__":
    logs_csv = "/Users/kushal/Downloads/BGL_logs.csv"
    splits_d = "/Users/kushal/log2graph/splits"
    out_pt = "/Users/kushal/log2graph/bgl_processed_graphs_temp.pt"
    graphs = prepare_temporal_bgl_dataset(logs_csv, splits_d, out_pt)
    torch.save(graphs, "/Users/kushal/log2graph/bgl_processed_graphs.pt")
    print("Saved to /Users/kushal/log2graph/bgl_processed_graphs.pt as well.")
