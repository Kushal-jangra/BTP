#!/usr/bin/env python3
"""Load a saved OCDiGCN checkpoint and score preprocessed graph tensors."""

import argparse
import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from DataLoader import DiGCN
from run_bgl_paper_exact import TempPaperDiGCN


def load_checkpoint(checkpoint_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") == "temporal_digcn":
        model = TempPaperDiGCN(
            nfeat=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            temporal_dim=checkpoint["temporal_dim"],
            alpha=checkpoint.get("temporal_alpha", 0.1),
        )
    else:
        model = DiGCN(
            nfeat=checkpoint["input_dim"],
            nhid=checkpoint["hidden_dim"],
            nlayer=1,
            bias=False,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    center = checkpoint["center"].to(device)
    return model, center, checkpoint, device


def score_graphs(model, center, graphs, device, batch_size=64):
    if isinstance(graphs, dict):
        graph_items = list(graphs.items())
    else:
        graph_items = [(getattr(graph, "node_id", str(i)), graph)
                       for i, graph in enumerate(graphs)]
    ids = [str(graph_id) for graph_id, _ in graph_items]
    data_list = [graph for _, graph in graph_items]
    scores = []
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if hasattr(model, "graph_embeddings"):
                embeddings = model.graph_embeddings(batch)
            else:
                node_embeddings = model(batch)
                embeddings = torch.stack([item.mean(dim=0) for item in node_embeddings])
            scores.extend(torch.linalg.vector_norm(embeddings - center, dim=1).cpu().tolist())
    labels = [int(graph.y.item()) if hasattr(graph, "y") else None for graph in data_list]
    return pd.DataFrame({"graph_id": ids, "label": labels, "anomaly_score": scores})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/enhanced_bgl/ocdigcn_seed_42.pt")
    parser.add_argument("--graphs", default="bgl_enhanced_digcn_graphs.pt")
    parser.add_argument("--output", default="artifacts/enhanced_bgl/scored_graphs.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, center, checkpoint, device = load_checkpoint(args.checkpoint, args.device)
    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)
    scores = score_graphs(model, center, graphs, device, args.batch_size)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"Loaded seed {checkpoint['seed']} checkpoint on {device}")
    print(f"Scored {len(scores)} graphs; wrote {args.output}")
