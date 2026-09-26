#!/usr/bin/env python3
"""Build the paper-aligned BGL graph dictionary.

Unlike the exploratory temporal builder, this script uses the official
structured BGL data: EventTemplate nodes, official 200-d embeddings, and raw
transition counts. Mean/variance inter-event times are retained as auxiliary
channels for the later temporal experiment but are ignored by the baseline.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def _timestamp_seconds(frame):
    if "Timestamp" in frame:
        return frame["Timestamp"].astype(float).to_numpy()
    return pd.to_datetime(frame["Time"]).astype("int64").to_numpy() / 1e9


def prepare_paper_dataset(
    structured_csv, embedding_json, splits_dir, output,
    enhanced_node_features=False,
    transformer_model="sentence-transformers/all-MiniLM-L6-v2",
):
    df = pd.read_csv(structured_csv)
    required = {"Node", "EventTemplate", "Timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Structured BGL data is missing columns: {sorted(missing)}")

    manifest = pd.read_csv(os.path.join(splits_dir, "BGL_graph_split_manifest.csv"))
    node_to_split = dict(zip(manifest["node"], manifest["split"]))
    node_to_label = dict(zip(manifest["node"], manifest["graph_label"]))

    enhanced_builder = None
    if enhanced_node_features:
        from enhanced_node_builder import EnhancedNodeBuilder
        enhanced_builder = EnhancedNodeBuilder(model_name=transformer_model)
        embedding_dim = None
        template_to_idx = None
    else:
        with open(embedding_json) as f:
            embedding_dict = json.load(f)
        first_embedding = next(iter(embedding_dict.values()))
        embedding_dim = len(first_embedding)
        if isinstance(first_embedding, dict):
            def embedding_vector(template):
                values = embedding_dict[template]
                return [values[str(i)] for i in range(embedding_dim)]
        else:
            def embedding_vector(template):
                return embedding_dict[template]
        template_to_idx = {template: i for i, template in enumerate(embedding_dict)}

    graphs = {}
    for node, group in tqdm(df.groupby("Node"), total=df["Node"].nunique()):
        if node not in node_to_split:
            continue

        templates = group["EventTemplate"].astype(str).tolist()
        local_templates = list(dict.fromkeys(templates))
        if enhanced_builder is not None:
            # The builder preserves unique-template first-occurrence order.
            x = enhanced_builder.build_node_features(group.to_dict(orient="records"))
        else:
            unknown = [t for t in local_templates if t not in template_to_idx]
            if unknown:
                raise KeyError(f"Embedding dictionary lacks {len(unknown)} BGL templates")
            x = torch.tensor(
                [embedding_vector(template) for template in local_templates],
                dtype=torch.float32,
            )

        local_idx = {template: i for i, template in enumerate(local_templates)}

        timestamps = _timestamp_seconds(group)
        edge_deltas = {}
        for i in range(len(templates) - 1):
            edge = (local_idx[templates[i]], local_idx[templates[i + 1]])
            dt = max(0.0, float(timestamps[i + 1] - timestamps[i]))
            edge_deltas.setdefault(edge, []).append(dt)

        if not edge_deltas:
            edges = [(0, 0)]
            attrs = [[1.0, 0.0, 0.0]]
        else:
            edges = list(edge_deltas)
            attrs = []
            for edge in edges:
                dts = edge_deltas[edge]
                attrs.append([len(dts), float(np.mean(dts)), float(np.var(dts))])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([int(node_to_label[node])], dtype=torch.long),
        )
        graph.node_id = node
        graph.graph_split = node_to_split[node]
        graph.embedding_source = (
            f"transformer:{transformer_model}" if enhanced_builder is not None
            else "official_glove_tfidf_200d"
        )
        graph.edge_weight_transform = "raw_count"
        graphs[node] = graph

    if len(graphs) != len(manifest):
        raise RuntimeError(f"Built {len(graphs)} graphs but manifest contains {len(manifest)}")
    torch.save(graphs, output)

    nodes = np.array([g.x.size(0) for g in graphs.values()])
    edges = np.array([g.edge_index.size(1) for g in graphs.values()])
    print(f"Saved {len(graphs)} paper-aligned graphs to {output}")
    print(f"Node feature dimension: {graphs[next(iter(graphs))].x.size(1)}")
    print(f"Nodes per graph: mean={nodes.mean():.2f}, median={np.median(nodes):.2f}")
    print(f"Edges per graph: mean={edges.mean():.2f}, median={np.median(edges):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-csv", default="Data/BGL/BGL.log_structured.csv")
    parser.add_argument("--embedding-json", default="Data/Gloves/Results/EmbeddingDict_BGL.json")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--output", default="bgl_paper_graphs.pt")
    parser.add_argument(
        "--enhanced-node-features", action="store_true",
        help="Use contextual Transformer and log-level/statistical node features.",
    )
    parser.add_argument(
        "--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face encoder used with --enhanced-node-features.",
    )
    args = parser.parse_args()
    prepare_paper_dataset(
        args.structured_csv, args.embedding_json, args.splits_dir, args.output,
        enhanced_node_features=args.enhanced_node_features,
        transformer_model=args.transformer_model,
    )
