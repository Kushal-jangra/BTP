#!/usr/bin/env python3
"""Build enhanced BGL graphs directly from the local processed BGL CSV.

The input is scanned in chunks to keep peak memory reasonable. It creates a
deterministic, stratified 10k-graph Protocol-A-style sample by default, then
stores graph tensors and compatible train/validation/test ID files in the
workspace. No dataset is copied into the repository.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from enhanced_node_builder import EnhancedNodeBuilder


def _column(columns, choices, required=True):
    lookup = {str(name).strip().lower(): name for name in columns}
    for choice in choices:
        if choice.lower() in lookup:
            return lookup[choice.lower()]
    if required:
        raise ValueError(f"CSV is missing one of the required columns: {choices}")
    return None


def _make_manifest(input_csv, max_graphs, seed, chunk_size):
    columns = pd.read_csv(input_csv, nrows=0).columns
    node_col = _column(columns, ["node", "groupid", "group_id"])
    label_col = _column(columns, ["binary_label", "label", "anomaly_label"])
    population = {}
    usecols = [node_col, label_col]
    print("Counting group labels (a group is anomalous if any record is anomalous)...")
    for chunk in pd.read_csv(
        input_csv, usecols=usecols, chunksize=chunk_size,
        dtype={node_col: "string"},
    ):
        chunk = chunk.dropna(subset=[node_col, label_col])
        chunk[node_col] = chunk[node_col].astype(str)
        chunk[label_col] = pd.to_numeric(chunk[label_col], errors="coerce")
        chunk = chunk.dropna(subset=[label_col])
        # Log-level BGL labels may vary within a host/session; aggregate with
        # the common group-anomaly rule that any anomalous event marks its graph.
        for node, label in chunk.groupby(node_col, sort=False)[label_col].max().items():
            value = int(label)
            if value not in (0, 1):
                raise ValueError(f"Expected binary labels 0/1; found {value!r}")
            node = str(node)
            population[node] = max(value, population.get(node, 0))

    normal = np.asarray([node for node, label in population.items() if label == 0], dtype=object)
    anomaly = np.asarray([node for node, label in population.items() if label == 1], dtype=object)
    total = len(normal) + len(anomaly)
    if not len(normal) or not len(anomaly):
        raise ValueError("The input must contain both normal and anomalous groups.")
    target = min(int(max_graphs), total)
    anomaly_count = min(len(anomaly), int(round(target * len(anomaly) / total)))
    normal_count = min(len(normal), target - anomaly_count)
    anomaly_count = min(len(anomaly), target - normal_count)
    rng = np.random.default_rng(seed)
    sampled_normal = rng.permutation(normal)[:normal_count]
    sampled_anomaly = rng.permutation(anomaly)[:anomaly_count]

    # Train only on normal groups. Validation includes a balanced anomaly set;
    # remaining groups form the held-out test set.
    train_end = int(round(0.70 * normal_count))
    val_normal_end = train_end + int(round(0.05 * normal_count))
    val_anomaly_count = min(val_normal_end - train_end, anomaly_count)
    rows = []
    rows.extend((node, "train", 0) for node in sampled_normal[:train_end])
    rows.extend((node, "validation", 0) for node in sampled_normal[train_end:val_normal_end])
    rows.extend((node, "test", 0) for node in sampled_normal[val_normal_end:])
    rows.extend((node, "validation", 1) for node in sampled_anomaly[:val_anomaly_count])
    rows.extend((node, "test", 1) for node in sampled_anomaly[val_anomaly_count:])
    manifest = pd.DataFrame(rows, columns=["node", "split", "graph_label"])
    if manifest["node"].duplicated().any():
        raise RuntimeError("Split generation produced duplicate group IDs")
    return manifest, node_col


def prepare_local_dataset(input_csv, output, splits_dir, model_name, max_graphs=10000,
                          chunk_size=100000, seed=42, device=None):
    manifest, node_col = _make_manifest(input_csv, max_graphs, seed, chunk_size)
    columns = pd.read_csv(input_csv, nrows=0).columns
    label_col = _column(columns, ["binary_label", "label", "anomaly_label"])
    template_col = _column(columns, ["event_template", "eventtemplate", "event_template_text", "event_id"])
    message_col = _column(columns, ["message", "content", "raw_message"], required=False)
    level_col = _column(columns, ["level", "severity", "log_level"], required=False)
    component_col = _column(columns, ["component", "module", "subsystem"], required=False)
    subsystem_col = _column(columns, ["subsystem", "module"], required=False)
    timestamp_col = _column(columns, ["timestamp", "absolute_timestamp_epoch", "datetime"], required=False)

    os.makedirs(splits_dir, exist_ok=True)
    manifest.to_csv(os.path.join(splits_dir, "BGL_graph_split_manifest.csv"), index=False)
    for split in ("train", "validation", "test"):
        manifest.loc[manifest.split == split, ["node", "graph_label"]].to_csv(
            os.path.join(splits_dir, f"{split}_graph_ids.csv"), index=False
        )

    selected = set(manifest.node)
    groups = {node: [] for node in selected}
    usecols = list(dict.fromkeys(
        [node_col, template_col, label_col]
        + [col for col in (message_col, level_col, component_col, subsystem_col, timestamp_col) if col]
    ))
    print(f"Reading selected groups from {input_csv} in chunks...")
    for chunk in pd.read_csv(
        input_csv, usecols=usecols, chunksize=chunk_size,
        low_memory=False, dtype={node_col: "string"},
    ):
        chunk[node_col] = chunk[node_col].astype(str)
        chunk = chunk[chunk[node_col].isin(selected)]
        for row in chunk.to_dict(orient="records"):
            node = str(row[node_col])
            template = str(row[template_col])
            if not template or template.lower() == "nan":
                template = str(row.get(message_col, "UNKNOWN")) if message_col else "UNKNOWN"
            record = {"EventTemplate": template}
            if message_col:
                record["Content"] = str(row.get(message_col, ""))
            if level_col:
                record["Level"] = str(row.get(level_col, "UNKNOWN"))
            if component_col:
                record["Component"] = str(row.get(component_col, "UNKNOWN"))
            if subsystem_col:
                record["Subsystem"] = str(row.get(subsystem_col, "UNKNOWN"))
            if timestamp_col:
                record["Timestamp"] = row.get(timestamp_col)
            groups[node].append(record)

    missing = [node for node, records in groups.items() if not records]
    if missing:
        raise RuntimeError(f"CSV scan did not find {len(missing)} sampled groups")

    builder = EnhancedNodeBuilder(model_name=model_name, device=device)
    all_templates = list(dict.fromkeys(
        row["EventTemplate"] for records in groups.values() for row in records
    ))
    print(f"Encoding {len(all_templates)} distinct templates with {model_name}...")
    builder.extract_template_embedding(all_templates)

    label_by_node = dict(zip(manifest.node, manifest.graph_label))
    split_by_node = dict(zip(manifest.node, manifest.split))
    graphs = {}
    for node, records in tqdm(groups.items(), total=len(groups), desc="Building graphs"):
        templates = [row["EventTemplate"] for row in records]
        local_templates = list(dict.fromkeys(templates))
        local_index = {event: idx for idx, event in enumerate(local_templates)}
        x = builder.build_node_features(records)
        times = []
        for record in records:
            value = record.get("Timestamp")
            try:
                times.append(float(value))
            except (TypeError, ValueError):
                times.append(float(len(times)))
        edges = {}
        for i in range(len(templates) - 1):
            src, dst = local_index[templates[i]], local_index[templates[i + 1]]
            dt = max(0.0, times[i + 1] - times[i])
            edges.setdefault((src, dst), []).append(dt)
        if edges:
            edge_pairs = list(edges)
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([
                [len(edges[key]), float(np.mean(edges[key])), float(np.var(edges[key]))]
                for key in edge_pairs
            ], dtype=torch.float32)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        graph = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=torch.tensor([int(label_by_node[node])], dtype=torch.long),
        )
        graph.node_id = node
        graph.graph_split = split_by_node[node]
        graph.embedding_source = f"transformer:{model_name}"
        graph.edge_weight_transform = "raw_count"
        graphs[node] = graph

    torch.save(graphs, output)
    print(f"Saved {len(graphs)} enhanced graphs to {output}")
    print(f"Splits saved to {splits_dir}")
    print(f"Node feature dimension: {next(iter(graphs.values())).x.size(1)}")
    print("Graph counts by split:", manifest.split.value_counts().to_dict())
    print("Labels by split:\n", pd.crosstab(manifest.split, manifest.graph_label))
    return graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local processed BGL CSV")
    parser.add_argument("--output", default="bgl_enhanced_graphs.pt")
    parser.add_argument("--splits-dir", default="splits_enhanced")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-graphs", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    prepare_local_dataset(
        args.input, args.output, args.splits_dir, args.model, args.max_graphs,
        args.chunk_size, args.seed, args.device,
    )
