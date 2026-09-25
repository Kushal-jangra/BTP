#!/usr/bin/env python3
"""Prepare high-resolution temporal summaries without modifying baseline graphs."""

import argparse
import numpy as np
import pandas as pd
import torch


def prepare(structured_csv, output):
    df = pd.read_csv(structured_csv)
    if "Time" not in df or "Node" not in df:
        raise ValueError("BGL structured CSV must contain Node and Time columns")
    timestamps = pd.to_datetime(
        df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce"
    )
    if timestamps.isna().any():
        timestamps = pd.to_datetime(df["Time"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Unable to parse high-resolution BGL Time values")
    seconds = timestamps.astype("int64").to_numpy() / 1e9

    features = {}
    for node, indices in df.groupby("Node", sort=False).groups.items():
        idx = np.asarray(indices)
        deltas = np.diff(seconds[idx])
        deltas = np.maximum(deltas, 0.0)
        log_deltas = np.log1p(deltas)
        if len(log_deltas) == 0:
            summary = np.zeros(6, dtype=np.float32)
        else:
            summary = np.array([
                log_deltas.mean(),
                np.median(log_deltas),
                np.percentile(log_deltas, 95),
                log_deltas.std(),
                log_deltas.max(),
                (deltas == 0).mean(),
            ], dtype=np.float32)
        features[node] = torch.tensor(summary).view(1, -1)
    torch.save(features, output)
    values = torch.cat(list(features.values()))
    print(f"Saved high-resolution temporal features for {len(features)} graphs to {output}")
    print(f"Feature mean: {values.mean(0).tolist()}")
    print(f"Feature std: {values.std(0, unbiased=False).tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-csv", default="Data/BGL/BGL.log_structured.csv")
    parser.add_argument("--output", default="bgl_temporal_features.pt")
    args = parser.parse_args()
    prepare(args.structured_csv, args.output)
