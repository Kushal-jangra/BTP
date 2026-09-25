#!/usr/bin/env python3
"""
generate_downsampled_protocol_a_splits.py

Stratified downsampling of BGL graph population (69,251 graphs -> 10,000 graphs)
following Protocol A (Base Paper benchmark setup):
- Preserves ~45.3% anomaly ratio (5,470 Normal graphs, 4,530 Anomalous graphs).
- Train Set: 3,829 Normal graphs (70% of normal pool), 0 Anomalous graphs.
- Val Set: 273 Normal graphs (5% of normal pool), 273 Anomalous graphs (50% anomaly ratio).
- Test Set: 1,368 Normal graphs (25% of normal pool), 4,257 Anomalous graphs.
"""

import os
import pandas as pd
import numpy as np

def generate_protocol_a_splits(splits_dir, seed=42):
    manifest_path = os.path.join(splits_dir, "BGL_graph_split_manifest.csv")
    print(f"Reading full manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"Total graph population in manifest: {len(df)}")

    normal_df = df[df['graph_label'] == 0].copy()
    anomaly_df = df[df['graph_label'] == 1].copy()

    total_population = len(df)
    n_normal_pop = len(normal_df)
    n_anomaly_pop = len(anomaly_df)
    print(f"Population breakdown: Normal={n_normal_pop}, Anomalous={n_anomaly_pop}")

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Stratified downsample to 10,000 total graphs
    target_total = 10000
    target_normal = int(round(target_total * (n_normal_pop / total_population))) # 5470
    target_anomaly = target_total - target_normal # 4530

    print(f"\n--- 1. Downsampling to {target_total} graphs ---")
    print(f"Target Normal: {target_normal} | Target Anomalous: {target_anomaly}")

    sampled_normal_indices = rng.choice(normal_df.index, size=target_normal, replace=False)
    sampled_anomaly_indices = rng.choice(anomaly_df.index, size=target_anomaly, replace=False)

    sample_normal_df = normal_df.loc[sampled_normal_indices].copy()
    sample_anomaly_df = anomaly_df.loc[sampled_anomaly_indices].copy()

    # Shuffle sampled subsets
    sample_normal_df = sample_normal_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample_anomaly_df = sample_anomaly_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 2. Partition Normal Graphs (70% Train, 5% Val, 25% Test)
    n_train_normal = int(round(0.70 * target_normal)) # 3829
    n_val_normal = int(round(0.05 * target_normal))   # 273
    n_test_normal = target_normal - n_train_normal - n_val_normal # 1368

    train_normal = sample_normal_df.iloc[:n_train_normal].copy()
    val_normal = sample_normal_df.iloc[n_train_normal : n_train_normal + n_val_normal].copy()
    test_normal = sample_normal_df.iloc[n_train_normal + n_val_normal:].copy()

    # Assign split labels
    train_normal['split'] = 'train'
    val_normal['split'] = 'validation'
    test_normal['split'] = 'test'

    # 3. Partition Anomalous Graphs (0 Train, Val matches Val_Normal, rest in Test)
    n_val_anomaly = n_val_normal # 273
    val_anomaly = sample_anomaly_df.iloc[:n_val_anomaly].copy()
    test_anomaly = sample_anomaly_df.iloc[n_val_anomaly:].copy()

    val_anomaly['split'] = 'validation'
    test_anomaly['split'] = 'test'

    # Combine into splits
    train_split = train_normal
    val_split = pd.concat([val_normal, val_anomaly], ignore_index=True)
    test_split = pd.concat([test_normal, test_anomaly], ignore_index=True)

    # Combine full 10,000 manifest
    downsampled_manifest = pd.concat([train_split, val_split, test_split], ignore_index=True)

    print("\n--- 2. Protocol A Split Counts ---")
    print(f"Train Set: {len(train_split)} total | Normal: {len(train_normal)} | Anomalous: 0")
    print(f"Val Set:   {len(val_split)} total | Normal: {len(val_normal)} | Anomalous: {len(val_anomaly)} (Anomaly Rate: {len(val_anomaly)/len(val_split)*100:.2f}%)")
    print(f"Test Set:  {len(test_split)} total | Normal: {len(test_normal)} | Anomalous: {len(test_anomaly)} (Anomaly Rate: {len(test_anomaly)/len(test_split)*100:.2f}%)")

    # 4. Leakage & Integrity Check
    train_nodes = set(train_split['node'])
    val_nodes = set(val_split['node'])
    test_nodes = set(test_split['node'])

    assert len(train_nodes & val_nodes) == 0, "Leakage between Train and Val!"
    assert len(train_nodes & test_nodes) == 0, "Leakage between Train and Test!"
    assert len(val_nodes & test_nodes) == 0, "Leakage between Val and Test!"
    print("\n✓ ZERO split leakage confirmed across Train, Val, and Test sets!")

    # 5. Save Split Files
    train_path = os.path.join(splits_dir, "train_graph_ids.csv")
    val_path = os.path.join(splits_dir, "validation_graph_ids.csv")
    test_path = os.path.join(splits_dir, "test_graph_ids.csv")
    out_manifest_path = os.path.join(splits_dir, "BGL_graph_split_manifest.csv")
    audit_path = os.path.join(splits_dir, "BGL_split_audit.txt")

    train_split[['node', 'graph_label']].to_csv(train_path, index=False)
    val_split[['node', 'graph_label']].to_csv(val_path, index=False)
    test_split[['node', 'graph_label']].to_csv(test_path, index=False)
    downsampled_manifest.to_csv(out_manifest_path, index=False)

    audit_text = f"""=================================================================
BGL LOG ANOMALY DETECTION - PROTOCOL A (BASE PAPER BENCHMARK) SPLIT AUDIT
=================================================================
Random seed: {seed}
Stratified Downsampled Population: {len(downsampled_manifest)} graphs

SOURCE POPULATION DOWN-SAMPLED
-----------------------------------------------------------------
Total sampled graphs: {len(downsampled_manifest)}
Normal graphs:        {len(sample_normal_df)} ({len(sample_normal_df)/len(downsampled_manifest)*100:.2f}%)
Anomalous graphs:     {len(sample_anomaly_df)} ({len(sample_anomaly_df)/len(downsampled_manifest)*100:.2f}%)

FINAL SPLITS (Protocol A)
-----------------------------------------------------------------
TRAIN:
  Total graphs: {len(train_split)}
  Normal:       {len(train_normal)}
  Anomalous:    0
  Anomaly rate: 0.00%

VALIDATION:
  Total graphs: {len(val_split)}
  Normal:       {len(val_normal)}
  Anomalous:    {len(val_anomaly)}
  Anomaly rate: {len(val_anomaly)/len(val_split)*100:.2f}%

TEST:
  Total graphs: {len(test_split)}
  Normal:       {len(test_normal)}
  Anomalous:    {len(test_anomaly)}
  Anomaly rate: {len(test_anomaly)/len(test_split)*100:.2f}%

LEAKAGE CHECK
-----------------------------------------------------------------
Train/Validation graph overlap: {len(train_nodes & val_nodes)}
Train/Test graph overlap:       {len(train_nodes & test_nodes)}
Validation/Test graph overlap: {len(val_nodes & test_nodes)}
=================================================================
"""
    with open(audit_path, "w") as f:
        f.write(audit_text)

    print(f"\nSaved updated split CSVs and manifest to: {splits_dir}")
    print(f"Audit log saved to: {audit_path}")

if __name__ == "__main__":
    splits_directory = "/Users/kushal/log2graph/splits"
    generate_protocol_a_splits(splits_directory, seed=42)
