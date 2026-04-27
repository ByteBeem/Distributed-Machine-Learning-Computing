"""
Run this ONCE on whichever machine holds the full dataset (or both
partitions).  It computes global mean/std from all available data
and writes data/norm_stats.json.

"""

import numpy as np
import pandas as pd
import json
import os
import glob

DATA_DIR    = "data"
OUT_PATH    = os.path.join(DATA_DIR, "norm_stats.json")
TARGET_COL  = "target"

def main():
    # Collect every partition CSV in data/
    partition_files = sorted(glob.glob(os.path.join(DATA_DIR, "partition_*.csv")))
    if not partition_files:
        raise FileNotFoundError(f"No partition_*.csv files found in '{DATA_DIR}/'")

    print(f"Found {len(partition_files)} partition(s):")
    for p in partition_files:
        print(f"  {p}")

    frames = [pd.read_csv(p) for p in partition_files]
    df_all = pd.concat(frames, ignore_index=True)

    feat_cols = [c for c in df_all.columns if c != TARGET_COL]
    X = df_all[feat_cols].values.astype(np.float64)

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8   # epsilon avoids division by zero

    stats = {
        "mean":     mean.tolist(),
        "std":      std.tolist(),
        "features": feat_cols,
        "n_samples": int(len(df_all)),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNorm stats computed from {len(df_all):,} total samples.")
    print(f"Features : {feat_cols}")
    print(f"Mean     : {np.round(mean, 4).tolist()}")
    print(f"Std      : {np.round(std,  4).tolist()}")
    print(f"\nSaved → {OUT_PATH}")
    print("\nNext step: copy this file to all nodes before training:")
    print(f"  scp {OUT_PATH} user@<worker-ip>:/path/to/project/data/")

if __name__ == "__main__":
    main()