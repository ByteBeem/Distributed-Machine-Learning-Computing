"""
generate_data.py
Run this ONCE on your local machine to create and split the dataset.
It will produce:
  data/full_dataset.csv
  data/partition_0.csv   <-- stays on your local machine (node 0)
  data/partition_1.csv   <-- copy this to your EC2 instance
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10_000          # total samples
FEATURES = 5        # number of input features
NOISE = 15.0        # noise level

# True weights the model should learn
true_weights = np.array([3.5, -2.1, 4.8, 1.2, -3.3])
true_bias    = 7.0

X = np.random.randn(N, FEATURES)
y = X @ true_weights + true_bias + np.random.randn(N) * NOISE

# Build DataFrame
cols = [f"feature_{i}" for i in range(FEATURES)]
df = pd.DataFrame(X, columns=cols)
df["target"] = y

os.makedirs("data", exist_ok=True)
df.to_csv("data/full_dataset.csv", index=False)
print(f"Full dataset saved: {len(df)} rows, {FEATURES} features")

# Split 50/50 for two nodes
mid = N // 2
df.iloc[:mid].to_csv("data/partition_0.csv", index=False)
df.iloc[mid:].to_csv("data/partition_1.csv", index=False)
print(f"Partition 0 (local machine): {mid} rows  → data/partition_0.csv")
print(f"Partition 1 (EC2 instance) : {N-mid} rows  → data/partition_1.csv")
print("\nNext step: copy data/partition_1.csv to your EC2 instance into the same project folder.")
