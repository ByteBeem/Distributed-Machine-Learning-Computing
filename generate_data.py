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
from sklearn.datasets import fetch_california_housing

np.random.seed(42)

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Feature names from the dataset
feature_names = housing.feature_names

# Build DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

N = len(df)
FEATURES = len(feature_names)

os.makedirs("data", exist_ok=True)
df.to_csv("data/full_dataset.csv", index=False)
print(f"Full dataset saved: {len(df)} rows, {FEATURES} features")
print(f"Features: {', '.join(feature_names)}")

# Split 50/50 for two nodes
mid = N // 2
df.iloc[:mid].to_csv("data/partition_0.csv", index=False)
df.iloc[mid:].to_csv("data/partition_1.csv", index=False)
print(f"Partition 0 (local machine): {mid} rows  → data/partition_0.csv")
print(f"Partition 1 (EC2 instance) : {N-mid} rows  → data/partition_1.csv")
print("\nNext step: copy data/partition_1.csv to your EC2 instance into the same project folder.")
