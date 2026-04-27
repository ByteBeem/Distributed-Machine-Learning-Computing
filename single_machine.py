"""
Uses the full dataset on one machine with plain gradient descent.

"""

import numpy as np
import pandas as pd
import time
import json
import os

# Hyperparameters 
LEARNING_RATE = 0.001
EPOCHS        = 50
DATA_PATH     = "data/full_dataset.csv"
RESULTS_DIR   = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y


def normalize(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def train(X, y, lr, epochs):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []

    for epoch in range(epochs):
        y_pred = X @ w + b
        error  = y_pred - y

        grad_w = (2 / n) * X.T @ error
        grad_b = (2 / n) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % 10 == 0 or epoch == epochs - 1:
            loss = mse(y, y_pred)
            history.append({"epoch": epoch, "loss": loss})
            print(f"  Epoch {epoch:>4d}/{epochs}  MSE={loss:.4f}")

    return w, b, history


def main():
    print("=" * 55)
    print("  SINGLE-MACHINE LINEAR REGRESSION BASELINE")
    print("=" * 55)

    # Load 
    print(f"\n[1] Loading data from {DATA_PATH} ...")
    X, y = load_data(DATA_PATH)
    print(f"    Samples: {len(X):,}   Features: {X.shape[1]}")

    # Normalize 
    X_norm, mean, std = normalize(X)

    # Train 
    print(f"\n[2] Training  (lr={LEARNING_RATE}, epochs={EPOCHS}) ...")
    t0 = time.perf_counter()
    w, b, history = train(X_norm, y, LEARNING_RATE, EPOCHS)
    elapsed = time.perf_counter() - t0

    # Metrics 
    y_pred = X_norm @ w + b
    metrics = {
        "mode"         : "single_machine",
        "samples"      : int(len(X)),
        "features"     : int(X.shape[1]),
        "epochs"       : EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_time_sec": round(elapsed, 4),
        "final_mse"    : round(mse(y, y_pred), 4),
        "final_rmse"   : round(rmse(y, y_pred), 4),
        "final_mae"    : round(mae(y, y_pred), 4),
        "r2_score"     : round(r2(y, y_pred), 4),
        "weights"      : w.tolist(),
        "bias"         : round(float(b), 4),
        "loss_history" : history,
    }

    # Save 
    out_path = os.path.join(RESULTS_DIR, "single_machine_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"  Training time : {elapsed:.4f} s")
    print(f"  MSE           : {metrics['final_mse']}")
    print(f"  RMSE          : {metrics['final_rmse']}")
    print(f"  MAE           : {metrics['final_mae']}")
    print(f"  R²            : {metrics['r2_score']}")
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
