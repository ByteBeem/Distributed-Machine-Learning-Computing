"""
coordinator.py  —  Node 0  (runs on YOUR LOCAL MACHINE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the MASTER / COORDINATOR node.
It:
  1. Waits for the worker (EC2) to connect
  2. Broadcasts current weights each epoch
  3. Receives the worker's gradient
  4. Averages gradients (Parameter Server approach)
  5. Updates global weights
  6. Saves final metrics

Usage:
    python coordinator.py

Start this BEFORE starting worker.py on EC2.
"""

import socket
import struct
import numpy as np
import pandas as pd
import time
import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
COORDINATOR_HOST = "0.0.0.0"   # listen on all interfaces
COORDINATOR_PORT = 9999         # open this port in your EC2 security group too
                                # (outbound from EC2, inbound to local — or swap)
PARTITION_PATH   = "data/partition_0.csv"
RESULTS_DIR      = "results"
LOGS_DIR         = "logs"

LEARNING_RATE    = 0.01
EPOCHS           = 200
FEATURES         = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ── Network helpers ───────────────────────────────────────────────────────────
def send_array(conn, arr: np.ndarray):
    """Send a float64 numpy array: 4-byte length header + data."""
    data = arr.astype(np.float64).tobytes()
    conn.sendall(struct.pack("!I", len(data)) + data)


def recv_array(conn, n_elements: int) -> np.ndarray:
    """Receive a float64 numpy array."""
    n_bytes = n_elements * 8
    buf = b""
    while len(buf) < n_bytes:
        chunk = conn.recv(n_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Worker disconnected")
        buf += chunk
    return np.frombuffer(buf, dtype=np.float64)


def send_scalar(conn, value: float):
    conn.sendall(struct.pack("!d", float(value)))


def recv_scalar(conn) -> float:
    raw = b""
    while len(raw) < 8:
        raw += conn.recv(8 - len(raw))
    return struct.unpack("!d", raw)[0]


# ── Data ──────────────────────────────────────────────────────────────────────
def load_partition(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y


def normalize(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


# ── Metrics ───────────────────────────────────────────────────────────────────
def mse(y_true, y_pred):   return float(np.mean((y_true - y_pred) ** 2))
def rmse(y_true, y_pred):  return float(np.sqrt(mse(y_true, y_pred)))
def mae(y_true, y_pred):   return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DISTRIBUTED LINEAR REGRESSION  —  COORDINATOR (Node 0)")
    print("=" * 60)

    # Load local partition
    print(f"\n[1] Loading partition → {PARTITION_PATH}")
    X_local, y_local = load_partition(PARTITION_PATH)
    X_local, _, _ = normalize(X_local)
    n_local = len(X_local)
    print(f"    Local samples: {n_local:,}")

    # Listen for worker
    print(f"\n[2] Listening for worker on port {COORDINATOR_PORT} ...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((COORDINATOR_HOST, COORDINATOR_PORT))
    server.listen(1)
    conn, addr = server.accept()
    print(f"    Worker connected from {addr}")

    # Receive worker's sample count
    n_worker = int(recv_scalar(conn))
    print(f"    Worker samples : {n_worker:,}")
    n_total = n_local + n_worker
    print(f"    Total samples  : {n_total:,}")

    # Initialise weights
    w = np.zeros(FEATURES)
    b = 0.0

    history    = []
    log_lines  = []
    t0         = time.perf_counter()

    print(f"\n[3] Training  (lr={LEARNING_RATE}, epochs={EPOCHS}) ...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        # ── Local gradient ──────────────────────────────────────
        y_pred_local = X_local @ w + b
        error_local  = y_pred_local - y_local
        grad_w_local = (2 / n_local) * X_local.T @ error_local
        grad_b_local = (2 / n_local) * np.sum(error_local)

        # ── Send weights to worker, receive worker gradient ──────
        send_array(conn, w)
        send_scalar(conn, b)

        grad_w_worker = recv_array(conn, FEATURES)
        grad_b_worker = recv_scalar(conn)

        # ── Federated average (weighted by partition size) ────────
        w_local  = n_local  / n_total
        w_worker = n_worker / n_total
        grad_w_avg = w_local * grad_w_local + w_worker * grad_w_worker
        grad_b_avg = w_local * grad_b_local + w_worker * grad_b_worker

        # ── Update global weights ─────────────────────────────────
        w -= LEARNING_RATE * grad_w_avg
        b -= LEARNING_RATE * grad_b_avg

        # ── Logging ───────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            loss = mse(y_local, X_local @ w + b)
            line = f"Epoch {epoch:>4d}/{EPOCHS}  MSE={loss:.4f}"
            print(f"  {line}")
            history.append({"epoch": epoch, "loss": round(loss, 4)})
            log_lines.append(line)

    elapsed = time.perf_counter() - t0

    # Signal worker to stop
    send_array(conn, np.array([-999.0]))   # sentinel
    conn.close()
    server.close()

    # ── Final metrics ─────────────────────────────────────────────────────────
    y_pred_final = X_local @ w + b
    metrics = {
        "mode"             : "distributed",
        "node"             : "coordinator",
        "total_samples"    : n_total,
        "local_samples"    : n_local,
        "worker_samples"   : n_worker,
        "features"         : FEATURES,
        "epochs"           : EPOCHS,
        "learning_rate"    : LEARNING_RATE,
        "training_time_sec": round(elapsed, 4),
        "final_mse"        : round(mse(y_local, y_pred_final), 4),
        "final_rmse"       : round(rmse(y_local, y_pred_final), 4),
        "final_mae"        : round(mae(y_local, y_pred_final), 4),
        "r2_score"         : round(r2(y_local, y_pred_final), 4),
        "weights"          : w.tolist(),
        "bias"             : round(float(b), 4),
        "loss_history"     : history,
    }

    out_path = os.path.join(RESULTS_DIR, "distributed_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log_path = os.path.join(LOGS_DIR, "coordinator.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print("\n" + "=" * 60)
    print("  DISTRIBUTED RESULTS (Coordinator view)")
    print("=" * 60)
    print(f"  Training time : {elapsed:.4f} s")
    print(f"  MSE           : {metrics['final_mse']}")
    print(f"  RMSE          : {metrics['final_rmse']}")
    print(f"  MAE           : {metrics['final_mae']}")
    print(f"  R²            : {metrics['r2_score']}")
    print(f"\n  Results saved → {out_path}")
    print(f"  Log saved     → {log_path}")


if __name__ == "__main__":
    main()
