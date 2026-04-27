"""
worker.py  —  Node 1  (runs on YOUR EC2 INSTANCE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the WORKER node.
It:
  1. Connects to the coordinator (your local machine)
  2. Sends its partition size
  3. Each epoch: receives weights → computes gradient → sends gradient back

BEFORE running:
  1. Copy partition_1.csv to this machine: data/partition_1.csv
  2. Set COORDINATOR_IP below to your LOCAL machine's PUBLIC IP
     (check with: curl ifconfig.me  on your local machine, OR use its
      private IP if both machines are in the same AWS VPC)

Usage:
    python worker.py
"""

import socket
import struct
import numpy as np
import pandas as pd
import time
import os

# ── EDIT THIS ─────────────────────────────────────────────────────────────────
COORDINATOR_IP   = "YOUR_LOCAL_MACHINE_PUBLIC_IP"   # <-- change this
COORDINATOR_PORT = 9999
# ──────────────────────────────────────────────────────────────────────────────

PARTITION_PATH = "data/partition_1.csv"
FEATURES       = 5
LOGS_DIR       = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)


# ── Network helpers (must match coordinator.py) ───────────────────────────────
def send_array(conn, arr: np.ndarray):
    data = arr.astype(np.float64).tobytes()
    conn.sendall(struct.pack("!I", len(data)) + data)


def recv_array(conn, n_elements: int) -> np.ndarray:
    n_bytes = n_elements * 8
    buf = b""
    while len(buf) < n_bytes:
        chunk = conn.recv(n_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Coordinator disconnected")
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DISTRIBUTED LINEAR REGRESSION  —  WORKER (Node 1 / EC2)")
    print("=" * 60)

    print(f"\n[1] Loading partition → {PARTITION_PATH}")
    X_local, y_local = load_partition(PARTITION_PATH)
    X_local, _, _    = normalize(X_local)
    n_local          = len(X_local)
    print(f"    Local samples: {n_local:,}")

    print(f"\n[2] Connecting to coordinator at {COORDINATOR_IP}:{COORDINATOR_PORT} ...")
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((COORDINATOR_IP, COORDINATOR_PORT))
    print("    Connected!")

    # Tell coordinator how many samples we have
    send_scalar(conn, float(n_local))

    log_lines = []
    epoch     = 0
    t0        = time.perf_counter()

    print("\n[3] Training ...")
    print("-" * 60)

    while True:
        # Receive current weights from coordinator
        w_data = recv_array(conn, FEATURES)

        # Sentinel: coordinator signals end of training
        if len(w_data) == 1 and w_data[0] == -999.0:
            print("  Training complete signal received.")
            break

        w = w_data
        b = recv_scalar(conn)

        # Compute local gradient
        y_pred = X_local @ w + b
        error  = y_pred - y_local
        grad_w = (2 / n_local) * X_local.T @ error
        grad_b = (2 / n_local) * np.sum(error)

        # Send gradients to coordinator
        send_array(conn, grad_w)
        send_scalar(conn, grad_b)

        if epoch % 10 == 0:
            loss = float(np.mean(error ** 2))
            line = f"Epoch {epoch:>4d}  local_MSE={loss:.4f}"
            print(f"  {line}")
            log_lines.append(line)

        epoch += 1

    elapsed = time.perf_counter() - t0
    conn.close()

    log_path = os.path.join(LOGS_DIR, "worker.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n  Done. Epochs completed : {epoch}")
    print(f"  Wall time              : {elapsed:.4f} s")
    print(f"  Log saved              → {log_path}")


if __name__ == "__main__":
    main()
