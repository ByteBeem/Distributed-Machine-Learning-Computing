# coordinator.py
import socket
import struct
import numpy as np
import pandas as pd
import time
import json
import os
from tqdm import tqdm


# ===============================
# CONFIG
# ===============================
HOST = "0.0.0.0"
PORT = 8443

PARTITION_PATH = "data/partition_1.csv"
NORM_PATH = "data/norm_stats.json"

RESULTS_DIR = "results"
LOGS_DIR = "logs"

LEARNING_RATE = 0.001
EPOCHS = 2000
FEATURES = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ===============================
# NETWORK HELPERS
# ===============================
def send_array(conn, arr):
    data = arr.astype(np.float64).tobytes()
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Disconnected")
        buf += chunk
    return buf


def recv_array(conn):
    size = struct.unpack("!I", recv_exact(conn, 4))[0]
    return np.frombuffer(recv_exact(conn, size), dtype=np.float64)


def send_scalar(conn, val):
    conn.sendall(struct.pack("!d", float(val)))


def recv_scalar(conn):
    return struct.unpack("!d", recv_exact(conn, 8))[0]


# ===============================
# METRICS
# ===============================
def mse(y, yp):
    return float(np.mean((y - yp) ** 2))


def r2(y, yp):
    return float(1 - np.sum((y - yp) ** 2) / np.sum((y - np.mean(y)) ** 2))


# ===============================
# MAIN
# ===============================
def main():
    print("=" * 60)
    print("COORDINATOR NODE")
    print("=" * 60)

    # Load norm stats
    with open(NORM_PATH) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    # Load data
    df = pd.read_csv(PARTITION_PATH)
    cols = [c for c in df.columns if c != "target"]

    X = (df[cols].values - mean) / std
    y = df["target"].values

    n_local = len(X)

    print("Local samples:", n_local)

    # Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for worker...")
    conn, addr = server.accept()
    print("Connected:", addr)

    n_worker = int(recv_scalar(conn))
    n_total = n_local + n_worker

    print("Worker samples:", n_worker)

    # Model
    w = np.zeros(FEATURES)
    b = 0.0

    start = time.perf_counter()

    # ===============================
    # REAL PROGRESS BAR
    # ===============================
    with tqdm(range(EPOCHS), desc="Training", unit="epoch") as pbar:

        for epoch in pbar:

            # local gradient
            yp = X @ w + b
            err = yp - y

            gw_local = (2 / n_local) * (X.T @ err)
            gb_local = (2 / n_local) * np.sum(err)

            send_array(conn, w)
            send_scalar(conn, b)

            gw_worker = recv_array(conn)
            gb_worker = recv_scalar(conn)

            wl = n_local / n_total
            ww = n_worker / n_total

            gw = wl * gw_local + ww * gw_worker
            gb = wl * gb_local + ww * gb_worker

            gw = np.clip(gw, -1, 1)
            gb = float(np.clip(gb, -1, 1))

            w -= LEARNING_RATE * gw
            b -= LEARNING_RATE * gb

            loss = mse(y, X @ w + b)

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "r2": f"{r2(y, X @ w + b):.3f}"
            })

    elapsed = time.perf_counter() - start

    # stop worker
    send_array(conn, np.array([-999.0]))
    conn.close()
    server.close()

    print("\nFinished Training")
    print("Time:", elapsed)


if __name__ == "__main__":
    main()