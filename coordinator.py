# coordinator.py
import socket
import struct
import numpy as np
import pandas as pd
import time
import json
import os
from tqdm import tqdm



# CONFIG
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

pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

# NETWORK HELPERS

def send_array(conn, arr):
    data = arr.astype(np.float64).tobytes()
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)


def recv_array(conn):
    raw_len = recv_exact(conn, 4)
    size = struct.unpack("!I", raw_len)[0]
    data = recv_exact(conn, size)
    return np.frombuffer(data, dtype=np.float64)


def send_scalar(conn, val):
    conn.sendall(struct.pack("!d", float(val)))


def recv_scalar(conn):
    raw = recv_exact(conn, 8)
    return struct.unpack("!d", raw)[0]


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Disconnected")
        buf += chunk
    return buf


# METRICS
def mse(y, yp):
    return float(np.mean((y - yp) ** 2))


def rmse(y, yp):
    return float(np.sqrt(mse(y, yp)))


def mae(y, yp):
    return float(np.mean(np.abs(y - yp)))


def r2(y, yp):
    num = np.sum((y - yp) ** 2)
    den = np.sum((y - np.mean(y)) ** 2)
    return float(1 - (num / den))


# MAIN
def main():
    print("=" * 60)
    print("COORDINATOR NODE")
    print("=" * 60)

    # Load normalization stats
    with open(NORM_PATH) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    # Load local data
    df = pd.read_csv(PARTITION_PATH)
    cols = [c for c in df.columns if c != "target"]

    X = df[cols].values
    y = df["target"].values

    X = (X - mean) / std
    n_local = len(X)

    print("Local samples:", n_local)

    # Server start
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for worker...")
    conn, addr = server.accept()
    print("Connected:", addr)

    n_worker = int(recv_scalar(conn))
    n_total = n_local + n_worker

    print("Worker samples:", n_worker)
    print("Total samples:", n_total)

    # Model params
    w = np.zeros(FEATURES)
    b = 0.0

    history = []
    logs = []

    start = time.perf_counter()

    for epoch in pbar:

        # local gradient
        yp = X_local @ w + b
        err = yp - y_local

        gw_local = (2 / n_local) * (X_local.T @ err)
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

        # compute loss 
        loss = mse(y_local, X_local @ w + b)

        # update progress bar
        pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "w_norm": f"{np.linalg.norm(w):.3f}"
        })

    elapsed = time.perf_counter() - start

    # Stop worker
    send_array(conn, np.array([-999.0]))
    conn.close()
    server.close()

    yp_final = X @ w + b

    results = {
        "mse": mse(y, yp_final),
        "rmse": rmse(y, yp_final),
        "mae": mae(y, yp_final),
        "r2": r2(y, yp_final),
        "weights": w.tolist(),
        "bias": float(b),
        "time_sec": elapsed,
        "history": history,
    }

    with open("results/distributed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("logs/coordinator.log", "w") as f:
        f.write("\n".join(logs))

    print("Finished Training")


if __name__ == "__main__":
    main()