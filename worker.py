# worker.py
import socket
import struct
import numpy as np
import pandas as pd
import json
import time
import os
from tqdm import tqdm

pbar = tqdm(total=None, desc="Worker training", unit="epoch")


# CONFIG

COORDINATOR_IP = "13.60.80.179"
PORT = 8443

PARTITION_PATH = "data/partition_0.csv"
NORM_PATH = "data/norm_stats.json"

FEATURES = 8
LOGS_DIR = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)


# NETWORK
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



# MAIN
def main():
    print("=" * 60)
    print("WORKER NODE")
    print("=" * 60)

    with open(NORM_PATH) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    df = pd.read_csv(PARTITION_PATH)
    cols = [c for c in df.columns if c != "target"]

    X = df[cols].values
    y = df["target"].values

    X = (X - mean) / std
    n_local = len(X)

    print("Samples:", n_local)

    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((COORDINATOR_IP, PORT))

    send_scalar(conn, n_local)

    logs = []
    epoch = 0

    start = time.perf_counter()

    while True:
        w = recv_array(conn)

        if len(w) == 1 and w[0] == -999.0:
            break

        b = recv_scalar(conn)

        yp = X @ w + b
        err = yp - y

        gw = (2 / n_local) * (X.T @ err)
        gb = (2 / n_local) * np.sum(err)

        gw = np.clip(gw, -1, 1)
        gb = float(np.clip(gb, -1, 1))

        send_array(conn, gw)
        send_scalar(conn, gb)

        loss = float(np.mean(err ** 2))
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        pbar.update(1)

        epoch += 1

     

    elapsed = time.perf_counter() - start

    with open("logs/worker.log", "w") as f:
        f.write("\n".join(logs))

    conn.close()

    print("Done")
    print("Epochs:", epoch)
    print("Time:", elapsed)


if __name__ == "__main__":
    main()