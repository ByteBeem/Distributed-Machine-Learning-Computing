# Distributed Machine Learning — Assignment 2
## Linear Regression across Local Machine + AWS EC2

---

## Project Structure

```
distributed_ml/
├── generate_data.py        # Step 1 — create & split dataset
├── single_machine.py       # Step 2 — baseline (one machine)
├── coordinator.py          # Step 3a — distributed master  (LOCAL machine)
├── worker.py               # Step 3b — distributed worker  (EC2 instance)
├── compare_metrics.py      # Step 4 — side-by-side results + charts
├── requirements.txt
├── data/
│   ├── full_dataset.csv    # created by generate_data.py
│   ├── partition_0.csv     # local machine's slice
│   └── partition_1.csv     # copy this to EC2
├── results/                # JSON results + charts saved here
└── logs/                   # per-node training logs
```

---

## Architecture

```
┌──────────────────────────┐          TCP Socket          ┌──────────────────────────┐
│   LOCAL MACHINE (Node 0) │  ◄──────────────────────────►│   AWS EC2   (Node 1)     │
│                          │                               │                          │
│  coordinator.py          │  sends: current weights (w,b)│  worker.py               │
│  partition_0.csv (50%)   │  recvs: local gradient        │  partition_1.csv (50%)   │
│                          │                               │                          │
│  • Aggregates gradients  │                               │  • Computes gradient on  │
│  • Updates global model  │                               │    its partition         │
│  • Saves final results   │                               │  • Sends back to master  │
└──────────────────────────┘                               └──────────────────────────┘
         Parameter Server Pattern — Federated Gradient Averaging
```

---

## Setup Instructions

### Prerequisites (both machines)
```bash
python --version   # need 3.8+
pip install -r requirements.txt
```

---

## Step 1 — Generate Data (LOCAL machine only)

```bash
python generate_data.py
```

This creates:
- `data/full_dataset.csv` — 10,000 samples, 5 features (for single-machine run)
- `data/partition_0.csv` — 5,000 samples for your local machine
- `data/partition_1.csv` — 5,000 samples to copy to EC2

---

## Step 2 — Run Single Machine Baseline (LOCAL machine)

```bash
python single_machine.py
```

Note the training time and metrics. Results saved to `results/single_machine_results.json`.

---

## Step 3 — EC2 Setup

### 3a. Launch EC2 Instance
- AMI: Ubuntu 22.04 LTS (free tier eligible: t2.micro)
- Security Group: add inbound rule → **TCP port 9999**, source = your local IP

### 3b. Copy partition to EC2
```bash
# From your local machine:
scp -i your-key.pem data/partition_1.csv ubuntu@<EC2-PUBLIC-IP>:~/distributed_ml/data/
scp -i your-key.pem worker.py ubuntu@<EC2-PUBLIC-IP>:~/distributed_ml/
scp -i your-key.pem requirements.txt ubuntu@<EC2-PUBLIC-IP>:~/distributed_ml/
```

### 3c. Install dependencies on EC2
```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
cd distributed_ml
pip install -r requirements.txt
```

### 3d. Edit worker.py on EC2
Open `worker.py` and set:
```python
COORDINATOR_IP = "YOUR_LOCAL_MACHINE_PUBLIC_IP"
```
Find your local machine's public IP: https://ifconfig.me

---

## Step 4 — Run Distributed Training

**Terminal 1 — LOCAL machine (start coordinator first):**
```bash
python coordinator.py
# Output: "Listening for worker on port 9999 ..."
```

**Terminal 2 — EC2 instance (start worker after coordinator is listening):**
```bash
python worker.py
# Output: "Connecting to coordinator ..." then "Connected!"
```

Both terminals will show epoch logs. When training finishes, coordinator saves results.

---

## Step 5 — Compare Results

```bash
python compare_metrics.py
```

Prints side-by-side table and saves `results/metrics_comparison.png`.

---

## Metrics Explained

| Metric | Description | What to say in presentation |
|--------|-------------|----------------------------|
| **MSE** | Mean Squared Error — average squared difference between prediction and truth | Lower is better; compare single vs. distributed |
| **RMSE** | Root MSE — same units as target variable | More interpretable than MSE |
| **MAE** | Mean Absolute Error — average absolute difference | Robust to outliers |
| **R²** | Coefficient of determination (0–1) | How much variance the model explains; closer to 1 = better |
| **Training Time** | Wall-clock seconds | Key speedup metric |
| **Speedup Factor** | Single time / Distributed time | Should be > 1 for large data |

---

## Expected Results & Talking Points

**Why MSE may be slightly different (not identical):**
- Each node normalizes its own partition independently
- Gradient averaging introduces minor numerical differences
- This is a known trade-off in distributed ML — accuracy vs. speed

**Why distributed may not be faster on small data:**
- With 10,000 samples, network communication overhead can exceed compute savings
- In production, distributed ML shines at millions of samples
- Mention this honestly — it demonstrates you understand the trade-offs

**Communication overhead:**
- Each epoch: coordinator sends (FEATURES+1) × 8 bytes, receives the same back
- For 200 epochs, 5 features: ~16 KB total transferred — very lightweight
- Real distributed ML (neural nets) transfers MB per epoch, hence the speedup

---

## Demo Script (for live presentation)

```
1. Show both terminals open (local + SSH to EC2)
2. Show data/partition_0.csv and partition_1.csv — explain the split
3. Run: python single_machine.py   → note the time and metrics
4. Start coordinator.py in Terminal 1 — show "listening" message
5. Start worker.py in Terminal 2 (EC2) — show connection established
6. Watch both terminals update simultaneously — this is the key visual!
7. After training: python compare_metrics.py → show the table and chart
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Worker can't connect | Check EC2 security group has port 9999 open inbound |
| `Connection refused` | Make sure coordinator.py is running BEFORE worker.py |
| `FileNotFoundError` | Run generate_data.py first; check data/ folder exists |
| Slow training | Normal — t2.micro is low compute; mention it in presentation |
| Different MSE values | Expected — see "Talking Points" above |
