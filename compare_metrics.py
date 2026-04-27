"""
compare_metrics.py
━━━━━━━━━━━━━━━━━━
Run this AFTER both single_machine.py and distributed training are done.
It reads results/single_machine_results.json and results/distributed_results.json,
prints a side-by-side comparison, and saves charts to results/.

Usage:
    python compare_metrics.py
"""

import json
import os
import numpy as np

RESULTS_DIR = "results"

# ── Load results ──────────────────────────────────────────────────────────────
def load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def print_comparison(single, dist):
    print("\n" + "=" * 65)
    print(f"  {'METRIC':<28}  {'SINGLE MACHINE':>14}  {'DISTRIBUTED':>14}")
    print("=" * 65)

    rows = [
        ("Samples",        f"{single['samples']:,}",           f"{dist['total_samples']:,}"),
        ("Training time (s)", f"{single['training_time_sec']:.4f}", f"{dist['training_time_sec']:.4f}"),
        ("Final MSE",      f"{single['final_mse']:.4f}",       f"{dist['final_mse']:.4f}"),
        ("Final RMSE",     f"{single['final_rmse']:.4f}",      f"{dist['final_rmse']:.4f}"),
        ("Final MAE",      f"{single['final_mae']:.4f}",       f"{dist['final_mae']:.4f}"),
        ("R² Score",       f"{single['r2_score']:.4f}",        f"{dist['r2_score']:.4f}"),
    ]

    for label, sv, dv in rows:
        print(f"  {label:<28}  {sv:>14}  {dv:>14}")

    print("=" * 65)

    # Speedup / accuracy delta
    speedup = single['training_time_sec'] / dist['training_time_sec']
    mse_delta = abs(single['final_mse'] - dist['final_mse'])
    r2_delta  = abs(single['r2_score']  - dist['r2_score'])

    print(f"\n  Speedup factor    : {speedup:.2f}x")
    print(f"  MSE difference   : {mse_delta:.4f}  (lower = distributed stayed accurate)")
    print(f"  R² difference    : {r2_delta:.4f}")

    if speedup > 1:
        print(f"\n  ✓ Distributed training was {speedup:.2f}x faster.")
    else:
        print(f"\n  ✗ Single machine was faster (network overhead > compute gain).")
        print("    This is EXPECTED for small datasets — mention it in your presentation!")

    print()


def try_plot(single, dist):
    """Try to generate matplotlib charts. Skips gracefully if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping charts.")
        print("  Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Single Machine vs. Distributed Linear Regression", fontsize=14, fontweight="bold")

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    ax = axes[0]
    s_epochs = [h["epoch"] for h in single["loss_history"]]
    s_losses = [h["loss"]  for h in single["loss_history"]]
    d_epochs = [h["epoch"] for h in dist["loss_history"]]
    d_losses = [h["loss"]  for h in dist["loss_history"]]

    ax.plot(s_epochs, s_losses, "b-o", markersize=4, label="Single Machine")
    ax.plot(d_epochs, d_losses, "r-s", markersize=4, label="Distributed")
    ax.set_title("Training Loss (MSE) over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2. Bar chart — key metrics ────────────────────────────────────────────
    ax = axes[1]
    metrics_labels = ["MSE", "RMSE", "MAE"]
    s_vals = [single["final_mse"], single["final_rmse"], single["final_mae"]]
    d_vals = [dist["final_mse"],   dist["final_rmse"],   dist["final_mae"]]

    x = np.arange(len(metrics_labels))
    width = 0.35
    ax.bar(x - width/2, s_vals, width, label="Single Machine", color="#4a90d9")
    ax.bar(x + width/2, d_vals, width, label="Distributed",    color="#e74c3c")
    ax.set_title("Error Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Error Value")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Training time & R² ─────────────────────────────────────────────────
    ax = axes[2]
    categories = ["Training Time (s)", "R² Score"]
    s_perf = [single["training_time_sec"], single["r2_score"]]
    d_perf = [dist["training_time_sec"],   dist["r2_score"]]

    x = np.arange(len(categories))
    ax.bar(x - width/2, s_perf, width, label="Single Machine", color="#4a90d9")
    ax.bar(x + width/2, d_perf, width, label="Distributed",    color="#e74c3c")
    ax.set_title("Performance: Speed & Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {out}")
    plt.close()


def main():
    print("=" * 65)
    print("  METRICS COMPARISON  —  Single Machine vs. Distributed")
    print("=" * 65)

    # Check files exist
    missing = []
    for fname in ["single_machine_results.json", "distributed_results.json"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            missing.append(fname)
    if missing:
        print(f"\n  Missing result files: {missing}")
        print("  Run single_machine.py and the distributed training first.")
        return

    single = load("single_machine_results.json")
    dist   = load("distributed_results.json")

    print_comparison(single, dist)
    try_plot(single, dist)


if __name__ == "__main__":
    main()
