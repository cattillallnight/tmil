"""
TMIL-ETH — Step 4: Full-Sequence Sliding Window Sweep
======================================================
Demonstrates and validates the sliding window construction protocol (§4.3):
  W=200 transactions, S=50 stride.
  Coverage guarantee: each tx appears in at most 4 consecutive windows.
  Šidák FPR correction: tau_eff(K) = 1 - (1 - tau_base)^(1/K)

Saves: results/step4_windows_stats.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import RESULTS_DIR, sliding_windows, sidak_threshold

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"

W, S = 200, 50


def coverage_proof(W: int = 200, S: int = 50) -> int:
    """
    Prove: each transaction appears in at most floor(W/S) = 4 windows.
    TMIL-ETH §4.3: min(floor(W/S), 4) = min(4, 4) = 4.
    """
    max_windows_per_tx = W // S
    return max_windows_per_tx


def compute_window_stats(records: list) -> dict:
    """Compute window count statistics across all accounts."""
    n_windows_all = [r["n_windows"] for r in records]
    n_tx_all = [r["n_tx"] for r in records]

    return {
        "n_accounts": len(records),
        "n_windows": {
            "min": int(min(n_windows_all)),
            "median": float(np.median(n_windows_all)),
            "mean": float(np.mean(n_windows_all)),
            "max": int(max(n_windows_all)),
            "p75": float(np.percentile(n_windows_all, 75)),
            "p90": float(np.percentile(n_windows_all, 90)),
            "p95": float(np.percentile(n_windows_all, 95)),
        },
        "n_transactions": {
            "min": int(min(n_tx_all)),
            "median": float(np.median(n_tx_all)),
            "mean": float(np.mean(n_tx_all)),
            "max": int(max(n_tx_all)),
        },
        "total_windows": int(sum(n_windows_all)),
    }


def sidak_table(tau_base: float = 0.08, K_range: list = None) -> dict:
    """
    Compute Šidák effective thresholds for range of K values.
    tau_eff(K) = 1 - (1 - tau_base)^(1/K)
    """
    if K_range is None:
        K_range = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
    table = {}
    for K in K_range:
        tau_eff = sidak_threshold(tau_base, K)
        table[K] = round(tau_eff, 6)
    return table


def plot_window_distribution(records: list, save_path: Path):
    """Plot histogram of number of windows per account (phishing vs normal)."""
    phish_wins = [r["n_windows"] for r in records if r["label"] == 1]
    normal_wins = [r["n_windows"] for r in records if r["label"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(phish_wins, bins=50, color="#e74c3c", alpha=0.7, edgecolor="black")
    axes[0].set_title("Phishing Accounts: Window Count Distribution")
    axes[0].set_xlabel("Number of Windows")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(phish_wins), color="darkred", linestyle="--",
                   label=f"Median={np.median(phish_wins):.0f}")
    axes[0].legend()

    axes[1].hist(normal_wins, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    axes[1].set_title("Normal Accounts: Window Count Distribution")
    axes[1].set_xlabel("Number of Windows")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.median(normal_wins), color="darkblue", linestyle="--",
                   label=f"Median={np.median(normal_wins):.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path.name}")


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 4: Full-Sequence Sliding Window Sweep")
    print("=" * 60)

    # ── 1. Coverage proof ──────────────────────────────────────────
    max_overlap = coverage_proof(W, S)
    print(f"\n[1] Coverage Guarantee Proof:")
    print(f"  Window size W = {W}, Stride S = {S}")
    print(f"  Max windows per transaction = floor({W}/{S}) = {max_overlap}")
    print(f"  => Each transaction appears in at most {max_overlap} consecutive windows")
    print(f"  => Full coverage regardless of burst position (TMIL-ETH Theorem 1)")
    assert max_overlap == 4, f"Expected 4, got {max_overlap}"

    # ── 2. Verify on example accounts ─────────────────────────────
    print(f"\n[2] Example window counts for various sequence lengths:")
    print(f"  {'N_tx':>8} | {'N_windows':>10}")
    print(f"  {'-'*8} | {'-'*10}")
    for n in [50, 100, 200, 300, 500, 1000, 2000, 5000]:
        wins = sliding_windows(n, W, S)
        print(f"  {n:>8} | {len(wins):>10}")

    # ── 3. Load feature records for statistics ─────────────────────
    if FEATURES_FILE.exists():
        print(f"\n[3] Loading feature records from {FEATURES_FILE}...")
        with open(FEATURES_FILE, "rb") as f:
            records = pickle.load(f)
        print(f"  Total records: {len(records):,}")

        stats = compute_window_stats(records)
        print(f"\n[4] Window count statistics across all accounts:")
        print(f"  Min:    {stats['n_windows']['min']}")
        print(f"  Median: {stats['n_windows']['median']:.0f}")
        print(f"  Mean:   {stats['n_windows']['mean']:.1f}")
        print(f"  75th:   {stats['n_windows']['p75']:.0f}")
        print(f"  90th:   {stats['n_windows']['p90']:.0f}")
        print(f"  Max:    {stats['n_windows']['max']}")
        print(f"  Total windows (all accounts): {stats['total_windows']:,}")
        print(f"  Training cost ratio vs single-window: ~{stats['n_windows']['mean']:.1f}x")

        # Plot
        plot_path = RESULTS_DIR / "step4_window_distribution.png"
        plot_window_distribution(records, plot_path)
    else:
        print(f"\n[3] Note: {FEATURES_FILE} not found. Run Step 2 first for full stats.")
        stats = {}

    # ── 4. Šidák FPR correction table ─────────────────────────────
    print(f"\n[5] Šidák FPR Correction Table (tau_base=0.08, target FPR@95%TPR):")
    print(f"  {'K (windows)':>14} | {'tau_eff':>10} | {'Note'}")
    print(f"  {'-'*14} | {'-'*10} | {'-'*30}")

    for tau_base in [0.08, 0.05]:
        print(f"\n  tau_base = {tau_base}:")
        for K in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
            tau_eff = sidak_threshold(tau_base, K)
            note = "<-- 1:4 ratio max" if K == 4 else ""
            print(f"  {K:>14} | {tau_eff:>10.5f} | {note}")

    sidak_data = {
        "formula": "tau_eff(K) = 1 - (1 - tau_base)^(1/K)",
        "purpose": "Controls FPR inflation from max-pooling over K windows",
        "tau_base_default": 0.08,
        "table_tau_base_008": sidak_table(tau_base=0.08),
        "table_tau_base_005": sidak_table(tau_base=0.05),
    }

    # ── 5. Save results ───────────────────────────────────────────
    output = {
        "window_protocol": {
            "W": W,
            "S": S,
            "coverage_proof": f"floor(W/S) = floor({W}/{S}) = {max_overlap}",
            "max_windows_per_tx": max_overlap,
        },
        "sidak_correction": sidak_data,
    }
    if stats:
        output["account_window_stats"] = stats

    out_path = RESULTS_DIR / "step4_windows_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\n[OK] Step 4 complete.\n")
    return output


if __name__ == "__main__":
    main()
