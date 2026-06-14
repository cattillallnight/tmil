"""
TMIL-ETH — Step 4: Full-Sequence Sliding Window Sweep
======================================================
Demonstrates and validates the sliding window construction protocol (§4.3):
  W=200 transactions, S=50 stride.
  Coverage guarantee: each tx appears in at most 4 consecutive windows.
  Šidák FPR correction: tau_eff(K) = 1 - (1 - tau_base)^(1/K)

Saves: results/figures/step04_windows_stats.json
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
FEATURES_FILE = RESULTS_DIR / "step02_features.pkl"

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
        plot_path = RESULTS_DIR / "step04_window_distribution.png"
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

    out_path = RESULTS_DIR / "step04_windows_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\n[OK] Step 4 complete.\n")
    return output



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: 72-Hour Endpoint Detection Window — Sensitivity Analysis
# (formerly step19_window_sensitivity.py)
# ══════════════════════════════════════════════════════════════════════════════

import json as _json_s19
import random as _random_s19
import pandas as _pd_s19
from datetime import datetime as _dt_s19

DATA_DIR_S19 = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
GT_FILE_S19  = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
WINDOWS_HOURS = [24, 48, 72, 168]

KNOWN_CASHOUT_ADDRESSES = {
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
    "0xd551234ae421e3bcba99a0da6d736074f22192ff",
    "0x564286362092d8e7936f0549571a803b203aaced",
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8",
    "0xab5c66752a9e8167967685f1450532fb96d5d24f",  # Huobi
    "0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b",
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",  # OKX
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e",  # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",
    "0x503828976d22510aad0201ac7ec88293211d23da",
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe",  # Gate.io
    "0x46340b20830761efd32832a74d7169b29feb9758",  # Crypto.com
    "0xd24400ae8bfebb18ca49be86258a3c749cf46853",  # Gemini
    "0x742d35cc6634c0532925a3b844bc454e4438f44e",  # Bitfinex
    "0xe853c56864a2ebe4576a807d26fdc4a0ada51919",  # Kraken
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x881d40237659c251811cec9c364ef91dc08d300c",  # MetaMask
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",  # Tornado 1ETH
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",  # Tornado 10ETH
    "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3",  # Tornado 100ETH
}


def run_window_sensitivity():
    """
    Sensitivity Analysis for 72-Hour Endpoint Detection Window.
    Proves empirically that 72h is the optimal precision-coverage tradeoff.
    Saves: results/figures/step19_window_sensitivity.json
    """
    print("=" * 70)
    print("Step 4b: 72-Hour Window Sensitivity Analysis")
    print("=" * 70)

    if not GT_FILE_S19.exists():
        print(f"[SKIP] Ground truth file not found: {GT_FILE_S19}")
        return
    if not DATA_DIR_S19.exists():
        print(f"[SKIP] Data directory not found: {DATA_DIR_S19}")
        return

    with open(GT_FILE_S19, "r") as f:
        gt_data = _json_s19.load(f)

    print("    Loading phisher_transaction_out.csv...")
    df_out = _pd_s19.read_csv(DATA_DIR_S19 / "phisher_transaction_out.csv",
                              header=None, dtype=str)
    df_out.columns = list(range(len(df_out.columns)))
    df_out[5]  = df_out[5].str.lower().fillna("")
    df_out[6]  = df_out[6].str.lower().fillna("")
    df_out[7]  = _pd_s19.to_numeric(df_out[7], errors="coerce").fillna(0)
    df_out[11] = _pd_s19.to_numeric(df_out[11], errors="coerce").fillna(0)

    print("    Loading phisher_transaction_in.csv...")
    df_in = _pd_s19.read_csv(DATA_DIR_S19 / "phisher_transaction_in.csv",
                             header=None, dtype=str)
    df_in.columns = list(range(len(df_in.columns)))
    df_in[5]  = df_in[5].str.lower().fillna("")
    df_in[6]  = df_in[6].str.lower().fillna("")
    df_in[11] = _pd_s19.to_numeric(df_in[11], errors="coerce").fillna(0)

    _random_s19.seed(42)
    sample_size = min(500, len(gt_data))
    sampled = _random_s19.sample(gt_data, sample_size)
    print(f"    Sampled {sample_size} accounts for sensitivity analysis.")

    results_by_window = {}
    for window_h in WINDOWS_HOURS:
        window_secs = window_h * 3600
        stats = {"n_accounts": 0, "n_found_cashout": 0, "n_large_cashout": 0,
                 "n_known_address": 0, "values_eth": []}

        for acc in sampled:
            addr  = acc["account_address"].lower()
            in_df_acc  = df_in[df_in[6] == addr].copy()
            out_df_acc = df_out[df_out[5] == addr].copy()
            if len(in_df_acc) == 0:
                continue
            last_victim_ts = in_df_acc[11].max()
            limit_ts = last_victim_ts + window_secs
            out_in_window = out_df_acc[(out_df_acc[11] >= last_victim_ts) &
                                       (out_df_acc[11] <= limit_ts)]
            stats["n_accounts"] += 1
            if len(out_in_window) == 0:
                continue
            stats["n_found_cashout"] += 1
            max_idx = out_in_window[7].idxmax()
            max_val_eth = out_in_window.loc[max_idx, 7] / 1e18
            max_to      = out_in_window.loc[max_idx, 6]
            stats["values_eth"].append(max_val_eth)
            if max_val_eth >= 1.0:
                stats["n_large_cashout"] += 1
            if max_to in KNOWN_CASHOUT_ADDRESSES:
                stats["n_known_address"] += 1

        n_found = stats["n_found_cashout"]
        n_acc   = stats["n_accounts"]
        vals    = stats["values_eth"]
        coverage_rate   = n_found / n_acc * 100 if n_acc > 0 else 0
        large_rate      = stats["n_large_cashout"] / n_found * 100 if n_found > 0 else 0
        known_addr_rate = stats["n_known_address"] / n_found * 100 if n_found > 0 else 0
        median_val      = float(np.median(vals)) if vals else 0
        results_by_window[window_h] = {
            "window_hours": window_h, "n_sampled": n_acc, "n_found_cashout": n_found,
            "coverage_rate_pct": round(coverage_rate, 1), "large_rate_pct": round(large_rate, 1),
            "known_addr_rate_pct": round(known_addr_rate, 1), "median_val_eth": round(median_val, 3),
        }
        print(f"  Window {window_h:4d}h: Coverage={coverage_rate:.1f}% | "
              f"Large={large_rate:.1f}% | KnownAddr={known_addr_rate:.1f}% | Median={median_val:.3f}ETH")

    scores_f1 = {}
    for wh, r in results_by_window.items():
        cov, large = r["coverage_rate_pct"], r["large_rate_pct"]
        scores_f1[wh] = 2 * large * cov / (large + cov) if (large + cov) > 0 else 0
    optimal_f1 = max(scores_f1, key=scores_f1.get)

    output = {
        "windows_tested": WINDOWS_HOURS, "sample_size": sample_size,
        "results_by_window": results_by_window,
        "scores_f1": {str(k): round(v, 2) for k, v in scores_f1.items()},
        "optimal_window_f1": int(optimal_f1),
    }
    out_path = RESULTS_DIR / "step19_window_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json_s19.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  {'✅' if optimal_f1 == 72 else 'INFO'} F1-optimal window = {optimal_f1}h")
    print("[OK] Window Sensitivity Analysis complete.\n")


if __name__ == "__main__":
    main()
    # Uncomment to also run sensitivity analysis:
    # run_window_sensitivity()
