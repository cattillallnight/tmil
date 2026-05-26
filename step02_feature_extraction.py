"""
TMIL-ETH — Step 2: Feature Extraction & Normalization
=======================================================
Builds feature vectors x_i = [C_i || h_i] for each transaction window.
  C_i: 4 hand-crafted features (z_amount, density, counterparty_novelty, value_ratio)
  h_i: 64-dim BERT4ETH embedding

Per-account normalization (not global) — captures relative anomaly.
Saves: results/step2_features.pkl
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import (
    load_phisher_accounts, load_embeddings, build_tx_sequences,
    compute_per_account_features, sliding_windows,
    DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT,
    NORMAL_TX_IN, NORMAL_TX_OUT
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampling limits to keep RAM manageable
# We take ALL phishers (7,067) and 4x normal (28,268) for 1:4 ratio
N_PHISHER_TARGET = 7068
N_NORMAL_TARGET  = 28272

# For normal accounts: scan TX_OUT to get FROM addresses
NORMAL_SCAN_ROWS = 20_000_000  # scan enough rows to find 28K+ accounts

W, S = 200, 50  # sliding window params


def get_normal_accounts(n_target: int, phisher_set: set, max_rows: int) -> list:
    """Derive normal account list from normal_tx_out FROM addresses."""
    print(f"  Scanning {NORMAL_TX_OUT.name} for {n_target:,} normal accounts...")
    candidates = set()
    with open(NORMAL_TX_OUT, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            parts = line.split(",")
            if len(parts) < 7:
                continue
            addr = parts[5].strip().lower()
            if addr and len(addr) == 42 and addr.startswith("0x"):
                candidates.add(addr)

    candidates -= phisher_set
    candidates = list(candidates)
    print(f"  Found {len(candidates):,} unique normal addresses (after phisher exclusion)")
    if len(candidates) > n_target:
        import random
        random.seed(42)
        candidates = random.sample(candidates, n_target)
    print(f"  Using {len(candidates):,} normal accounts")
    return candidates


def build_account_dataset(accounts: list, labels: dict,
                          tx_in_path: Path, tx_out_path: Path,
                          tag: str, max_rows: int = None) -> list:
    """
    Build per-account feature records.
    Returns list of dicts with keys: address, label, windows, n_tx
    """
    account_set = set(accounts)
    print(f"\n  Building sequences for {tag} accounts (n={len(accounts):,})...")
    txs = build_tx_sequences(tx_in_path, tx_out_path, account_set, max_rows_per_file=max_rows)

    records = []
    no_tx = 0
    for addr in tqdm(accounts, desc=f"  Features [{tag}]"):
        if addr not in txs:
            no_tx += 1
            continue
        tx_list = txs[addr]  # sorted by timestamp
        if len(tx_list) < 3:
            no_tx += 1
            continue

        feats = compute_per_account_features(tx_list)
        z_amount = feats["z_amount"]
        density  = feats["density"]
        novelty  = feats["counterparty_novelty"]
        v_ratio  = feats["value_ratio"]

        hand_crafted = np.stack([z_amount, density, novelty, v_ratio], axis=1)  # (n, 4)

        wins = sliding_windows(len(tx_list), W=W, S=S)
        records.append({
            "address": addr,
            "label": labels[addr],
            "n_tx": len(tx_list),
            "n_windows": len(wins),
            "hand_crafted": hand_crafted,  # (n_tx, 4)
            "windows": wins,               # list of (start, end)
        })

    print(f"  Done. Records: {len(records)}, no-tx/short: {no_tx}")
    return records


def attach_bert_embeddings(records: list, emb_matrix: np.ndarray,
                           addr2idx: dict) -> list:
    """
    Attach BERT4ETH account-level embedding h (shape 64) to each record.
    Note: BERT4ETH produces one embedding per account (not per transaction).
    We use this as a fixed context vector appended to each transaction's hand-crafted features.
    For accounts not in embedding index, we use zero vector.
    """
    print("\n  Attaching BERT4ETH embeddings...")
    emb_dim = emb_matrix.shape[1]
    missing = 0
    for rec in records:
        addr = rec["address"]
        idx = addr2idx.get(addr, None)
        if idx is not None:
            rec["bert_embedding"] = emb_matrix[idx]  # shape (64,)
        else:
            rec["bert_embedding"] = np.zeros(emb_dim, dtype=np.float32)
            missing += 1

    coverage = (len(records) - missing) / len(records) * 100 if records else 0
    print(f"  Embedding coverage: {len(records)-missing}/{len(records)} ({coverage:.1f}%)")
    print(f"  Missing embeddings (zero-filled): {missing}")
    return records


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 2: Feature Extraction & Normalization")
    print("=" * 60)

    # Load phishers
    phisher_accounts = load_phisher_accounts()
    phisher_set = set(a.lower() for a in phisher_accounts)
    # Use all 7,067 phishers
    phisher_list = [a.lower() for a in phisher_accounts]
    print(f"\nPhisher accounts: {len(phisher_list):,}")

    # Load embeddings
    print("\nLoading BERT4ETH embeddings...")
    emb_matrix, addr2idx = load_embeddings()
    print(f"  Embedding matrix: {emb_matrix.shape} ({emb_matrix.dtype})")
    print(f"  Address index size: {len(addr2idx):,}")

    # Labels
    labels = {}
    for a in phisher_list:
        labels[a] = 1

    # Build phisher records
    phisher_records = build_account_dataset(
        phisher_list, labels,
        PHISHER_TX_IN, PHISHER_TX_OUT,
        "phishing"
    )

    # Get normal accounts
    normal_list = get_normal_accounts(N_NORMAL_TARGET, phisher_set, NORMAL_SCAN_ROWS)
    for a in normal_list:
        labels[a] = 0

    # Build normal records (scan up to 30M rows of normal TX for feature coverage)
    normal_records = build_account_dataset(
        normal_list, labels,
        NORMAL_TX_IN, NORMAL_TX_OUT,
        "normal",
        max_rows=30_000_000
    )

    all_records = phisher_records + normal_records
    print(f"\nTotal records: {len(all_records):,}")
    print(f"  Phishing: {sum(1 for r in all_records if r['label']==1):,}")
    print(f"  Normal:   {sum(1 for r in all_records if r['label']==0):,}")

    # Attach BERT embeddings
    all_records = attach_bert_embeddings(all_records, emb_matrix, addr2idx)

    # Statistics
    n_tx_all = [r["n_tx"] for r in all_records]
    n_win_all = [r["n_windows"] for r in all_records]
    print(f"\nSequence length stats:")
    print(f"  Median n_tx:     {np.median(n_tx_all):.0f}")
    print(f"  Mean n_tx:       {np.mean(n_tx_all):.1f}")
    print(f"  Median n_windows:{np.median(n_win_all):.0f}")
    print(f"  Mean n_windows:  {np.mean(n_win_all):.1f}")
    print(f"  Max n_windows:   {max(n_win_all)}")

    # Save
    out_path = RESULTS_DIR / "step2_features.pkl"
    print(f"\nSaving features to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(all_records, f)

    # Also save summary stats
    summary = {
        "n_total":   len(all_records),
        "n_phishing": sum(1 for r in all_records if r["label"]==1),
        "n_normal":  sum(1 for r in all_records if r["label"]==0),
        "embedding_dim": int(emb_matrix.shape[1]),
        "hand_crafted_dim": 4,
        "feature_dim_total": int(emb_matrix.shape[1]) + 4,
        "window_W": W,
        "window_S": S,
        "median_n_tx": float(np.median(n_tx_all)),
        "mean_n_tx": float(np.mean(n_tx_all)),
        "median_n_windows": float(np.median(n_win_all)),
        "mean_n_windows": float(np.mean(n_win_all)),
        "max_n_windows": int(max(n_win_all)) if n_win_all else 0,
    }
    import json
    with open(RESULTS_DIR / "step2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Step 2 complete.\n")
    return all_records


if __name__ == "__main__":
    main()
