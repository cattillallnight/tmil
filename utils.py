"""
TMIL-ETH — Shared Utilities
=============================
Common functions used across all 10 steps:
  - Data loading helpers
  - Feature computation
  - Transaction sequence building
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# ─── Path configuration ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
EMBED_DIR   = DATA_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

PHISHER_ACCOUNTS_FILE = DATA_DIR / "phisher_account.txt"
PHISHER_TX_IN         = DATA_DIR / "phisher_transaction_in.csv"
PHISHER_TX_OUT        = DATA_DIR / "phisher_transaction_out.csv"
NORMAL_TX_IN          = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT         = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

EMBED_FILE   = EMBED_DIR / "embedding_bert4eth_exp_104000.npy"
ADDRESS_FILE = EMBED_DIR / "address_bert4eth_exp_104000.npy"

# Column indices (matching gen_seq.py HEADER):
# hash,nonce,block_hash,block_number,tx_index,from_address,to_address,
# value,gas,gas_price,input,block_timestamp,...
COL_FROM      = 5
COL_TO        = 6
COL_VALUE     = 7   # Wei (int)
COL_TIMESTAMP = 11


# ─── Account loading ──────────────────────────────────────────────────────────

def load_phisher_accounts() -> List[str]:
    """Load list of phisher addresses (lowercase)."""
    with open(PHISHER_ACCOUNTS_FILE, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load BERT4ETH pre-computed embeddings.
    Returns (embedding_matrix, address_to_idx).
    embedding_matrix: shape (N, 64) — float32
    """
    emb = np.load(EMBED_FILE)
    addrs = np.load(ADDRESS_FILE, allow_pickle=True)
    addr2idx = {str(a).lower(): i for i, a in enumerate(addrs)}
    return emb, addr2idx


# ─── Transaction sequence builder ────────────────────────────────────────────

def build_tx_sequences(
    tx_in_path: Path,
    tx_out_path: Path,
    account_set: Set[str],
    max_rows_per_file: int = None,
) -> Dict[str, List[Tuple]]:
    """
    Build per-account transaction sequences.
    Returns: {addr_lower: [(timestamp, value_eth, direction, counterpart)]}
    Sorted by timestamp (ascending).
    """
    txs: Dict[str, List] = defaultdict(list)

    def _read(path, direction):
        rows = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if max_rows_per_file and rows >= max_rows_per_file:
                    break
                parts = line.rstrip("\n").split(",")
                if len(parts) < 12:
                    continue
                try:
                    from_addr = parts[COL_FROM].strip().lower()
                    to_addr   = parts[COL_TO].strip().lower()
                    value_wei = int(parts[COL_VALUE])
                    ts        = int(parts[COL_TIMESTAMP])
                except (ValueError, IndexError):
                    continue
                if not from_addr or not to_addr:
                    continue
                value_eth = value_wei / 1e18
                if direction == "OUT" and from_addr in account_set:
                    txs[from_addr].append((ts, value_eth, "OUT", to_addr))
                    rows += 1
                elif direction == "IN" and to_addr in account_set:
                    txs[to_addr].append((ts, value_eth, "IN", from_addr))
                    rows += 1

    _read(tx_out_path, "OUT")
    _read(tx_in_path, "IN")

    # Sort by timestamp
    for addr in txs:
        txs[addr].sort(key=lambda x: x[0])

    return dict(txs)


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_per_account_features(txs_sorted: List[Tuple]) -> Dict[str, np.ndarray]:
    """
    Compute 4 hand-crafted features per transaction (TMIL-ETH Section 3.1 & 4.2):
      z_amount_i          — per-account normalized amount
      density_i           — local density within 1-hour window
      counterparty_novelty_i — 1 if counterpart first seen
      value_ratio_i       — running in/out ratio

    Returns dict of feature arrays, each shape (n_txs,).
    """
    n = len(txs_sorted)
    if n == 0:
        return {k: np.zeros(0) for k in
                ["z_amount", "density", "counterparty_novelty", "value_ratio"]}

    timestamps   = np.array([t[0] for t in txs_sorted], dtype=np.float64)
    values_eth   = np.array([t[1] for t in txs_sorted], dtype=np.float64)
    directions   = [t[2] for t in txs_sorted]
    counterparts = [t[3] for t in txs_sorted]

    # ── z_amount_i: per-account normalization (TMIL-ETH §4.2) ────
    mu    = values_eth.mean()
    sigma = values_eth.std() + 1e-9
    z_amount = np.clip((values_eth - mu) / sigma, -3.0, 3.0) / 3.0

    # ── density_i: transactions in ±1800s window ─────────────────
    density = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = timestamps[i] - 1800.0
        hi = timestamps[i] + 1800.0
        density[i] = float(np.sum((timestamps >= lo) & (timestamps <= hi)))

    # ── counterparty_novelty_i ────────────────────────────────────
    seen = set()
    novelty = np.zeros(n, dtype=np.float64)
    for i, cp in enumerate(counterparts):
        if cp not in seen:
            novelty[i] = 1.0
            seen.add(cp)

    # ── value_ratio_i: running IN / (OUT + eps) ───────────────────
    cum_in = cum_out = 0.0
    value_ratio = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if directions[i] == "IN":
            cum_in += values_eth[i]
        else:
            cum_out += values_eth[i]
        value_ratio[i] = cum_in / (cum_out + 1e-9)

    return {
        "z_amount":              z_amount,
        "density":               density,
        "counterparty_novelty":  novelty,
        "value_ratio":           value_ratio,
    }


def bag_purity_proxy(density: np.ndarray, value_ratio: np.ndarray) -> float:
    """
    §4.4: bag_purity_proxy = |txs: density_i > p75 AND value_ratio_i > 1| / |total txs|
    """
    n = len(density)
    if n == 0:
        return 0.0
    p75 = np.percentile(density, 75)
    burst = np.sum((density > p75) & (value_ratio > 1.0))
    return float(burst) / n


# ─── Sliding window construction ──────────────────────────────────────────────

def sliding_windows(seq_len: int, W: int = 200, S: int = 50) -> List[Tuple[int, int]]:
    """
    Generate (start, end) index pairs for sliding windows.
    §4.3: W=200, S=50. If seq_len < W, returns one window covering all.
    """
    if seq_len <= W:
        return [(0, seq_len)]
    windows = []
    start = 0
    while start < seq_len:
        end = min(start + W, seq_len)
        windows.append((start, end))
        if end == seq_len:
            break
        start += S
    return windows


def sidak_threshold(tau_base: float, K: int) -> float:
    """
    §4.3 Šidák FPR correction:
      τ_eff(K) = 1 − (1 − τ_base)^(1/K)
    """
    if K <= 1:
        return tau_base
    return 1.0 - (1.0 - tau_base) ** (1.0 / K)
