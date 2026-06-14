"""
PG-EGAE Step 1: Peer-Group Clustering
===================================================
Goal: Extract macroscopic lifetime features for all accounts and cluster them into Peer Groups.
This solves the baseline-conditioning problem.

Macro-features:
1. Total Transaction Count
2. Total ETH Volume (In + Out)
3. Active Lifespan (Max Timestamp - Min Timestamp in hours)
4. Max Single Transaction Value

Method:
- K-Means clustering on Normal accounts to define K baselines.
- Assign Phisher accounts to the nearest Normal cluster.
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR / "BERT4ETH" / "Data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHISHER_ACCOUNTS_FILE = DATA_DIR / "phisher_account.txt"
PHISHER_TX_IN         = DATA_DIR / "phisher_transaction_in.csv"
PHISHER_TX_OUT        = DATA_DIR / "phisher_transaction_out.csv"
NORMAL_TX_IN          = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT         = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

COL_FROM      = 5
COL_TO        = 6
COL_VALUE     = 7   # Wei
COL_TIMESTAMP = 11

def load_phishers():
    with open(PHISHER_ACCOUNTS_FILE, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

def extract_macro_features(tx_in_path, tx_out_path, valid_accounts):
    """
    Reads raw CSVs and computes macro features:
    {address: [total_tx, total_vol_eth, lifespan_hours, max_tx_eth]}
    """
    stats = defaultdict(lambda: {"tx_count": 0, "total_vol": 0.0, "max_val": 0.0, "min_ts": float('inf'), "max_ts": 0.0})
    
    def process_csv(path, is_inbound):
        print(f"Reading {path.name}...")
        # Read in chunks to save memory
        chunk_iter = pd.read_csv(path, chunksize=100000, header=None, low_memory=False)
        for chunk in chunk_iter:
            # Drop malformed rows
            chunk = chunk.dropna(subset=[COL_FROM, COL_TO, COL_VALUE, COL_TIMESTAMP])
            
            # Convert
            try:
                values = pd.to_numeric(chunk[COL_VALUE], errors='coerce') / 1e18
                timestamps = pd.to_numeric(chunk[COL_TIMESTAMP], errors='coerce')
                
                # Filter valid
                valid_mask = values.notna() & timestamps.notna()
                chunk = chunk[valid_mask]
                values = values[valid_mask]
                timestamps = timestamps[valid_mask]
            except:
                continue
                
            addrs = chunk[COL_TO] if is_inbound else chunk[COL_FROM]
            addrs = addrs.astype(str).str.lower()
            
            for addr, val, ts in zip(addrs, values, timestamps):
                if valid_accounts is not None and addr not in valid_accounts:
                    continue
                
                st = stats[addr]
                st["tx_count"] += 1
                st["total_vol"] += float(val)
                if val > st["max_val"]: st["max_val"] = float(val)
                if ts < st["min_ts"]: st["min_ts"] = float(ts)
                if ts > st["max_ts"]: st["max_ts"] = float(ts)

    if tx_in_path.exists(): process_csv(tx_in_path, is_inbound=True)
    if tx_out_path.exists(): process_csv(tx_out_path, is_inbound=False)
    
    # Finalize features
    features = {}
    for addr, st in stats.items():
        if st["tx_count"] == 0: continue
        lifespan_hours = (st["max_ts"] - st["min_ts"]) / 3600.0
        features[addr] = [
            np.log1p(st["tx_count"]),
            np.log1p(st["total_vol"]),
            np.log1p(lifespan_hours),
            np.log1p(st["max_val"])
        ]
    return features

def main():
    print("--- Phase 1: Macro-Feature Extraction ---")
    phisher_set = load_phishers()
    
    print("Extracting Normal Accounts...")
    normal_feats = extract_macro_features(NORMAL_TX_IN, NORMAL_TX_OUT, None)
    # Remove phishers from normal set if any leaked
    normal_feats = {k: v for k, v in normal_feats.items() if k not in phisher_set}
    
    print("Extracting Phisher Accounts...")
    phisher_feats = extract_macro_features(PHISHER_TX_IN, PHISHER_TX_OUT, phisher_set)
    
    print(f"Found {len(normal_feats)} normal accounts and {len(phisher_feats)} phisher accounts.")
    
    # Clustering
    print("\n--- Clustering Normal Accounts (K-Means) ---")
    addrs_norm = list(normal_feats.keys())
    X_norm = np.array([normal_feats[a] for a in addrs_norm])
    
    scaler = StandardScaler()
    X_norm_scaled = scaler.fit_transform(X_norm)
    
    K = 4 # E.g., Whales, Active, Casual, Newbies
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_norm = kmeans.fit_predict(X_norm_scaled)
    
    # Profile clusters
    print("Cluster Profiles (Mean unscaled features):")
    for i in range(K):
        mask = (labels_norm == i)
        means = np.expm1(X_norm[mask].mean(axis=0)) # expm1 to invert log1p
        print(f"  Cluster {i} (N={mask.sum()}): Tx={means[0]:.1f}, Vol={means[1]:.2f} ETH, Life={means[2]:.1f}h, MaxTx={means[3]:.2f} ETH")
    
    # Map phishers
    print("\n--- Mapping Phishers to Peer Groups ---")
    addrs_phish = list(phisher_feats.keys())
    X_phish = np.array([phisher_feats[a] for a in addrs_phish])
    X_phish_scaled = scaler.transform(X_phish)
    labels_phish = kmeans.predict(X_phish_scaled)
    
    for i in range(K):
        mask = (labels_phish == i)
        print(f"  Phishers mapped to Cluster {i}: {mask.sum()}")
    
    # Save
    out_data = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "kmeans_centers": kmeans.cluster_centers_.tolist(),
        "normal_groups": {a: int(l) for a, l in zip(addrs_norm, labels_norm)},
        "phisher_groups": {a: int(l) for a, l in zip(addrs_phish, labels_phish)},
        "normal_feats": {a: f for a, f in normal_feats.items()},
        "phisher_feats": {a: f for a, f in phisher_feats.items()}
    }
    
    out_file = RESULTS_DIR / "pg_gae_step01_clusters.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f)
        
    print(f"\n[OK] Clustering complete. Saved to {out_file.name}")

if __name__ == "__main__":
    main()
