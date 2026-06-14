"""
PG-EGAE Step 2: Dynamic Temporal Star-Graph Construction
===================================================
Goal: Convert raw transaction series into PyTorch Geometric `Data` objects.
Method:
- Split an account's history into "Bursts". A burst ends if gap > 1 hour.
- Build a Star-Graph for each Burst.
- Nodes: Target Account (is_target=1) and Counterparties (is_target=0).
- Edges: Transactions (Value, Time_Diff, Direction).
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR / "BERT4ETH" / "Data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PHISHER_TX_IN         = DATA_DIR / "phisher_transaction_in.csv"
PHISHER_TX_OUT        = DATA_DIR / "phisher_transaction_out.csv"
NORMAL_TX_IN          = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT         = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

COL_FROM      = 5
COL_TO        = 6
COL_VALUE     = 7
COL_TIMESTAMP = 11

def load_clusters():
    path = RESULTS_DIR / "pg_gae_step01_clusters.json"
    with open(path, "r") as f:
        return json.load(f)

def load_transactions(in_path, out_path, target_accounts):
    """ Loads all transactions for target_accounts into a dictionary of lists. """
    tx_history = defaultdict(list)
    
    def process_csv(path, is_inbound):
        print(f"  Reading {path.name}...")
        chunk_iter = pd.read_csv(path, chunksize=100000, header=None, low_memory=False)
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=[COL_FROM, COL_TO, COL_VALUE, COL_TIMESTAMP])
            values = pd.to_numeric(chunk[COL_VALUE], errors='coerce') / 1e18
            timestamps = pd.to_numeric(chunk[COL_TIMESTAMP], errors='coerce')
            valid = values.notna() & timestamps.notna()
            
            addrs = chunk[COL_TO] if is_inbound else chunk[COL_FROM]
            others = chunk[COL_FROM] if is_inbound else chunk[COL_TO]
            
            for a, o, v, t in zip(addrs[valid], others[valid], values[valid], timestamps[valid]):
                a = str(a).lower()
                if a in target_accounts:
                    # (timestamp, counterparty, value, is_inbound)
                    tx_history[a].append((float(t), str(o).lower(), float(v), is_inbound))
                    
    if in_path.exists(): process_csv(in_path, is_inbound=True)
    if out_path.exists(): process_csv(out_path, is_inbound=False)
    
    # Sort by timestamp
    for a in tx_history:
        tx_history[a].sort(key=lambda x: x[0])
    return tx_history

def build_graphs(tx_list, max_gap_hours=1.0):
    """
    Splits tx_list into bursts and builds a PyG Data object for each.
    """
    if not tx_list: return []
    
    bursts = []
    current_burst = [tx_list[0]]
    
    for tx in tx_list[1:]:
        gap = (tx[0] - current_burst[-1][0]) / 3600.0
        if gap > max_gap_hours:
            bursts.append(current_burst)
            current_burst = [tx]
        else:
            current_burst.append(tx)
    bursts.append(current_burst)
    
    graphs = []
    for burst in bursts:
        if len(burst) < 2: continue # Ignore isolated single transactions
        
        # Nodes mapping: 0 is target account
        node_map = {}
        node_feats = [[1.0]] # Target is 1
        
        edges_src = []
        edges_dst = []
        edge_feats = []
        
        prev_ts = burst[0][0]
        for t, c, v, is_in in burst:
            if c not in node_map:
                node_map[c] = len(node_feats)
                node_feats.append([0.0]) # Counterparty is 0
            
            n_idx = node_map[c]
            gap = (t - prev_ts) / 3600.0
            prev_ts = t
            
            # log features
            v_log = np.log1p(v)
            gap_log = np.log1p(gap)
            dir_feat = 1.0 if is_in else -1.0
            
            if is_in:
                edges_src.append(n_idx)
                edges_dst.append(0)
            else:
                edges_src.append(0)
                edges_dst.append(n_idx)
                
            edge_feats.append([v_log, gap_log, dir_feat])
            
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
    return graphs

def main():
    print("--- Phase 2: Dynamic Star-Graph Construction ---")
    data = load_clusters()
    
    # Subsample 5,000 normals per cluster for training to save space
    print("Selecting subset of accounts...")
    target_normals = set()
    np.random.seed(42)
    for c_id in range(4):
        addrs = [k for k, v in data["normal_groups"].items() if v == c_id]
        if len(addrs) > 5000:
            addrs = np.random.choice(addrs, 5000, replace=False).tolist()
        target_normals.update(addrs)
        print(f"  Cluster {c_id}: selected {len(addrs)} normals")
        
    target_phishers = set(data["phisher_groups"].keys())
    print(f"  Total Phishers to process: {len(target_phishers)}")
    
    # Process Phishers
    print("\nProcessing Phisher graphs...")
    phish_tx = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_phishers)
    phish_graphs = {}
    for a, txs in phish_tx.items():
        gs = build_graphs(txs)
        if gs: phish_graphs[a] = gs
    
    out_phish = RESULTS_DIR / "pg_gae_step02_phisher_graphs.pt"
    torch.save(phish_graphs, out_phish)
    print(f"  Saved {sum(len(v) for v in phish_graphs.values())} graphs for {len(phish_graphs)} phishers.")
    
    # Process Normals
    print("\nProcessing Normal graphs...")
    norm_tx = load_transactions(NORMAL_TX_IN, NORMAL_TX_OUT, target_normals)
    norm_graphs = {}
    for a, txs in norm_tx.items():
        gs = build_graphs(txs)
        if gs: norm_graphs[a] = gs
        
    out_norm = RESULTS_DIR / "pg_gae_step02_normal_graphs.pt"
    torch.save(norm_graphs, out_norm)
    print(f"  Saved {sum(len(v) for v in norm_graphs.values())} graphs for {len(norm_graphs)} normals.")

    print("\n[OK] Graph Construction Complete.")

if __name__ == "__main__":
    main()
