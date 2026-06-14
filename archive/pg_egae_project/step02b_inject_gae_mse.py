"""
Step 02b: Inject PG-EGAE MSE into TMIL Features (Hybrid Approach)
=================================================================
This script computes the unsupervised anomaly score (MSE) from the PG-EGAE
models for every transaction in the TMIL dataset, and appends it as the 
5th feature to the `hand_crafted` array.

Process:
1. Load `step02_features.pkl`
2. Read raw CSVs to reconstruct full sequence for each account
3. Pass sequences through PG-EGAE to get MSE per edge
4. Map MSE back to original transaction index. Isolated txs get MSE = 0.0.
5. Save to `step02b_features_hybrid.pkl`
"""

import sys
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE
from utils import (
    DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT,
    NORMAL_TX_IN, NORMAL_TX_OUT
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_transactions(in_path, out_path, target_accounts, max_rows=None):
    tx_history = defaultdict(list)
    def process_csv(path, is_inbound):
        if not path.exists(): return
        print(f"  Reading {path.name}...")
        chunk_iter = pd.read_csv(path, chunksize=500000, header=None, low_memory=False)
        rows_read = 0
        for chunk in chunk_iter:
            if max_rows and rows_read >= max_rows: break
            chunk = chunk.dropna(subset=[5, 6, 7, 11])
            values = pd.to_numeric(chunk[7], errors='coerce') / 1e18
            timestamps = pd.to_numeric(chunk[11], errors='coerce')
            valid = values.notna() & timestamps.notna()
            
            addrs = chunk[6] if is_inbound else chunk[5]
            others = chunk[5] if is_inbound else chunk[6]
            
            for a, o, v, t in zip(addrs[valid], others[valid], values[valid], timestamps[valid]):
                a = str(a).lower()
                if a in target_accounts:
                    tx_history[a].append((float(t), str(o).lower(), float(v), is_inbound))
            rows_read += len(chunk)
            
    process_csv(in_path, is_inbound=True)
    process_csv(out_path, is_inbound=False)
    
    for a in tx_history:
        tx_history[a].sort(key=lambda x: x[0])
    return tx_history

def sequence_to_graphs_with_indices(tx_list, max_gap_hours=1.0):
    if not tx_list: return []
    bursts = []
    # tx_list is [ (t, o, v, is_in, orig_idx), ... ]
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
        if len(burst) < 2: continue
        node_map = {}
        node_feats = [[1.0]]
        edges_src, edges_dst, edge_feats, edge_indices = [], [], [], []
        prev_ts = burst[0][0]
        for t, c, v, is_in, orig_idx in burst:
            if c not in node_map:
                node_map[c] = len(node_feats)
                node_feats.append([0.0])
            n_idx = node_map[c]
            gap = (t - prev_ts) / 3600.0
            prev_ts = t
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
            edge_indices.append(orig_idx)
            
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        graphs.append({'data': Data(x=x, edge_index=edge_index, edge_attr=edge_attr), 'indices': edge_indices})
    return graphs

def main():
    print("Loading PG-EGAE Models...")
    models = []
    for c_id in range(4):
        p = RESULTS_DIR / 'checkpoints' / f'pg_gae_cluster_{c_id}.pt'
        if p.exists():
            m = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            m.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)
    print(f"Loaded {len(models)} models.")

    print("\nLoading step02_features.pkl...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
        
    phisher_addrs = set(r['address'] for r in records if r['label'] == 1)
    normal_addrs = set(r['address'] for r in records if r['label'] == 0)
    print(f"Total Phishers: {len(phisher_addrs)}, Total Normals: {len(normal_addrs)}")
    
    print("\nExtracting raw transactions for Phishers...")
    phish_txs = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, phisher_addrs)
    print("\nExtracting raw transactions for Normals...")
    norm_txs = load_transactions(NORMAL_TX_IN, NORMAL_TX_OUT, normal_addrs, max_rows=30_000_000)
    
    all_txs = {**phish_txs, **norm_txs}
    print(f"\nLoaded transactions for {len(all_txs)} accounts.")
    
    print("\nInjecting MSE features...")
    hybrid_records = []
    missing_txs = 0
    mismatch_txs = 0
    
    for r in tqdm(records):
        addr = r['address']
        n_tx_expected = r['n_tx']
        
        if addr not in all_txs:
            missing_txs += 1
            mse_col = np.zeros((n_tx_expected, 1), dtype=np.float32)
            r['hand_crafted'] = np.concatenate([r['hand_crafted'], mse_col], axis=1)
            hybrid_records.append(r)
            continue
            
        tx_list = all_txs[addr]
        if len(tx_list) != n_tx_expected:
            # Due to chunking/max_rows limits during step02 vs now, length might differ
            mismatch_txs += 1
            # Fallback: append 0s
            mse_col = np.zeros((n_tx_expected, 1), dtype=np.float32)
            r['hand_crafted'] = np.concatenate([r['hand_crafted'], mse_col], axis=1)
            hybrid_records.append(r)
            continue
            
        # Enumerate
        tx_list_enum = [(tx[0], tx[1], tx[2], tx[3], i) for i, tx in enumerate(tx_list)]
        graphs = sequence_to_graphs_with_indices(tx_list_enum)
        
        mse_scores = np.zeros(n_tx_expected, dtype=np.float32)
        
        if graphs:
            for g_dict in graphs:
                data = g_dict['data'].to(DEVICE)
                indices = g_dict['indices']
                
                mse_sum = torch.zeros(data.edge_attr.size(0), device=DEVICE)
                with torch.no_grad():
                    for model in models:
                        pred = model(data)
                        err = ((pred - data.edge_attr)**2).mean(dim=1)
                        mse_sum += err
                edge_mses = (mse_sum / len(models)).cpu().numpy()
                
                for idx, mse in zip(indices, edge_mses):
                    mse_scores[idx] = float(mse)
                    
        mse_col = mse_scores.reshape(-1, 1)
        r['hand_crafted'] = np.concatenate([r['hand_crafted'], mse_col], axis=1)
        hybrid_records.append(r)

    print(f"\nMissing txs: {missing_txs}, Mismatch txs: {mismatch_txs}")
    
    out_path = RESULTS_DIR / 'step02b_features_hybrid.pkl'
    print(f"Saving {len(hybrid_records)} hybrid records to {out_path}...")
    with open(out_path, 'wb') as f:
        pickle.dump(hybrid_records, f)
        
    print("\n[OK] Step 02b Complete.")

if __name__ == '__main__':
    main()
