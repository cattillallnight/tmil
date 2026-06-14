"""
Step 23: Evaluate PG-EGAE on Tornado Cash Cashout Ground Truth (Test Set A)
===========================================================================
This script evaluates how well PG-EGAE localizes real TC cashout transactions
within the full transaction history of phisher accounts.

Logic:
1. Load TC cashout hits from step16 (the ground truth: tx hashes that went to TC).
2. Load full raw transactions for those specific accounts from CSVs.
3. Build sequence-to-graphs, tracking transaction hashes.
4. Pass each phisher account's graphs through PG-EGAE models.
5. Score all edges by MSE (reconstruction error).
6. Check the rank of confirmed TC cashout transactions among ALL txs.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR / "BERT4ETH" / "Data"
RESULTS_DIR = BASE_DIR / "tmil_eth" / "results"

PHISHER_TX_IN         = DATA_DIR / "phisher_transaction_in.csv"
PHISHER_TX_OUT        = DATA_DIR / "phisher_transaction_out.csv"
NORMAL_TX_IN          = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT         = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def sequence_to_graphs(tx_list, max_gap_hours=1.0):
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
        if len(burst) < 2: continue
        node_map = {}
        node_feats = [[1.0]]
        edges_src, edges_dst, edge_feats, edge_hashes = [], [], [], []
        prev_ts = burst[0][0]
        for t, c, v, is_in, h in burst:
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
            edge_hashes.append(h.lower())
            
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        graphs.append({'data': Data(x=x, edge_index=edge_index, edge_attr=edge_attr), 'hashes': edge_hashes})
    return graphs

def load_transactions(in_path, out_path, target_accounts):
    tx_history = defaultdict(list)
    def process_csv(path, is_inbound):
        print(f"  Reading {path.name}...")
        chunk_iter = pd.read_csv(path, chunksize=100000, header=None, low_memory=False)
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=[0, 5, 6, 7, 11])
            values = pd.to_numeric(chunk[7], errors='coerce') / 1e18
            timestamps = pd.to_numeric(chunk[11], errors='coerce')
            valid = values.notna() & timestamps.notna()
            
            hashes = chunk[0]
            addrs = chunk[6] if is_inbound else chunk[5]
            others = chunk[5] if is_inbound else chunk[6]
            
            for a, o, v, t, h in zip(addrs[valid], others[valid], values[valid], timestamps[valid], hashes[valid]):
                a = str(a).lower()
                if a in target_accounts:
                    tx_history[a].append((float(t), str(o).lower(), float(v), is_inbound, str(h)))
                    
    if in_path.exists(): process_csv(in_path, is_inbound=True)
    if out_path.exists(): process_csv(out_path, is_inbound=False)
    for a in tx_history:
        tx_history[a].sort(key=lambda x: x[0])
    return tx_history

def evaluate():
    print("Loading TC Ground Truth...")
    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)

    tc_gt = {}
    total_cashout_txs = 0
    for addr, txs in tc_hits.items():
        hashes = set(tx['hash'].lower() for tx in txs)
        tc_gt[addr.lower()] = hashes
        total_cashout_txs += len(hashes)

    target_accounts = set(tc_gt.keys())
    print(f"Loaded {len(target_accounts)} phisher accounts with TC cashouts.")
    print(f"Total TC cashout transactions: {total_cashout_txs}")

    print("\nReading raw transactions for these accounts...")
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    print(f"Found transactions for {len(tx_history)} accounts.")

    print("\nLoading cluster models...")
    models = []
    for c_id in range(4):
        model_path = RESULTS_DIR / 'checkpoints' / f'pg_gae_cluster_{c_id}.pt'
        if model_path.exists():
            m = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)
            
    metrics = {
        'total_cashouts_evaluated': 0,
        'hits_top_1': 0,
        'hits_top_5': 0,
        'hits_top_10': 0,
        'sum_ranks': 0
    }
    
    all_cashout_mses = []
    all_normal_mses = []

    print("\nEvaluating accounts...")
    for addr, txs in tx_history.items():
        graphs = sequence_to_graphs(txs)
        if not graphs: continue
        
        gt_hashes = tc_gt[addr]
        addr_tx_mses = {}
        
        for g_dict in graphs:
            data = g_dict['data'].to(DEVICE)
            hashes = g_dict['hashes']
            
            mse_sum = torch.zeros(data.edge_attr.size(0), device=DEVICE)
            with torch.no_grad():
                for model in models:
                    pred = model(data)
                    err = ((pred - data.edge_attr) ** 2).mean(dim=1)
                    mse_sum += err
            edge_mses = (mse_sum / len(models)).cpu().numpy()
            
            for h, mse in zip(hashes, edge_mses):
                addr_tx_mses[h] = float(mse)
                
        if not addr_tx_mses: continue
        
        ranked_txs = sorted(addr_tx_mses.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total_cashouts_evaluated'] += 1
                metrics['sum_ranks'] += rank
                if rank <= 1: metrics['hits_top_1'] += 1
                if rank <= 5: metrics['hits_top_5'] += 1
                if rank <= 10: metrics['hits_top_10'] += 1
                all_cashout_mses.append(addr_tx_mses[ch])
                
        for h, mse in addr_tx_mses.items():
            if h not in gt_hashes:
                all_normal_mses.append(mse)

    print("\n==================================================")
    print("TEST SET A: TORNADO CASH CASHOUT EVALUATION RESULTS")
    print("==================================================")
    n_v = metrics['total_cashouts_evaluated']
    if n_v > 0:
        print(f"TC Cashout Txs Evaluated: {n_v}")
        print(f"Rank@1  : {metrics['hits_top_1']} ({(metrics['hits_top_1']/n_v)*100:.2f}%)")
        print(f"Rank@5  : {metrics['hits_top_5']} ({(metrics['hits_top_5']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['hits_top_10']} ({(metrics['hits_top_10']/n_v)*100:.2f}%)")
        print(f"Mean Rank: {metrics['sum_ranks']/n_v:.2f}")
        
        mean_c_mse = np.mean(all_cashout_mses) if all_cashout_mses else 0
        mean_n_mse = np.mean(all_normal_mses) if all_normal_mses else 0
        
        print("\nMSE Separability:")
        print(f"Mean MSE of TC Cashout Txs : {mean_c_mse:.4f}")
        print(f"Mean MSE of Normal Txs     : {mean_n_mse:.4f}")
        if mean_n_mse > 0:
            print(f"Enrichment Ratio           : {mean_c_mse/mean_n_mse:.2f}x")
    else:
        print("No TC cashout txs were successfully mapped.")

if __name__ == '__main__':
    evaluate()
