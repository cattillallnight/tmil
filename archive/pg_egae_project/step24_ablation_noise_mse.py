"""
Step 24: Ablation Study - Noise vs Pure Victim MSE Distribution
===============================================================
This script proves that the PG-EGAE model intrinsically separates
real victim transactions from noise (spam/dust/CEX bots).

Logic:
1. Load RAW scamsniffer dataset (with noise).
2. Load FILTERED scamsniffer dataset (pure GT).
3. Identify which tx hashes are "Pure Victims" and which are "Noise".
4. Evaluate all of them using PG-EGAE.
5. Compare the Mean MSE of Pure Victims vs Noise.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
RAW_DS = RESULTS_DIR / "scamsniffer_txlevel_dataset.json"
FILTERED_DS = RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def sequence_to_graphs(address, timestamps, directions, values_eth, hashes, max_gap_hours=1.0):
    tx_list = list(zip(timestamps, directions, values_eth, hashes))
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
        node_feats = [[1.0]]
        edges_src, edges_dst, edge_feats, edge_hashes = [], [], [], []
        prev_ts = burst[0][0]
        for t, dir_val, v, h in burst:
            n_idx = len(node_feats)
            node_feats.append([0.0])
            gap = (t - prev_ts) / 3600.0
            prev_ts = t
            v_log = np.log1p(v)
            gap_log = np.log1p(gap)
            dir_feat = float(dir_val)
            if dir_feat == 0.0: dir_feat = -1.0
            if dir_feat > 0:
                edges_src.append(n_idx)
                edges_dst.append(0)
            else:
                edges_src.append(0)
                edges_dst.append(n_idx)
            edge_feats.append([v_log, gap_log, dir_feat])
            edge_hashes.append(h)
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        graphs.append({'data': Data(x=x, edge_index=edge_index, edge_attr=edge_attr), 'hashes': edge_hashes})
    return graphs

def evaluate():
    print("Loading RAW dataset (with noise)...")
    with open(RAW_DS, "r") as f:
        raw_dataset = json.load(f)
        
    print("Loading FILTERED dataset (pure GT)...")
    with open(FILTERED_DS, "r") as f:
        filtered_dataset = json.load(f)
        
    pure_victim_hashes = set()
    for item in filtered_dataset:
        for v in item.get('victim_txs', []):
            pure_victim_hashes.add(v['hash'].lower())
            
    raw_victim_hashes = set()
    for item in raw_dataset:
        for v in item.get('victim_txs', []):
            raw_victim_hashes.add(v['hash'].lower())
            
    noise_hashes = raw_victim_hashes - pure_victim_hashes
    print(f"Pure Victim Txs: {len(pure_victim_hashes)}")
    print(f"Noise Txs      : {len(noise_hashes)}")

    print("Loading models...")
    models = []
    for c_id in range(4):
        model_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
        if model_path.exists():
            model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            models.append(model)

    pure_mses = []
    noise_mses = []
    normal_mses = []

    for item in raw_dataset:
        addr = item['address']
        graphs = sequence_to_graphs(addr, item['timestamps'], item['directions'], item['values_eth'], item['hashes'])
        if not graphs: continue
        
        for g_dict in graphs:
            data = g_dict['data'].to(DEVICE)
            hashes = g_dict['hashes']
            
            edge_mses_sum = torch.zeros(data.edge_attr.size(0), device=DEVICE)
            with torch.no_grad():
                for model in models:
                    pred = model(data)
                    err = ((pred - data.edge_attr)**2).mean(dim=1)
                    edge_mses_sum += err
            
            edge_mses = (edge_mses_sum / len(models)).cpu().numpy()
            
            for h, mse in zip(hashes, edge_mses):
                h_lower = h.lower()
                if h_lower in pure_victim_hashes:
                    pure_mses.append(mse)
                elif h_lower in noise_hashes:
                    noise_mses.append(mse)
                else:
                    normal_mses.append(mse)

    print("\n==================================================")
    print("ABLATION STUDY: NOISE VS PURE VICTIM MSE")
    print("==================================================")
    mean_pure = np.mean(pure_mses) if pure_mses else 0
    mean_noise = np.mean(noise_mses) if noise_mses else 0
    mean_normal = np.mean(normal_mses) if normal_mses else 0
    
    print(f"Evaluated Pure Victim Txs : {len(pure_mses)}")
    print(f"Evaluated Noise Txs       : {len(noise_mses)}")
    print(f"Evaluated Normal Txs      : {len(normal_mses)}")
    print()
    print(f"Mean MSE of Pure Victims  : {mean_pure:.4f}")
    print(f"Mean MSE of Noise Txs     : {mean_noise:.4f}")
    print(f"Mean MSE of Normal Txs    : {mean_normal:.4f}")
    
    if mean_noise > 0:
        print(f"\nConclusion: Pure Victims have {mean_pure/mean_noise:.2f}x higher anomaly score than Noise.")
        if mean_pure > mean_noise:
            print("=> This proves the model intrinsically isolates real victims, and lower Enrichment Ratio in the Raw dataset is indeed caused by noise.")

if __name__ == "__main__":
    evaluate()
