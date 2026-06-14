"""
Step 28b: ScamSniffer Pilot - PG-EGAE Extraction
================================================
Calculates the PG-EGAE MSE (5th feature dimension) for the 5 pilot accounts.
"""

import sys
import os
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PILOT_ACCOUNTS = [
    '0x9ce67dc9856c9f887ef5a80ae2178d5903864155', 
    '0x7e4384ad48860ae13107b8c8a2b877191edfe2a6', 
    '0x9307d0730bbe0e2df8f747e3f693772ad83debcb', 
    '0x40881dd5b6482854fc01d010ed99fd346f0608b1', 
    '0xe455395bd3468069e0f506e22e13f61666eba36a'
]

def sequence_to_graphs(timestamps, directions, values_eth, hashes, max_gap_hours=1.0):
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

def main():
    print("--- ScamSniffer Pilot: Extracting PG-EGAE MSE ---")
    
    with open(RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json", "r") as f:
        data = json.load(f)
        
    pilot_data = [x for x in data if x['address'].lower() in PILOT_ACCOUNTS]
    
    print("Loading PG-EGAE models...")
    models = []
    for c_id in range(4):
        model_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
        if model_path.exists():
            model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            models.append(model)
            
    addr2mse = {}
    
    for item in pilot_data:
        addr = item['address'].lower()
        graphs = sequence_to_graphs(item['timestamps'], item['directions'], item['values_eth'], item['hashes'])
        
        tx_mse = {}
        for g_dict in graphs:
            d = g_dict['data'].to(DEVICE)
            hashes = g_dict['hashes']
            
            edge_mses_sum = torch.zeros(d.edge_attr.size(0), device=DEVICE)
            with torch.no_grad():
                for model in models:
                    pred = model(d)
                    err = ((pred - d.edge_attr)**2).mean(dim=1)
                    edge_mses_sum += err
            
            edge_mses = (edge_mses_sum / len(models)).cpu().numpy()
            
            for h, mse in zip(hashes, edge_mses):
                tx_mse[h.lower()] = float(mse)
                
        # Fill missing MSEs with noise baseline (~0.12) or 0
        ordered_mses = []
        for h in item['hashes']:
            h_low = h.lower()
            mse = tx_mse.get(h_low, 0.12) # 0.12 is typical noise MSE
            ordered_mses.append(mse)
            
        addr2mse[addr] = ordered_mses
        print(f"Extracted MSE for {addr}: min={np.min(ordered_mses):.4f}, max={np.max(ordered_mses):.4f}")
        
    out_pkl = RESULTS_DIR / "step28_scamsniffer_pgegae_mses.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(addr2mse, f)
    print(f"Saved PG-EGAE MSEs to {out_pkl}")
    print("[OK] Step 28b Complete.")

if __name__ == "__main__":
    main()
