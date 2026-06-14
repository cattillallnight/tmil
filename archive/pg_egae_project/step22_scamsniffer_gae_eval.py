"""
Step 22: Evaluate PG-EGAE on ScamSniffer Transaction-Level Ground Truth
=======================================================================
This script evaluates how well the unsupervised Graph Autoencoder (PG-EGAE)
localizes real, independently verified victim transactions.

Logic:
1. Load scamsniffer_txlevel_dataset.json.
2. Convert tx sequences into PyG Data star-graphs (similar to Step 2).
3. Pass through trained PG-EGAE models (ensembling across the 4 clusters).
4. Compute edge reconstruction MSE.
5. Rank edges by MSE (highest = most anomalous).
6. Check the rank of the confirmed victim transactions.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
SCAMSNIFFER_DS = RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json"
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
        if len(burst) < 2: continue # Ignore isolated single transactions
        
        node_feats = [[1.0]] # Target is 1
        edges_src = []
        edges_dst = []
        edge_feats = []
        edge_hashes = []
        
        prev_ts = burst[0][0]
        for t, dir_val, v, h in burst:
            # Fake counterparty mapping: each tx is a unique counterparty for simplicity 
            # (since we didn't store exact counterparty addresses in the dataset, and star-graphs
            # treat counterparties as isolated nodes anyway).
            n_idx = len(node_feats)
            node_feats.append([0.0]) # Counterparty is 0
            
            gap = (t - prev_ts) / 3600.0
            prev_ts = t
            
            # log features
            v_log = np.log1p(v)
            gap_log = np.log1p(gap)
            dir_feat = float(dir_val) # 1.0 for IN, -1.0 for OUT
            if dir_feat == 0.0: dir_feat = -1.0 # fallback
            
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
        
        graphs.append({
            'data': Data(x=x, edge_index=edge_index, edge_attr=edge_attr),
            'hashes': edge_hashes
        })
        
    return graphs

def evaluate():
    print("Loading dataset...")
    with open(SCAMSNIFFER_DS, "r") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} addresses.")
    
    print("Loading models...")
    models = []
    for c_id in range(4):
        model_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
        if model_path.exists():
            model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.eval()
            models.append(model)
            
    if not models:
        print("No models found!")
        return
        
    print(f"Loaded {len(models)} cluster models for ensembling.")
    
    metrics = {
        'total_victims_evaluated': 0,
        'hits_top_1': 0,
        'hits_top_5': 0,
        'hits_top_10': 0,
        'sum_ranks': 0
    }
    
    # Track overall AUC-like metrics
    all_victim_mses = []
    all_normal_mses = []
    
    for item in dataset:
        addr = item['address']
        victim_txs = item.get('victim_txs', [])
        if not victim_txs: continue
        
        victim_hashes = set(v['hash'] for v in victim_txs)
        
        # Build graphs
        graphs = sequence_to_graphs(
            addr, 
            item['timestamps'], 
            item['directions'], 
            item['values_eth'], 
            item['hashes']
        )
        
        if not graphs: continue
        
        # Accumulate edge MSEs across all graphs of this address
        addr_tx_mses = {} # hash -> mean MSE
        
        for g_dict in graphs:
            data = g_dict['data'].to(DEVICE)
            hashes = g_dict['hashes']
            
            # Ensemble predictions
            edge_mses_sum = torch.zeros(data.edge_attr.size(0), device=DEVICE)
            with torch.no_grad():
                for model in models:
                    pred = model(data)
                    # Compute per-edge MSE
                    err = ((pred - data.edge_attr)**2).mean(dim=1)
                    edge_mses_sum += err
            
            edge_mses = (edge_mses_sum / len(models)).cpu().numpy()
            
            for h, mse in zip(hashes, edge_mses):
                addr_tx_mses[h] = float(mse)
                
        if not addr_tx_mses: continue
        
        # Rank transactions by MSE (descending, highest error = most anomalous)
        ranked_txs = sorted(addr_tx_mses.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        # Check ranks of victims
        found_victims = 0
        for vh in victim_hashes:
            if vh in ranked_hashes:
                rank = ranked_hashes.index(vh) + 1
                metrics['total_victims_evaluated'] += 1
                metrics['sum_ranks'] += rank
                
                if rank <= 1: metrics['hits_top_1'] += 1
                if rank <= 5: metrics['hits_top_5'] += 1
                if rank <= 10: metrics['hits_top_10'] += 1
                
                all_victim_mses.append(addr_tx_mses[vh])
                found_victims += 1
                
        # Record normal mses
        for h, mse in addr_tx_mses.items():
            if h not in victim_hashes:
                all_normal_mses.append(mse)

    # Print Report
    print("\n==================================================")
    print("SCAMSNIFFER GROUND TRUTH EVALUATION RESULTS")
    print("==================================================")
    
    n_v = metrics['total_victims_evaluated']
    if n_v > 0:
        print(f"Victim Txs Evaluated (within valid bursts): {n_v}")
        print(f"Rank@1  : {metrics['hits_top_1']} ({(metrics['hits_top_1']/n_v)*100:.2f}%)")
        print(f"Rank@5  : {metrics['hits_top_5']} ({(metrics['hits_top_5']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['hits_top_10']} ({(metrics['hits_top_10']/n_v)*100:.2f}%)")
        print(f"Mean Rank: {metrics['sum_ranks']/n_v:.2f}")
        
        mean_v_mse = np.mean(all_victim_mses) if all_victim_mses else 0
        mean_n_mse = np.mean(all_normal_mses) if all_normal_mses else 0
        
        print("\nMSE Separability:")
        print(f"Mean MSE of Victim Txs  : {mean_v_mse:.4f}")
        print(f"Mean MSE of Normal Txs  : {mean_n_mse:.4f}")
        if mean_n_mse > 0:
            print(f"Enrichment Ratio        : {mean_v_mse/mean_n_mse:.2f}x")
    else:
        print("No victim txs were successfully mapped to bursts.")

if __name__ == "__main__":
    evaluate()
