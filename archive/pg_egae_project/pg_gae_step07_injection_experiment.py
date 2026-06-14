"""
PG-EGAE Step 7: Synthetic Injection Experiment
===================================================
1. Sample clean normal transaction sequences.
2. Inject synthetic Tornado Cash transactions (Ground Truth Anomalies).
3. Evaluate with PG-EGAE to see if Edge-MSE accurately flags the injected transactions.
"""

import sys
import os
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR_ROOT = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR_ROOT / "BERT4ETH" / "Data"
RESULTS_DIR = BASE_DIR_ROOT / "tmil_eth" / "results"
CLUSTERS_FILE = RESULTS_DIR / "pg_gae_step01_clusters.json"

NORMAL_TX_IN  = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Known TC endpoint (100 ETH pool)
TC_ADDRESS = "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3"

def load_transactions(in_path, out_path, target_accounts):
    tx_history = defaultdict(list)
    def process_csv(path, is_inbound):
        chunk_iter = pd.read_csv(path, chunksize=100000, header=None, low_memory=False)
        for chunk in chunk_iter:
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
                    
    if in_path.exists(): process_csv(in_path, True)
    if out_path.exists(): process_csv(out_path, False)
    
    for a in tx_history:
        tx_history[a].sort(key=lambda x: x[0])
    return tx_history

def build_graphs_with_mapping(tx_list, max_gap_hours=1.0):
    if not tx_list: return [], []
    
    bursts = []
    burst_original_indices = []
    
    current_burst = [tx_list[0]]
    current_indices = [0]
    
    for i, tx in enumerate(tx_list[1:], start=1):
        gap = (tx[0] - current_burst[-1][0]) / 3600.0
        if gap > max_gap_hours:
            bursts.append(current_burst)
            burst_original_indices.append(current_indices)
            current_burst = [tx]
            current_indices = [i]
        else:
            current_burst.append(tx)
            current_indices.append(i)
            
    bursts.append(current_burst)
    burst_original_indices.append(current_indices)
    
    graphs = []
    mappings = [] # list of dicts: edge_index -> original_tx_idx
    
    for burst, indices in zip(bursts, burst_original_indices):
        if len(burst) < 2: continue
        
        node_map = {}
        node_feats = [[1.0]]
        edges_src = []
        edges_dst = []
        edge_feats = []
        
        edge_to_orig = {}
        
        prev_ts = burst[0][0]
        for e_idx, (tx, orig_idx) in enumerate(zip(burst, indices)):
            t, c, v, is_in = tx
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
            edge_to_orig[e_idx] = orig_idx
            
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        mappings.append(edge_to_orig)
        
    return graphs, mappings

def run_experiment():
    print("========== PG-EGAE Synthetic Injection Experiment ==========")
    
    # Load clusters to sample Normal Cluster 1 accounts
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
        
    c1_normals = [a for a, c in clusters["normal_groups"].items() if c == 1]
    random.seed(42)
    sample_normals = set(random.sample(c1_normals, min(2000, len(c1_normals))))
    print(f"Sampled {len(sample_normals)} Normal Accounts from Cluster 1.")
    
    print("Extracting raw sequences...")
    tx_history = load_transactions(NORMAL_TX_IN, NORMAL_TX_OUT, sample_normals)
    print(f"Loaded sequences for {len(tx_history)} accounts.")
    
    # Prepare model
    model_path = RESULTS_DIR / "checkpoints" / "pg_gae_cluster_1.pt"
    model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    criterion = EdgeReconstructionLoss().to(DEVICE)
    
    # We also need the threshold for Cluster 1
    # We will compute threshold dynamically from these 1000 clean sequences
    print("\nComputing clean baseline threshold...")
    clean_mses = []
    valid_accounts = []
    
    for addr, seq in tx_history.items():
        if len(seq) < 10: continue
        valid_accounts.append((addr, seq))
        graphs, _ = build_graphs_with_mapping(seq)
        for g in graphs:
            g_dev = g.to(DEVICE)
            with torch.no_grad():
                pred = model(g_dev)
                _, err = criterion(pred, g_dev.edge_attr)
                clean_mses.extend(err.cpu().numpy().tolist())
                
    tau = np.percentile(clean_mses, 99)
    print(f"Clean Edges evaluated: {len(clean_mses)}")
    print(f"Cluster 1 Threshold (99th pct): {tau:.4f}")
    
    # INJECTION PHASE
    print("\n--- INJECTION PHASE ---")
    results = []
    top_1_hits = 0
    top_3_hits = 0
    threshold_hits = 0
    
    total_trials = 0
    
    for addr, seq in valid_accounts:
        L = len(seq)
        # Choose a random index to inject (avoid first edge to allow time gap)
        inj_idx = random.randint(1, L-1)
        
        # Craft TC payload: 1 second after the previous transaction
        t_prev = seq[inj_idx-1][0]
        inj_tx = (t_prev + 1.0, TC_ADDRESS, 100.0, False)
        
        # Inject
        injected_seq = seq.copy()
        injected_seq.insert(inj_idx, inj_tx)
        
        # Rebuild graphs
        graphs, mappings = build_graphs_with_mapping(injected_seq)
        
        # Evaluate all edges in all graphs
        all_edge_mses = [] # List of (orig_idx, mse)
        
        for g, mapping in zip(graphs, mappings):
            g_dev = g.to(DEVICE)
            with torch.no_grad():
                pred = model(g_dev)
                _, err = criterion(pred, g_dev.edge_attr)
                
            err = err.cpu().numpy()
            for e_idx in range(len(err)):
                orig_idx = mapping[e_idx]
                all_edge_mses.append((orig_idx, err[e_idx]))
                
        if not all_edge_mses: continue
        
        total_trials += 1
        
        # Sort by MSE descending
        all_edge_mses.sort(key=lambda x: x[1], reverse=True)
        
        # Did the injected index cross threshold?
        inj_mse = next((mse for o, mse in all_edge_mses if o == inj_idx), 0.0)
        if inj_mse > tau:
            threshold_hits += 1
            
        # Top-K
        top1_idx = all_edge_mses[0][0]
        top3_indices = [x[0] for x in all_edge_mses[:3]]
        
        if top1_idx == inj_idx:
            top_1_hits += 1
        if inj_idx in top3_indices:
            top_3_hits += 1
            
        if total_trials % 100 == 0:
            print(f"Processed {total_trials} sequences...")

    print("\n==================================================")
    print("INJECTION EXPERIMENT RESULTS")
    print("==================================================")
    print(f"Total Sequences Evaluated : {total_trials}")
    print(f"Recall @ Threshold        : {threshold_hits} ({(threshold_hits/total_trials)*100:.2f}%)")
    print(f"Recall @ Top-1            : {top_1_hits} ({(top_1_hits/total_trials)*100:.2f}%)")
    print(f"Recall @ Top-3            : {top_3_hits} ({(top_3_hits/total_trials)*100:.2f}%)")
    print("==================================================")
    
    if (top_1_hits/total_trials) > 0.5:
        print("-> SUCCESS: Model strictly isolates the injected anomaly with >50% Top-1 Recall!")
    else:
        print("-> FAILURE: Model fails to localize the injected anomaly.")

if __name__ == "__main__":
    run_experiment()
