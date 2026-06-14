"""
Test 3: Topology-Only Injection Experiment
===================================================
1. Remove `v_log` and `gap_log`. Edge features = `[dir_feat]`.
2. Train GAE (edge_in_dim=1) on Normal Cluster 1.
3. Inject TC anomaly (OUT direction) into Normal sequences.
4. Evaluate Recall@1. If high -> GAE learns topology. If low -> GAE only learned value.
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
from torch_geometric.loader import DataLoader
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
sys.path.append(str(BASE_DIR))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

BASE_DIR_ROOT = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR_ROOT / "BERT4ETH" / "Data"
RESULTS_DIR = BASE_DIR / "results"
CLUSTERS_FILE = RESULTS_DIR / "pg_gae_step01_clusters.json"

NORMAL_TX_IN  = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def build_topology_graphs(tx_list, max_gap_hours=1.0):
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
    mappings = [] 
    
    for burst, indices in zip(bursts, burst_original_indices):
        if len(burst) < 2: continue
        
        node_map = {}
        node_feats = [[1.0]]
        edges_src = []
        edges_dst = []
        edge_feats = []
        
        edge_to_orig = {}
        
        for e_idx, (tx, orig_idx) in enumerate(zip(burst, indices)):
            t, c, v, is_in = tx
            if c not in node_map:
                node_map[c] = len(node_feats)
                node_feats.append([0.0])
            
            n_idx = node_map[c]
            dir_feat = 1.0 if is_in else -1.0
            
            if is_in:
                edges_src.append(n_idx)
                edges_dst.append(0)
            else:
                edges_src.append(0)
                edges_dst.append(n_idx)
                
            # TOPOLOGY ONLY: Only direction feature!
            edge_feats.append([dir_feat])
            edge_to_orig[e_idx] = orig_idx
            
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        mappings.append(edge_to_orig)
        
    return graphs, mappings

def run_test3():
    print("========== TEST 3: TOPOLOGY-ONLY GAE ==========")
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
        
    c1_normals = [a for a, c in clusters["normal_groups"].items() if c == 1]
    random.seed(42)
    sample_normals = set(random.sample(c1_normals, min(2000, len(c1_normals))))
    
    print("Extracting raw sequences...")
    tx_history = load_transactions(NORMAL_TX_IN, NORMAL_TX_OUT, sample_normals)
    valid_accounts = [(addr, seq) for addr, seq in tx_history.items() if len(seq) >= 10]
    
    print(f"Building clean Topology-Only graphs for {len(valid_accounts)} accounts...")
    clean_graphs = []
    for addr, seq in valid_accounts:
        gs, _ = build_topology_graphs(seq)
        clean_graphs.extend(gs)
        
    print(f"Total clean graphs: {len(clean_graphs)}")
    
    # 1. Train Model
    print("\n--- Training Topology-Only GAE ---")
    model = EdgeGAE(node_in_dim=1, edge_in_dim=1, hidden_dim=64, num_layers=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = EdgeReconstructionLoss().to(DEVICE)
    loader = DataLoader(clean_graphs, batch_size=128, shuffle=True)
    
    model.train()
    for epoch in range(1, 11):
        total_loss = 0
        total_edges = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss, _ = criterion(pred, batch.edge_attr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.edge_index.size(1)
            total_edges += batch.edge_index.size(1)
        print(f"Epoch {epoch:02d}/10 | MSE: {total_loss/max(1,total_edges):.4f}")
        
    # 2. Compute Threshold
    print("\n--- Computing Threshold ---")
    model.eval()
    clean_mses = []
    with torch.no_grad():
        for batch in DataLoader(clean_graphs, batch_size=256, shuffle=False):
            batch = batch.to(DEVICE)
            pred = model(batch)
            _, err = criterion(pred, batch.edge_attr)
            clean_mses.extend(err.cpu().numpy().tolist())
            
    tau = np.percentile(clean_mses, 99)
    print(f"Threshold (99th pct): {tau:.4f}")
    
    # 3. Injection Evaluation
    print("\n--- INJECTION PHASE ---")
    threshold_hits = 0
    top_1_hits = 0
    total_trials = 0
    
    for addr, seq in valid_accounts:
        L = len(seq)
        inj_idx = random.randint(1, L-1)
        
        t_prev = seq[inj_idx-1][0]
        # Payload has NO VALUE IMPACT on the model, only its direction (OUT=False) and topological position matters
        inj_tx = (t_prev + 1.0, TC_ADDRESS, 100.0, False) 
        
        injected_seq = seq.copy()
        injected_seq.insert(inj_idx, inj_tx)
        
        graphs, mappings = build_topology_graphs(injected_seq)
        
        all_edge_mses = []
        for g, mapping in zip(graphs, mappings):
            g_dev = g.to(DEVICE)
            with torch.no_grad():
                pred = model(g_dev)
                _, err = criterion(pred, g_dev.edge_attr)
            err = err.cpu().numpy()
            for e_idx in range(len(err)):
                all_edge_mses.append((mapping[e_idx], err[e_idx]))
                
        if not all_edge_mses: continue
        total_trials += 1
        
        all_edge_mses.sort(key=lambda x: x[1], reverse=True)
        
        inj_mse = next((mse for o, mse in all_edge_mses if o == inj_idx), 0.0)
        if inj_mse > tau:
            threshold_hits += 1
            
        if all_edge_mses[0][0] == inj_idx:
            top_1_hits += 1
            
    print("\n==================================================")
    print("TEST 3 RESULTS: TOPOLOGY-ONLY GAE")
    print("==================================================")
    print(f"Total Sequences Evaluated : {total_trials}")
    print(f"Recall @ Threshold        : {threshold_hits} ({(threshold_hits/total_trials)*100:.2f}%)")
    print(f"Recall @ Top-1            : {top_1_hits} ({(top_1_hits/total_trials)*100:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    run_test3()
