"""
PG-EGAE Step 6: Validation against Hard Ground Truth
===================================================
1. IN vs OUT ratio of flagged edges.
2. True Positive Rate on Known Tornado Cash transactions.
3. Random baseline / Baseline MSE distribution shift check.
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR_ROOT = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR_ROOT / "BERT4ETH" / "Data"
RESULTS_DIR = BASE_DIR_ROOT / "tmil_eth" / "results"
CLUSTERS_FILE = RESULTS_DIR / "pg_gae_step01_clusters.json"
NORM_GRAPHS_FILE = RESULTS_DIR / "pg_gae_step02_normal_graphs.pt"
PHISH_GRAPHS_FILE = RESULTS_DIR / "pg_gae_step02_phisher_graphs.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_tornado_cash_endpoints():
    tc_in = pd.read_csv(DATA_DIR / "tornado_trans_in_removed.csv", header=None, usecols=[6], dtype=str)
    return set(tc_in[6].str.lower().tolist())

def load_phisher_tc_transactions(tc_endpoints):
    # Load all OUT transactions from phishers
    df_out = pd.read_csv(DATA_DIR / "phisher_transaction_out.csv", header=None, low_memory=False)
    df_out = df_out.dropna(subset=[5, 6, 7, 11])
    
    addrs_from = df_out[5].astype(str).str.lower()
    addrs_to = df_out[6].astype(str).str.lower()
    values = pd.to_numeric(df_out[7], errors='coerce') / 1e18
    timestamps = pd.to_numeric(df_out[11], errors='coerce')
    
    # Filter TC
    mask_tc = addrs_to.isin(tc_endpoints)
    
    tc_txs = defaultdict(list)
    for f, v, t in zip(addrs_from[mask_tc], values[mask_tc], timestamps[mask_tc]):
        tc_txs[f].append((float(t), float(v)))
    return tc_txs

from collections import defaultdict

def validate():
    print("Loading data...")
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
        
    phish_graphs = torch.load(PHISH_GRAPHS_FILE, map_location='cpu', weights_only=False)
    norm_graphs = torch.load(NORM_GRAPHS_FILE, map_location='cpu', weights_only=False)
    
    tc_endpoints = load_tornado_cash_endpoints()
    print(f"Loaded {len(tc_endpoints)} Tornado Cash endpoints.")
    
    tc_txs_by_phisher = load_phisher_tc_transactions(tc_endpoints)
    print(f"Found {sum(len(v) for v in tc_txs_by_phisher.values())} Tornado Cash transactions across {len(tc_txs_by_phisher)} phishers.")
    
    total_flagged_in = 0
    total_flagged_out = 0
    total_flagged = 0
    
    tc_matched_in_graph = 0
    tc_flagged = 0
    
    all_norm_mses = []
    all_phish_mses = []

    for c_id in range(4):
        print(f"\nEvaluating Cluster {c_id}...")
        model_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
        if not model_path.exists(): continue
            
        model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        criterion = EdgeReconstructionLoss().to(DEVICE)
        
        # 1. Get Normal Threshold
        c_norm_gs = [g for a, gs in norm_graphs.items() if clusters["normal_groups"].get(a) == c_id for g in gs]
        if not c_norm_gs: continue
        
        loader_norm = DataLoader(c_norm_gs, batch_size=256, shuffle=False)
        norm_mses = []
        with torch.no_grad():
            for batch in loader_norm:
                batch = batch.to(DEVICE)
                pred = model(batch)
                _, err = criterion(pred, batch.edge_attr)
                norm_mses.extend(err.cpu().numpy().tolist())
        
        tau = np.percentile(norm_mses, 99)
        all_norm_mses.extend(norm_mses)
        
        # 2. Evaluate Phishers and Match TC
        c_phish_addrs = [a for a in phish_graphs.keys() if clusters["phisher_groups"].get(a) == c_id]
        
        for addr in c_phish_addrs:
            gs = phish_graphs[addr]
            known_tc = tc_txs_by_phisher.get(addr, [])
            
            for g in gs:
                g_dev = g.to(DEVICE)
                with torch.no_grad():
                    pred = model(g_dev)
                    _, err = criterion(pred, g_dev.edge_attr)
                    
                err = err.cpu().numpy()
                all_phish_mses.extend(err.tolist())
                
                is_flagged = err > tau
                
                # Check 1: IN vs OUT
                dirs = g.edge_attr[:, 2].numpy() # 1.0 for IN, -1.0 for OUT
                total_flagged += is_flagged.sum()
                total_flagged_in += (is_flagged & (dirs > 0)).sum()
                total_flagged_out += (is_flagged & (dirs < 0)).sum()
                
                # Check 2: Match TC transactions
                if not known_tc: continue
                
                # Try to match edges to known TC by checking value and direction
                # TC transactions are OUT (dir < 0)
                # Since we stored log(value+1), we can recover approx value
                values = np.expm1(g.edge_attr[:, 0].numpy())
                
                for tc_ts, tc_val in known_tc:
                    # Match condition: OUT direction AND value is very close
                    # (Timestamp is harder to match exactly because we only stored gaps, but value is very unique for TC e.g., 0.1, 1, 10, 100)
                    matches = (dirs < 0) & (np.abs(values - tc_val) < 1e-3)
                    if matches.any():
                        tc_matched_in_graph += 1
                        # If ANY of the matching edges is flagged, we count it as a hit
                        if is_flagged[matches].any():
                            tc_flagged += 1
                        # Remove from known_tc to avoid double counting
                        known_tc.remove((tc_ts, tc_val))
                        break

    print("\n==================================================")
    print("CHECK 1: IN vs OUT Ratio of Flagged Edges")
    print("==================================================")
    print(f"Total Flagged Edges : {total_flagged}")
    print(f"IN Flagged (Victims): {total_flagged_in} ({(total_flagged_in/total_flagged)*100:.2f}%)")
    print(f"OUT Flagged (Cashout): {total_flagged_out} ({(total_flagged_out/total_flagged)*100:.2f}%)")
    
    print("\n==================================================")
    print("CHECK 2: Ground Truth Sanity Check (Tornado Cash)")
    print("==================================================")
    print(f"Total TC Transactions identified in Graphs: {tc_matched_in_graph}")
    print(f"Successfully Flagged by GAE             : {tc_flagged}")
    if tc_matched_in_graph > 0:
        print(f"True Positive Rate (Recall) on TC       : {(tc_flagged/tc_matched_in_graph)*100:.2f}%")
        
    print("\n==================================================")
    print("CHECK 3: Baseline MSE Distribution Shift Check")
    print("==================================================")
    norm_mean = np.mean(all_norm_mses)
    phish_mean = np.mean(all_phish_mses)
    print(f"Mean Edge MSE on Normal Graphs : {norm_mean:.4f}")
    print(f"Mean Edge MSE on Phish Graphs  : {phish_mean:.4f}")
    print(f"Global Shift                   : {phish_mean / norm_mean:.2f}x")
    
    if phish_mean / norm_mean > 5.0:
        print("-> WARNING: Phisher graphs have massively higher baseline error. Enrichment factor might be inflated by structural domain shift.")
    else:
        print("-> OK: Baseline shift is modest. The model is flagging specific anomalous edges, not failing on the entire graph.")

if __name__ == "__main__":
    validate()
