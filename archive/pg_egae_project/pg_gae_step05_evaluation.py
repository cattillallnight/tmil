"""
PG-EGAE Step 5: Evaluation & Ground Truth Extraction
===================================================
Goal: Pass Normal and Phisher graphs through their respective Peer-Group EGAE.
Method:
- Calculate Edge-level MSE for Normal graphs to establish the 99th percentile threshold per Cluster.
- Calculate Edge-level MSE for Phisher graphs.
- Flag any edge in a Phisher graph that exceeds the threshold as "OOD / Anomalous".
- Report the detection rates.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
CLUSTERS_FILE = RESULTS_DIR / "pg_gae_step01_clusters.json"
NORM_GRAPHS_FILE = RESULTS_DIR / "pg_gae_step02_normal_graphs.pt"
PHISH_GRAPHS_FILE = RESULTS_DIR / "pg_gae_step02_phisher_graphs.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256

def load_data():
    print("Loading clusters and graphs...")
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
        
    norm_graphs = torch.load(NORM_GRAPHS_FILE, map_location='cpu', weights_only=False)
    phish_graphs = torch.load(PHISH_GRAPHS_FILE, map_location='cpu', weights_only=False)
    
    # Group by cluster
    norm_by_c = {0: [], 1: [], 2: [], 3: []}
    phish_by_c = {0: [], 1: [], 2: [], 3: []}
    
    for a, gs in norm_graphs.items():
        if a in clusters["normal_groups"]:
            norm_by_c[clusters["normal_groups"][a]].extend(gs)
            
    for a, gs in phish_graphs.items():
        if a in clusters["phisher_groups"]:
            phish_by_c[clusters["phisher_groups"][a]].extend(gs)
            
    return norm_by_c, phish_by_c

def evaluate_cluster(c_id, norm_gs, phish_gs):
    if not norm_gs or not phish_gs: return None
    
    model_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
    if not model_path.exists():
        print(f"Model for cluster {c_id} not found.")
        return None
        
    model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    criterion = EdgeReconstructionLoss().to(DEVICE)
    
    def get_edge_errors(graphs):
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        errors = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                _, err_per_edge = criterion(pred, batch.edge_attr)
                errors.extend(err_per_edge.cpu().numpy().tolist())
        return np.array(errors)

    print(f"\n--- Cluster {c_id} ---")
    print("Computing errors for Normal edges...")
    norm_errs = get_edge_errors(norm_gs)
    
    # Dynamic Threshold: 99th percentile of normal errors
    tau = np.percentile(norm_errs, 99)
    print(f"Normal edges: {len(norm_errs)}")
    print(f"Threshold (99th pct): {tau:.4f}")
    
    print("Computing errors for Phisher edges...")
    phish_errs = get_edge_errors(phish_gs)
    print(f"Phisher edges: {len(phish_errs)}")
    
    # Detection
    n_flagged_norm = np.sum(norm_errs > tau)
    n_flagged_phish = np.sum(phish_errs > tau)
    
    p_flagged_norm = (n_flagged_norm / len(norm_errs)) * 100
    p_flagged_phish = (n_flagged_phish / len(phish_errs)) * 100
    
    print(f"Flagged Normal  : {n_flagged_norm} ({p_flagged_norm:.2f}%)")
    print(f"Flagged Phisher : {n_flagged_phish} ({p_flagged_phish:.2f}%)")
    
    # Enrichment factor
    enrichment = p_flagged_phish / p_flagged_norm if p_flagged_norm > 0 else 0
    print(f"Enrichment Factor: {enrichment:.2f}x")
    
    return {
        "threshold": float(tau),
        "norm_flagged": int(n_flagged_norm),
        "norm_total": len(norm_errs),
        "phish_flagged": int(n_flagged_phish),
        "phish_total": len(phish_errs)
    }

def main():
    print("========== Phase 5: Evaluation & Ground Truth Extraction ==========")
    norm_by_c, phish_by_c = load_data()
    
    results = {}
    for c_id in range(4):
        res = evaluate_cluster(c_id, norm_by_c[c_id], phish_by_c[c_id])
        if res: results[c_id] = res
        
    # Aggregate
    tot_norm_f = sum(r["norm_flagged"] for r in results.values())
    tot_norm_t = sum(r["norm_total"] for r in results.values())
    tot_phish_f = sum(r["phish_flagged"] for r in results.values())
    tot_phish_t = sum(r["phish_total"] for r in results.values())
    
    print("\n========== FINAL AGGREGATE RESULTS ==========")
    print(f"Total Normal Edges Flagged : {tot_norm_f} / {tot_norm_t} ({(tot_norm_f/tot_norm_t)*100:.2f}%)")
    print(f"Total Phisher Edges Flagged: {tot_phish_f} / {tot_phish_t} ({(tot_phish_f/tot_phish_t)*100:.2f}%)")
    enrich = (tot_phish_f/tot_phish_t) / (tot_norm_f/tot_norm_t)
    print(f"Global Enrichment Factor   : {enrich:.2f}x")
    
    out_file = RESULTS_DIR / "pg_gae_step05_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f)
    print(f"\n[OK] Results saved to {out_file.name}")

if __name__ == "__main__":
    main()
