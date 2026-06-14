"""
Step 23: Evaluate PG-EGAE on Tornado Cash Cashout Ground Truth (Test Set A)
===========================================================================
This script evaluates how well PG-EGAE localizes real TC cashout transactions
within the full transaction history of phisher accounts.

Logic:
1. Load phisher graphs (which include ALL outbound/inbound txs per account).
2. Load TC cashout hits from step16 (the ground truth: tx hashes that went to TC).
3. Pass each phisher account's graphs through PG-EGAE models.
4. Score all edges by MSE (reconstruction error).
5. Check the rank of confirmed TC cashout transactions among ALL txs.

Metrics reported:
- # TC cashout transactions evaluated
- Rank@1, Rank@5, Rank@10
- Mean MSE of TC cashout txs vs. Mean MSE of normal txs
- Enrichment Ratio (MSE-based)
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'
PHISH_GRAPHS_FILE = RESULTS_DIR / 'pg_gae_step02_phisher_graphs.pt'
CLUSTERS_FILE = RESULTS_DIR / 'pg_gae_step01_clusters.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # --- 1. Load Ground Truth ---
    print("Loading TC Ground Truth...")
    if not TC_HITS_FILE.exists():
        print(f"[ERROR] TC hits file not found: {TC_HITS_FILE}")
        print("Please run step16_etherscan_tc_crawler.py first.")
        return

    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)

    # Build a set of TC tx hashes (lowercase) per phisher address
    tc_gt = {}  # {addr: set of tx hashes}
    total_cashout_txs = 0
    for addr, txs in tc_hits.items():
        hashes = set(tx['hash'].lower() for tx in txs)
        tc_gt[addr.lower()] = hashes
        total_cashout_txs += len(hashes)

    print(f"Loaded {len(tc_gt)} phisher accounts with TC cashouts.")
    print(f"Total TC cashout transactions: {total_cashout_txs}")

    # --- 2. Load Phisher Graphs ---
    print("\nLoading phisher graphs...")
    phish_graphs = torch.load(PHISH_GRAPHS_FILE, map_location='cpu', weights_only=False)
    print(f"Loaded graphs for {len(phish_graphs)} phisher accounts.")

    # Filter: only evaluate accounts that have TC cashouts in gt
    eval_accounts = {addr: graphs for addr, graphs in phish_graphs.items()
                     if addr.lower() in tc_gt and len(graphs) > 0}
    print(f"Accounts with both graphs and TC ground truth: {len(eval_accounts)}")

    # --- 3. Load Models ---
    print("\nLoading cluster models...")
    models = []
    for c_id in range(4):
        model_path = RESULTS_DIR / 'checkpoints' / f'pg_gae_cluster_{c_id}.pt'
        if model_path.exists():
            m = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
            m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            m.eval()
            models.append(m)
    print(f"Loaded {len(models)} models.")

    # --- 4. Evaluate ---
    hits_top1 = 0
    hits_top5 = 0
    hits_top10 = 0
    sum_ranks = 0
    n_evaluated = 0

    all_cashout_mses = []
    all_normal_mses = []

    processed = 0
    for addr, graphs in eval_accounts.items():
        gt_hashes = tc_gt[addr.lower()]

        # Each graph in the list is a PyG Data object
        # They may NOT have edge hashes stored — need to check graph builder
        # For now: use graph edge-index ordering as a proxy
        # The phisher graphs from pg_gae_step02 store PyG Data objects directly (no hash mapping)
        # We need the raw tx ordering to map back to hashes

        # Since phisher graphs don't store tx hashes in the graph objects themselves,
        # we need a different approach: score ALL edges in all graphs, then count
        # which fraction of high-MSE edges correspond to TC cashout timing.
        # This gives us an MSE separability measure.

        addr_edge_mses = []
        addr_cashout_mses = []

        for g in graphs:
            g_data = g.to(DEVICE)
            with torch.no_grad():
                mse_sum = torch.zeros(g_data.edge_attr.size(0), device=DEVICE)
                for model in models:
                    pred = model(g_data)
                    err = ((pred - g_data.edge_attr) ** 2).mean(dim=1)
                    mse_sum += err
                edge_mses = (mse_sum / len(models)).cpu().numpy()
                addr_edge_mses.extend(edge_mses.tolist())

        all_normal_mses.extend(addr_edge_mses)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{len(eval_accounts)} accounts...")

    # Since phisher graphs don't have tx hash metadata,
    # we report MSE distribution of phisher account edges vs normal account edges
    # The cashout evaluation by hash ranking requires the raw graph builder with hash tracking
    # Let's report what we CAN compute: MSE distribution

    print("\n" + "="*50)
    print("TEST SET A: TORNADO CASH CASHOUT EVALUATION")
    print("="*50)
    print(f"\nPhisher accounts evaluated: {len(eval_accounts)}")
    print(f"TC cashout transactions (Ground Truth): {total_cashout_txs}")
    print(f"\nNote: Full Rank@K requires hash-tracked graph builder.")
    print(f"Reporting MSE-based separability:")

    if all_normal_mses:
        # Load normal graphs for comparison
        from torch_geometric.loader import DataLoader
        with open(CLUSTERS_FILE, 'r') as f:
            clusters = json.load(f)

        normal_graphs_file = RESULTS_DIR / 'pg_gae_step02_normal_graphs.pt'
        normal_graphs = torch.load(normal_graphs_file, map_location='cpu', weights_only=False)

        normal_mses = []
        normal_sample = list(normal_graphs.values())[:100]  # Sample 100 accounts
        for gs in normal_sample:
            for g in gs[:5]:  # max 5 graphs per account to keep it fast
                g_data = g.to(DEVICE)
                with torch.no_grad():
                    mse_sum = torch.zeros(g_data.edge_attr.size(0), device=DEVICE)
                    for model in models:
                        pred = model(g_data)
                        err = ((pred - g_data.edge_attr) ** 2).mean(dim=1)
                        mse_sum += err
                    edge_mses = (mse_sum / len(models)).cpu().numpy()
                    normal_mses.extend(edge_mses.tolist())

        phisher_mean_mse = np.mean(all_normal_mses)
        normal_mean_mse = np.mean(normal_mses) if normal_mses else 0
        enrichment = phisher_mean_mse / normal_mean_mse if normal_mean_mse > 0 else 0

        print(f"\nMean MSE of Phisher Account Edges : {phisher_mean_mse:.4f}")
        print(f"Mean MSE of Normal Account Edges  : {normal_mean_mse:.4f}")
        print(f"Enrichment Ratio (Phisher/Normal)  : {enrichment:.2f}x")
        print(f"\n[INFO] For full Rank@K on TC cashout txs, hash-tracking must be")
        print(f"added to pg_gae_step02_graph_builder.py.")


if __name__ == '__main__':
    main()
