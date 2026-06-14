import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy import stats
import math

# Fix paths for flat architecture
from utils import DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT
from step07_evaluate_baseline import load_transactions

def load_all_ground_truths():
    # 1. Tornado Cash GT
    TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'
    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)
    gt_dict = {}
    for addr, txs in tc_hits.items():
        gt_dict[addr.lower()] = set(tx['hash'].lower() for tx in txs)
        
    # 2. ScamSniffer GT
    SS_FILE = RESULTS_DIR / 'scamsniffer_txlevel_dataset.json'
    if SS_FILE.exists():
        with open(SS_FILE, 'r') as f:
            ss_data = json.load(f)
        for rec in ss_data:
            addr = rec.get('address', '').lower()
            if not addr: continue
            if addr not in gt_dict:
                gt_dict[addr] = set()
            for tx in rec.get('victim_txs', []):
                gt_dict[addr].add(tx['hash'].lower())
                
    return gt_dict

def main():
    print("--- Bước 0: Feature Separability Check ---")
    
    gt_dict = load_all_ground_truths()
    print(f"Loaded {len(gt_dict)} accounts with Ground Truth.")
    
    features_file = RESULTS_DIR / "step02_features.pkl"
    if not features_file.exists():
        print("Features file not found!")
        return
        
    print("Loading step02_features.pkl...")
    with open(features_file, "rb") as f:
        records = pickle.load(f)
    record_dict = {r['address'].lower(): r for r in records}
    
    target_accounts = set(gt_dict.keys())
    print("Loading raw transaction history to match hashes...")
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    
    # feature indices: [z_amount, density, counterparty_novelty, value_ratio]
    gt_features = []
    non_gt_features = []
    
    for addr, gt_hashes in gt_dict.items():
        if addr not in tx_history or addr not in record_dict:
            continue
            
        tx_list = tx_history[addr]
        hashes = [tx[4].lower() for tx in tx_list] # index 4 is hash
        hc = record_dict[addr]["hand_crafted"] # shape (N, 4)
        
        n_expected = len(hashes)
        if hc.shape[0] != n_expected:
            continue
            
        for i, h in enumerate(hashes):
            if h in gt_hashes:
                gt_features.append(hc[i])
            else:
                non_gt_features.append(hc[i])
                
    gt_features = np.array(gt_features)
    non_gt_features = np.array(non_gt_features)
    
    print("\n" + "="*50)
    print(f"Total GT (Burst) transactions: {len(gt_features)}")
    print(f"Total Non-GT (Normal) transactions: {len(non_gt_features)}")
    print("="*50)
    
    feature_names = ["z_amount", "density", "counterparty_novelty", "value_ratio"]
    
    print("\n[Mean Analysis]")
    gt_means = np.mean(gt_features, axis=0)
    non_gt_means = np.mean(non_gt_features, axis=0)
    for i, name in enumerate(feature_names):
        ratio = gt_means[i] / (non_gt_means[i] + 1e-9)
        print(f"  {name:20s}: Burst={gt_means[i]:.4f} | Normal={non_gt_means[i]:.4f} | Ratio: {ratio:.2f}x")

    print("\n[Median Analysis]")
    gt_meds = np.median(gt_features, axis=0)
    non_gt_meds = np.median(non_gt_features, axis=0)
    for i, name in enumerate(feature_names):
        ratio = gt_meds[i] / (non_gt_meds[i] + 1e-9)
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        pool_sd = math.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        if pool_sd == 0: return 0.0
        return (np.mean(x) - np.mean(y)) / pool_sd

    print("\n[Statistical Tests (Mann-Whitney U & Cohen's d)]")
    for i, name in enumerate(feature_names):
        burst_vals = gt_features[:, i]
        normal_vals = non_gt_features[:, i]
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(burst_vals, normal_vals, alternative='two-sided')
        
        # Cohen's d
        d_val = cohens_d(burst_vals, normal_vals)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {name:20s}: Cohen's d = {d_val:>7.4f} | p-value = {p_val:>8.2e} {sig}")
        
if __name__ == "__main__":
    main()
