"""
Step 25: Evaluate Hybrid TMIL on Ground Truth
==============================================
This script evaluates the trained Hybrid TMIL model (which uses PG-EGAE MSE 
as a pre-signal feature) on Test Set A (Tornado Cash).

Logic:
1. Load TC Ground Truth hashes from `step16_etherscan_tc_hits.json`.
2. Load raw transactions for target accounts to extract hashes.
3. Load `step02b_features_hybrid.pkl` for target accounts.
4. Pass features through Hybrid TMIL to get Attention weights.
5. Rank transaction hashes by Attention weight.
6. Calculate Rank@1, Rank@5, Rank@10 and Enrichment Ratio (Attention).
"""

import sys
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step05_model_architecture import GatedTMILETH
from utils import (
    DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'

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

def main():
    print("=" * 60)
    print("Step 25: Evaluate Hybrid TMIL on Test Set A (Tornado Cash)")
    print("=" * 60)
    
    print("\n[1] Loading TC Ground Truth...")
    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)

    tc_gt = {}
    for addr, txs in tc_hits.items():
        hashes = set(tx['hash'].lower() for tx in txs)
        tc_gt[addr.lower()] = hashes

    target_accounts = set(tc_gt.keys())
    print(f"Target Phishers: {len(target_accounts)}")

    print("\n[2] Loading raw transactions for target accounts...")
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    print(f"Found transactions for {len(tx_history)} accounts.")

    print("\n[3] Loading Hybrid Features (step02b)...")
    with open(RESULTS_DIR / 'step02b_features_hybrid.pkl', 'rb') as f:
        records = pickle.load(f)
    
    record_dict = {r['address']: r for r in records}
    
    print("\n[4] Loading Hybrid TMIL Model...")
    model_path = RESULTS_DIR / 'checkpoints' / 'tmil_hybrid_final.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Run step07b first.")
        return
        
    model = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    metrics = {
        'total_cashouts_evaluated': 0,
        'hits_top_1': 0,
        'hits_top_5': 0,
        'hits_top_10': 0,
        'sum_ranks': 0
    }
    
    all_cashout_attn = []
    all_normal_attn = []

    print("\n[5] Evaluating accounts with Hybrid TMIL...")
    
    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict:
            continue
            
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"]  
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        
        n_expected = len(hashes)
        if hc.shape[0] != n_expected:
            continue
            
        # Accumulate max attention score for each transaction across overlapping windows
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < 200:
                pad = np.zeros((200 - n, 5), dtype=np.float32)
                hc_win_pad = np.vstack([hc_win, pad])
            else:
                hc_win_pad = hc_win[:200]
                
            hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 200, -1).to(DEVICE)
            
            with torch.no_grad():
                p, attn = model(hc_t, bert_t)
                
            attn_scores = attn.squeeze(0).cpu().numpy()[:n]
            
            for i in range(n):
                tx_idx = start + i
                # TMIL Attention is highly localized per window. Max accumulation works best.
                tx_attn_scores[tx_idx] = max(tx_attn_scores[tx_idx], attn_scores[i])
                
        # Normalize attention scores to sum to 1 for the whole account (optional, but good for interpretation)
        if np.sum(tx_attn_scores) > 0:
            tx_attn_scores = tx_attn_scores / np.sum(tx_attn_scores)
            
        addr_tx_attn = {hashes[i].lower(): tx_attn_scores[i] for i in range(n_expected)}
        
        ranked_txs = sorted(addr_tx_attn.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total_cashouts_evaluated'] += 1
                metrics['sum_ranks'] += rank
                if rank <= 1: metrics['hits_top_1'] += 1
                if rank <= 5: metrics['hits_top_5'] += 1
                if rank <= 10: metrics['hits_top_10'] += 1
                all_cashout_attn.append(addr_tx_attn[ch])
                
        for h, attn_val in addr_tx_attn.items():
            if h not in gt_hashes:
                all_normal_attn.append(attn_val)
                
    print("\n==================================================")
    print("HYBRID TMIL: TORNADO CASH CASHOUT EVALUATION RESULTS")
    print("==================================================")
    n_v = metrics['total_cashouts_evaluated']
    if n_v > 0:
        print(f"TC Cashout Txs Evaluated: {n_v}")
        print(f"Rank@1  : {metrics['hits_top_1']} ({(metrics['hits_top_1']/n_v)*100:.2f}%)")
        print(f"Rank@5  : {metrics['hits_top_5']} ({(metrics['hits_top_5']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['hits_top_10']} ({(metrics['hits_top_10']/n_v)*100:.2f}%)")
        print(f"Mean Rank: {metrics['sum_ranks']/n_v:.2f}")
        
        mean_c_attn = np.mean(all_cashout_attn) if all_cashout_attn else 0
        mean_n_attn = np.mean(all_normal_attn) if all_normal_attn else 0
        
        print("\nAttention Separability:")
        print(f"Mean Attention of TC Cashout : {mean_c_attn:.6f}")
        print(f"Mean Attention of Normal Txs : {mean_n_attn:.6f}")
        if mean_n_attn > 0:
            print(f"Enrichment Ratio (Attention) : {mean_c_attn/mean_n_attn:.2f}x")
    else:
        print("No TC cashout txs were successfully mapped.")

if __name__ == '__main__':
    main()
