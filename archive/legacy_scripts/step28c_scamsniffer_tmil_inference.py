"""
Step 28c: ScamSniffer Pilot - TMIL Hybrid Inference
===================================================
Combines Hand-crafted features, PG-EGAE MSE, and BERT4ETH embeddings
to evaluate the Hybrid TMIL model on the 5 ScamSniffer pilot accounts.
"""

import sys
import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step05_model_architecture import GatedTMILETH
from utils import RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PILOT_ACCOUNTS = [
    '0x9ce67dc9856c9f887ef5a80ae2178d5903864155', 
    '0x7e4384ad48860ae13107b8c8a2b877191edfe2a6', 
    '0x9307d0730bbe0e2df8f747e3f693772ad83debcb', 
    '0x40881dd5b6482854fc01d010ed99fd346f0608b1', 
    '0xe455395bd3468069e0f506e22e13f61666eba36a'
]

def sliding_windows(seq_len: int, W: int = 200, S: int = 50):
    if seq_len <= W:
        return [(0, seq_len)]
    windows = []
    start = 0
    while start < seq_len:
        end = min(start + W, seq_len)
        windows.append((start, end))
        if end == seq_len:
            break
        start += S
    return windows

def compute_per_account_features(timestamps, values_eth, directions, counterparts):
    n = len(timestamps)
    if n == 0:
        return np.zeros((0, 4))

    mu = np.mean(values_eth)
    sigma = np.std(values_eth) + 1e-9
    z_amount = np.clip((values_eth - mu) / sigma, -3.0, 3.0) / 3.0

    density = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = timestamps[i] - 1800.0
        hi = timestamps[i] + 1800.0
        density[i] = float(np.sum((timestamps >= lo) & (timestamps <= hi)))

    seen = set()
    novelty = np.zeros(n, dtype=np.float64)
    for i, cp in enumerate(counterparts):
        if cp not in seen:
            novelty[i] = 1.0
            seen.add(cp)

    cum_in = cum_out = 0.0
    value_ratio = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if directions[i] == 1:
            cum_in += values_eth[i]
        else:
            cum_out += values_eth[i]
        value_ratio[i] = cum_in / (cum_out + 1e-9)

    return np.column_stack([z_amount, density, novelty, value_ratio])

def main():
    print("--- ScamSniffer Pilot: TMIL Inference ---")
    
    with open(RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json", "r") as f:
        data = json.load(f)
    pilot_data = [x for x in data if x['address'].lower() in PILOT_ACCOUNTS]
    
    with open(RESULTS_DIR / "step28_scamsniffer_bert_embeddings.pkl", "rb") as f:
        addr2bert = pickle.load(f)
        
    with open(RESULTS_DIR / "step28_scamsniffer_pgegae_mses.pkl", "rb") as f:
        addr2mse = pickle.load(f)
        
    model = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / "tmil_hybrid_final.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    
    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    all_c_attn, all_n_attn = [], []
    
    print("\nEvaluating Accounts:")
    for item in pilot_data:
        addr = item['address'].lower()
        if addr not in addr2bert or addr not in addr2mse:
            print(f"  Skipping {addr} (Missing features)")
            continue
            
        gt_hashes = set(tx['hash'].lower() for tx in item.get('victim_txs', []))
        if not gt_hashes: continue
        
        hashes = item['hashes']
        n_expected = len(hashes)
        
        hc_4dim = compute_per_account_features(np.array(item['timestamps']), np.array(item['values_eth']), item['directions'], item['counterparties'])
        mse_dim = np.array(addr2mse[addr]).reshape(-1, 1)
        
        # NOTE: Normalize MSE using the training max (~4.2)
        mse_norm = np.clip(mse_dim / 4.2, 0.0, 1.0)
        
        hc = np.hstack([hc_4dim, mse_norm])
        bert = addr2bert[addr]
        
        wins = sliding_windows(n_expected)
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for start, end in wins:
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
                tx_attn_scores[start + i] = max(tx_attn_scores[start + i], attn_scores[i])
                
        if np.sum(tx_attn_scores) > 0:
            tx_attn_scores = tx_attn_scores / np.sum(tx_attn_scores)
            
        addr_tx_attn = {hashes[i].lower(): tx_attn_scores[i] for i in range(n_expected)}
        ranked_txs = sorted(addr_tx_attn.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total'] += 1
                if rank <= 1: metrics['h1'] += 1
                if rank <= 5: metrics['h5'] += 1
                if rank <= 10: metrics['h10'] += 1
                all_c_attn.append(addr_tx_attn[ch])
                
        for h, attn_val in addr_tx_attn.items():
            if h not in gt_hashes:
                all_n_attn.append(attn_val)
                
        print(f"  {addr}: Found {len(gt_hashes)} GT txs. Top rank: {ranked_hashes.index(list(gt_hashes)[0])+1 if gt_hashes else -1}")
        
    n_v = metrics['total']
    print("\n--- RESULTS ---")
    print(f"Total ScamSniffer Victims Evaluated: {n_v}")
    if n_v > 0:
        print(f"Rank@1  : {metrics['h1']} ({(metrics['h1']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['h10']} ({(metrics['h10']/n_v)*100:.2f}%)")
        mean_c = np.mean(all_c_attn) if all_c_attn else 0
        mean_n = np.mean(all_n_attn) if all_n_attn else 0
        print(f"Mean Victim Attn: {mean_c:.6f}")
        print(f"Mean Normal Attn: {mean_n:.6f}")
        if mean_n > 0:
            print(f"Enrichment Ratio: {mean_c/mean_n:.2f}x")

if __name__ == "__main__":
    main()
