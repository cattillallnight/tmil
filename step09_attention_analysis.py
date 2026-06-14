"""
Step 27: Mechanism Analysis - Attention Saliency Maps
======================================================
Visualizes the attention distribution shift between 68-dim and 69-dim models
for a specific Tornado Cash phishing account. This proves *why* Enrichment increases.
"""

import os
import sys
import json
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step05_model_architecture import GatedTMILETH
from utils import RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_transactions(in_path, out_path, target_accounts):
    import pandas as pd
    tx_history = defaultdict(list)
    def process_csv(path, is_inbound):
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
    print("Loading TC Ground Truth...")
    TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'
    with open(TC_HITS_FILE, 'r') as f: tc_hits = json.load(f)

    # Pick an account with exactly 1 or 2 clear cashouts
    # Let's find one that has a good number of transactions
    target_addr = None
    target_hashes = None
    for addr, txs in tc_hits.items():
        if len(txs) == 1: # Find an account with a single distinct cashout
            target_addr = addr.lower()
            target_hashes = set(tx['hash'].lower() for tx in txs)
            break
            
    if not target_addr:
        target_addr = list(tc_hits.keys())[0].lower()
        target_hashes = set(tx['hash'].lower() for tx in tc_hits[target_addr])
        
    print(f"Selected Target Account: {target_addr}")

    print("Loading raw transactions...")
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, {target_addr})
    tx_list = tx_history[target_addr]
    hashes = [tx[4] for tx in tx_list]

    print("Loading Original (68-dim) and Hybrid (69-dim) features...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        recs_orig = pickle.load(f)
        rec_orig = {r['address']: r for r in recs_orig}[target_addr]
        
    with open(RESULTS_DIR / 'step02d_features_hybrid_norm.pkl', 'rb') as f:
        recs_hybr = pickle.load(f)
        rec_hybr = {r['address']: r for r in recs_hybr}[target_addr]

    print("Loading Models...")
    model_orig = GatedTMILETH(hand_crafted_dim=4, bert_dim=64).to(DEVICE)
    model_orig.load_state_dict(torch.load(RESULTS_DIR / 'checkpoints' / 'tmil_eth_final.pt', map_location=DEVICE, weights_only=True))
    model_orig.eval()
    
    model_hybr = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model_hybr.load_state_dict(torch.load(RESULTS_DIR / 'checkpoints' / 'tmil_hybrid_final.pt', map_location=DEVICE, weights_only=True))
    model_hybr.eval()

    def get_attention(model, rec, dim):
        hc = rec["hand_crafted"]
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        n_expected = len(hashes)
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < 200:
                pad = np.zeros((200 - n, dim), dtype=np.float32)
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
        return tx_attn_scores

    attn_orig = get_attention(model_orig, rec_orig, 4)
    attn_hybr = get_attention(model_hybr, rec_hybr, 5)

    # Plotting
    print("Generating Saliency Map...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    x = np.arange(len(hashes))
    is_cashout = np.array([h.lower() in target_hashes for h in hashes])
    
    def plot_attn(ax, attn, title, color):
        ax.bar(x[~is_cashout], attn[~is_cashout], color='gray', alpha=0.5, label='Normal Tx')
        ax.bar(x[is_cashout], attn[is_cashout], color=color, alpha=0.9, label='Tornado Cash Tx')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Attention Weight", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
    plot_attn(axes[0], attn_orig, "Original TMIL (68-dim) Attention Distribution", 'orange')
    plot_attn(axes[1], attn_hybr, "Hybrid TMIL (69-dim with PG-EGAE) Attention Distribution", 'red')
    
    axes[1].set_xlabel("Transaction Index (Chronological)", fontsize=12)
    plt.tight_layout()
    
    out_path = RESULTS_DIR / 'saliency_maps' / 'attention_shift_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
