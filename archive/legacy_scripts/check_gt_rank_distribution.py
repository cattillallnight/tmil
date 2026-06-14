import sys
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GatedTMILETH
from src.utils import DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'

def load_transactions(in_path, out_path, target_accounts):
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
    print("Computing GT Rank Distribution for W=200...")
    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)

    tc_gt = {}
    for addr, txs in tc_hits.items():
        hashes = set(tx['hash'].lower() for tx in txs)
        tc_gt[addr.lower()] = hashes

    target_accounts = set(tc_gt.keys())
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)

    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
    record_dict = {r['address']: r for r in records}
    
    model_path = RESULTS_DIR / 'checkpoints' / 'tmil_eth_final.pt'
    model = GatedTMILETH(hand_crafted_dim=4, bert_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    ranks = []
    
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
            
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < 200:
                pad = np.zeros((200 - n, 4), dtype=np.float32)
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
                tx_attn_scores[tx_idx] = max(tx_attn_scores[tx_idx], attn_scores[i])
                
        if np.sum(tx_attn_scores) > 0:
            tx_attn_scores = tx_attn_scores / np.sum(tx_attn_scores)
            
        addr_tx_attn = {hashes[i].lower(): tx_attn_scores[i] for i in range(n_expected)}
        
        ranked_txs = sorted(addr_tx_attn.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                ranks.append(rank)

    dist = {
        '1': 0,
        '2-5': 0,
        '6-10': 0,
        '11-20': 0,
        '>20': 0
    }
    
    for r in ranks:
        if r == 1: dist['1'] += 1
        elif 2 <= r <= 5: dist['2-5'] += 1
        elif 6 <= r <= 10: dist['6-10'] += 1
        elif 11 <= r <= 20: dist['11-20'] += 1
        else: dist['>20'] += 1
        
    print("\n--- GT Rank Distribution ---")
    print(f"Rank\tCount")
    print(f"1\t{dist['1']}")
    print(f"2-5\t{dist['2-5']}")
    print(f"6-10\t{dist['6-10']}")
    print(f"11-20\t{dist['11-20']}")
    print(f">20\t{dist['>20']}")
    print(f"Total\t{len(ranks)}")

if __name__ == '__main__':
    main()
