import sys
import os
import pickle
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RESULTS_DIR
from tmil_architecture import GatedTMILETH, GatedCompoundLoss
from step04_train_baseline import AccountWindowDataset, collate_fn, train_one_epoch, evaluate_epoch

import json
from collections import defaultdict
import pandas as pd
from utils import PHISHER_TX_IN, PHISHER_TX_OUT
from step07_evaluate_baseline import load_transactions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on_tornado(model, records, W, S):
    TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'
    with open(TC_HITS_FILE, 'r') as f:
        tc_hits = json.load(f)

    tc_gt = {}
    for addr, txs in tc_hits.items():
        tc_gt[addr.lower()] = set(tx['hash'].lower() for tx in txs)

    target_accounts = set(tc_gt.keys())
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    record_dict = {r['address']: r for r in records}

    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    ranks_list = []

    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict: continue
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"]  
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        
        n_expected = len(hashes)
        if hc.shape[0] != n_expected: continue
            
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < W:
                pad = np.zeros((W - n, 4), dtype=np.float32)
                hc_win_pad = np.vstack([hc_win, pad])
            else:
                hc_win_pad = hc_win[:W]
                
            hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, W, -1).to(DEVICE)
            
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
                ranks_list.append(rank)
                metrics['total'] += 1
                if rank <= 1: metrics['h1'] += 1
                if rank <= 5: metrics['h5'] += 1
                if rank <= 10: metrics['h10'] += 1

    dist = {'1': 0, '2-5': 0, '6-10': 0, '11-20': 0, '>20': 0}
    for r in ranks_list:
        if r == 1: dist['1'] += 1
        elif 2 <= r <= 5: dist['2-5'] += 1
        elif 6 <= r <= 10: dist['6-10'] += 1
        elif 11 <= r <= 20: dist['11-20'] += 1
        else: dist['>20'] += 1

    print("\n--- GT Rank Distribution ---", flush=True)
    print(f"Rank\tCount", flush=True)
    print(f"1\t{dist['1']}", flush=True)
    print(f"2-5\t{dist['2-5']}", flush=True)
    print(f"6-10\t{dist['6-10']}", flush=True)
    print(f"11-20\t{dist['11-20']}", flush=True)
    print(f">20\t{dist['>20']}", flush=True)
    
    n_v = metrics['total']
    print(f"\nTC Cashout Txs Evaluated: {n_v}", flush=True)
    if n_v > 0:
        print(f"Rank@1  : {metrics['h1']} ({(metrics['h1']/n_v)*100:.2f}%)", flush=True)
        print(f"Rank@5  : {metrics['h5']} ({(metrics['h5']/n_v)*100:.2f}%)", flush=True)
        print(f"Rank@10 : {metrics['h10']} ({(metrics['h10']/n_v)*100:.2f}%)", flush=True)

def main():
    W = 200
    S = 50
    print(f"--- Running Experiment A: No Consistency Loss (lambda1=0) for W={W}, S={S} ---", flush=True)

    features_file = RESULTS_DIR / f"step02_features.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)

    labels_arr = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(records, test_size=0.2, stratify=labels_arr, random_state=42)

    train_ds = AccountWindowDataset(train_recs, W=W)
    val_ds   = AccountWindowDataset(val_recs,   W=W)

    labels_train = [item[2] for item in train_ds]
    n_phish  = sum(1 for l in labels_train if l == 1)
    n_normal = sum(1 for l in labels_train if l == 0)
    w_phish  = 1.0 / n_phish  if n_phish  > 0 else 1.0
    w_normal = 1.0 / n_normal if n_normal > 0 else 1.0
    sample_weights = [w_phish if l == 1 else w_normal for l in labels_train]

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=50000, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = GatedTMILETH(hand_crafted_dim=4, bert_dim=64, proj_dim=64, attn_hidden=128, mlp_hidden=256).to(DEVICE)
    loss_fn = GatedCompoundLoss(lambda1=0.0) # EXPERIMENT A: lambda1 = 0

    print("Phase 1: Warm-up", flush=True)
    model.freeze_bert()
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    best_val_auc = 0.0
    for epoch in range(1, 21):
        train_one_epoch(model, train_loader, loss_fn, optimizer1, DEVICE)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, loss_fn, DEVICE)
        from sklearn.metrics import roc_auc_score
        try: auc = roc_auc_score(val_labels, val_preds)
        except: auc = 0.0
        if auc > best_val_auc: best_val_auc = auc
        print(f"Phase 1 - Epoch {epoch} complete. Val AUC: {auc:.4f}", flush=True)

    print("Phase 2: Fine-tuning", flush=True)
    model.unfreeze_all()
    optimizer2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    for epoch in range(1, 31):
        train_one_epoch(model, train_loader, loss_fn, optimizer2, DEVICE)
        print(f"Phase 2 - Epoch {epoch} complete.", flush=True)

    model.eval()
    evaluate_on_tornado(model, records, W, S)

if __name__ == "__main__":
    main()
