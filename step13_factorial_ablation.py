import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utils import RESULTS_DIR
from step07_evaluate_baseline import load_transactions, PHISHER_TX_IN, PHISHER_TX_OUT
from step10_ctmil_experiment import CounterpartyDataset, collate_fn, train_epoch
from tmil_architecture import CounterpartyTMILETH

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, records):
    model.eval()
    with open(RESULTS_DIR / 'step16_etherscan_tc_hits.json', 'r') as f:
        tc_hits = json.load(f)
    tc_gt = {addr.lower(): set(tx['hash'].lower() for tx in txs) for addr, txs in tc_hits.items()}
    target_accounts = set(tc_gt.keys())
    
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    record_dict = {r['address']: r for r in records}
    
    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    
    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict: continue
        
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"]
        cp_ids = rec.get("counterparty_ids", np.zeros(hc.shape[0], dtype=np.int32))
        bert = rec.get("bert_embedding", np.zeros(64, dtype=np.float32))
        if bert.ndim == 1:
            bert = np.tile(bert, (hc.shape[0], 1))
            
        wins = rec["windows"]
        if hc.shape[0] != len(hashes): continue
        
        tx_attn_scores = np.zeros(len(hashes), dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            cp_win = cp_ids[start:end]
            bert_win = bert[start:end]
            out_win = rec.get("is_outbound", np.ones(hc.shape[0], dtype=bool))[start:end]
            n = hc_win.shape[0]
            
            hc_t = torch.tensor(hc_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            cp_t = torch.tensor(cp_win, dtype=torch.long).unsqueeze(0).to(DEVICE)
            bert_t = torch.tensor(bert_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out_t = torch.tensor(out_win, dtype=torch.bool).unsqueeze(0).to(DEVICE)
            mask_t = torch.ones((1, n), dtype=torch.bool).to(DEVICE)
            
            with torch.no_grad():
                _, attn = model(hc_t, cp_t, bert_t, mask=mask_t, outbound_mask=out_t)
            
            attn_scores = attn.squeeze(0).cpu().numpy()
            for i in range(n):
                tx_attn_scores[start+i] = max(tx_attn_scores[start+i], attn_scores[i])
                
        ranked_idx = np.argsort(tx_attn_scores)[::-1]
        ranked_hashes = [hashes[i] for i in ranked_idx]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total'] += 1
                if rank <= 1: metrics['h1'] += 1
                if rank <= 5: metrics['h5'] += 1
                if rank <= 10: metrics['h10'] += 1
                
    n = metrics['total']
    if n == 0: return 0, 0, 0
    return (metrics['h1']/n)*100, (metrics['h5']/n)*100, (metrics['h10']/n)*100

def main():
    print("=" * 60)
    print("Step 13: 2³ Factorial Ablation Study")
    print("=" * 60)
    
    print("Loading features...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
        
    phishers = [r for r in records if r['label'] == 1]
    normals = [r for r in records if r['label'] == 0]
    
    train_records = phishers[:-500] + normals[:-5000]
    train_ds = CounterpartyDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    results = []
    
    # 2^3 combinations
    for use_bert, use_cp, use_mask in itertools.product([False, True], repeat=3):
        config_name = f"BERT={use_bert} | CP={use_cp} | Mask={use_mask}"
        print(f"\n--- Running: {config_name} ---")
        
        model = CounterpartyTMILETH(use_bert=use_bert, use_cp=use_cp).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        epochs = 10
        for epoch in range(1, epochs + 1):
            model.train()
            bce = nn.BCEWithLogitsLoss()
            for hc_b, cp_b, out_b, bert_b, y_b, mask_b in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
                hc_b, cp_b, bert_b = hc_b.to(DEVICE), cp_b.to(DEVICE), bert_b.to(DEVICE)
                out_b, y_b, mask_b = out_b.to(DEVICE), y_b.to(DEVICE), mask_b.to(DEVICE)
                
                optimizer.zero_grad()
                # Apply mask if use_mask is True
                outbound_m = out_b if use_mask else None
                logits, attn = model(hc_b, cp_b, bert_b, mask=mask_b, outbound_mask=outbound_m)
                loss = bce(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Evaluate
        print("Evaluating...")
        h1, h5, h10 = evaluate_model(model, records)
        print(f"Hit@1: {h1:.2f}% | Hit@5: {h5:.2f}% | Hit@10: {h10:.2f}%")
        
        results.append({
            'BERT': use_bert,
            'Counterparty': use_cp,
            'Outbound_Mask': use_mask,
            'Hit@1': h1,
            'Hit@5': h5,
            'Hit@10': h10
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by=['BERT', 'Counterparty', 'Outbound_Mask'])
    df.to_csv(RESULTS_DIR / 'step13_factorial_ablation.csv', index=False)
    print("\n[OK] Factorial Ablation complete! Results saved to step13_factorial_ablation.csv")
    print(df)

if __name__ == "__main__":
    main()
