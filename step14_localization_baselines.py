import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utils import RESULTS_DIR
from step07_evaluate_baseline import load_transactions, PHISHER_TX_IN, PHISHER_TX_OUT
from step10_ctmil_experiment import CounterpartyDataset, collate_fn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TTAGNProxy(nn.Module):
    """
    Temporal Transaction Aggregation Graph Network (Proxy for Localization)
    Mô phỏng kiến trúc: Mạng Temporal (GRU/LSTM) -> Self-Attention -> Classifier.
    Lớp Self-Attention sẽ được trích xuất trọng số để làm Localization Score.
    """
    def __init__(self, in_dim=64, hidden_dim=128):
        super().__init__()
        # Temporal Encoder
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Self-Attention on Temporal features
        self.attn_w = nn.Linear(hidden_dim * 2, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, bert_emb, mask=None):
        """
        bert_emb: [B, W, 64]
        mask: [B, W]
        """
        # Temporal processing
        h, _ = self.gru(bert_emb) # [B, W, 256]
        
        # Self-Attention
        scores = self.attn_w(h).squeeze(-1) # [B, W]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=1) # [B, W]
        
        # Aggregate
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1) # [B, 256]
        
        # Classify
        logits = self.classifier(bag_rep).squeeze(-1) # [B]
        
        return logits, attn_weights

def evaluate_localization(model, records, strategy="attention"):
    if model is not None:
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
        bert = rec.get("bert_embedding", np.zeros(64, dtype=np.float32))
        if bert.ndim == 1:
            bert = np.tile(bert, (hc.shape[0], 1))
            
        wins = rec["windows"]
        if hc.shape[0] != len(hashes): continue
        
        tx_attn_scores = np.zeros(len(hashes), dtype=np.float32)
        
        if strategy == "attention":
            for win_idx, (start, end) in enumerate(wins):
                bert_win = bert[start:end]
                n = bert_win.shape[0]
                
                bert_t = torch.tensor(bert_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                mask_t = torch.ones((1, n), dtype=torch.bool).to(DEVICE)
                
                with torch.no_grad():
                    _, attn = model(bert_t, mask=mask_t)
                
                attn_scores = attn.squeeze(0).cpu().numpy()
                for i in range(n):
                    tx_attn_scores[start+i] = max(tx_attn_scores[start+i], attn_scores[i])
                    
        elif strategy == "max_amount":
            # Baseline: Choose transaction with highest normalized amount (z_amount = hc[:, 0])
            for i in range(len(hashes)):
                tx_attn_scores[i] = hc[i, 0]
                
        elif strategy == "random":
            # Baseline: Random score
            tx_attn_scores = np.random.rand(len(hashes))

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
    print("Step 14: Localization Baselines (TTAGN-Proxy & Heuristics)")
    print("=" * 60)
    
    print("Loading features...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
        
    phishers = [r for r in records if r['label'] == 1]
    normals = [r for r in records if r['label'] == 0]
    
    train_records = phishers[:-500] + normals[:-5000]
    train_ds = CounterpartyDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    print("\n--- Training TTAGN Proxy ---")
    model = TTAGNProxy().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 10
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for hc_b, cp_b, out_b, bert_b, y_b, mask_b in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
            bert_b, y_b, mask_b = bert_b.to(DEVICE), y_b.to(DEVICE), mask_b.to(DEVICE)
            optimizer.zero_grad()
            logits, attn = model(bert_b, mask=mask_b)
            loss = bce(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    print("\nEvaluating Baselines...")
    h1, h5, h10 = evaluate_localization(model, records, strategy="attention")
    print(f"[TTAGN-Proxy]  Hit@1: {h1:5.2f}% | Hit@5: {h5:5.2f}% | Hit@10: {h10:5.2f}%")
    
    h1, h5, h10 = evaluate_localization(None, records, strategy="max_amount")
    print(f"[Max Amount]   Hit@1: {h1:5.2f}% | Hit@5: {h5:5.2f}% | Hit@10: {h10:5.2f}%")
    
    h1, h5, h10 = evaluate_localization(None, records, strategy="random")
    print(f"[Random]       Hit@1: {h1:5.2f}% | Hit@5: {h5:5.2f}% | Hit@10: {h10:5.2f}%")
    
if __name__ == "__main__":
    main()
