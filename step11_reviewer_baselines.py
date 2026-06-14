"""
Step 11: Reviewer Baselines
===========================
Generates a comprehensive comparison matrix for the paper:
1. Feature-Engineering Only (Random Forest on max-pooled handcrafted features)
2. 2x2 C-TMIL Ablation Matrix:
   - Base TMIL (no mask, no cp)
   - TMIL + Mask
   - TMIL + CP
   - C-TMIL Full (Mask + CP)
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utils import RESULTS_DIR
from step05_evaluate_baseline import load_transactions, PHISHER_TX_IN, PHISHER_TX_OUT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 50001
EMBED_DIM = 64

# --- Architecture ---

class CounterpartyTMILETH(nn.Module):
    def __init__(self, use_cp=True, use_mask=True, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hc_dim=4):
        super().__init__()
        self.use_cp = use_cp
        self.use_mask = use_mask
        
        # 1. Counterparty Embedding Table (trainable!)
        if self.use_cp:
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        # 2. Handcrafted Projection
        self.hc_norm = nn.LayerNorm(hc_dim)
        self.hc_proj = nn.Sequential(
            nn.Linear(hc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 3. Attention Mechanism (Gated)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_V = nn.Linear(embed_dim, 128)
        self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        
        # 4. Bag Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, hc, cp_ids, mask=None, outbound_mask=None):
        h_hc = self.hc_proj(self.hc_norm(hc)) # [B, W, 64]
        
        if self.use_cp:
            h_cp = self.cp_embed(cp_ids)        # [B, W, 64]
            h_fused = self.attn_norm(h_cp + h_hc) # [B, W, 64]
        else:
            h_fused = self.attn_norm(h_hc)
        
        # Gated Attention
        A_V = torch.tanh(self.attn_V(h_fused))
        A_U = torch.sigmoid(self.attn_U(h_fused))
        attn_scores = self.attn_w(A_V * A_U).squeeze(-1)  # [B, W]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        if self.use_mask and outbound_mask is not None:
            has_outbound = outbound_mask.any(dim=1, keepdim=True)
            mask_cond = has_outbound & ~outbound_mask
            attn_scores = attn_scores.masked_fill(mask_cond, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)      # [B, W]
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), h_fused).squeeze(1) # [B, 64]
        logits = self.classifier(bag_rep).squeeze(-1)     # [B]
        
        return logits, attn_weights

# --- Data ---

class CounterpartyDataset(Dataset):
    def __init__(self, records):
        self.samples = []
        for r in records:
            lbl = float(r["label"])
            hc = r["hand_crafted"]
            cp_ids = r.get("counterparty_ids", np.zeros(hc.shape[0], dtype=np.int32))
            is_out = r.get("is_outbound", np.ones(hc.shape[0], dtype=bool))
            wins = r["windows"]
            
            for (start, end) in wins:
                hc_win = hc[start:end]
                cp_win = cp_ids[start:end]
                out_win = is_out[start:end]
                self.samples.append((hc_win, cp_win, out_win, lbl))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    W = 200
    B = len(batch)
    
    hc_b = np.zeros((B, W, 4), dtype=np.float32)
    cp_b = np.zeros((B, W), dtype=np.longlong)
    y_b = np.zeros(B, dtype=np.float32)
    mask_b = np.zeros((B, W), dtype=bool)
    out_b = np.zeros((B, W), dtype=bool)
    
    for i, (hc_win, cp_win, out_win, lbl) in enumerate(batch):
        n = hc_win.shape[0]
        n_use = min(n, W)
        hc_b[i, :n_use, :] = hc_win[:n_use]
        cp_b[i, :n_use] = cp_win[:n_use]
        out_b[i, :n_use] = out_win[:n_use]
        mask_b[i, :n_use] = True
        y_b[i] = lbl
        
    return (
        torch.tensor(hc_b),
        torch.tensor(cp_b),
        torch.tensor(out_b),
        torch.tensor(y_b),
        torch.tensor(mask_b)
    )

# --- Training ---

def train_epoch(model, loader, optimizer, DEVICE):
    model.train()
    total_loss = 0
    bce = nn.BCEWithLogitsLoss()
    
    for hc_b, cp_b, out_b, y_b, mask_b in loader:
        hc_b, cp_b, out_b, y_b, mask_b = hc_b.to(DEVICE), cp_b.to(DEVICE), out_b.to(DEVICE), y_b.to(DEVICE), mask_b.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(hc_b, cp_b, mask_b, outbound_mask=out_b)
        loss = bce(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, records, tc_gt, tx_history, record_dict):
    model.eval()
    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    
    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict: continue
        
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"]
        cp_ids = rec.get("counterparty_ids", np.zeros(hc.shape[0], dtype=np.int32))
        wins = rec["windows"]
        
        if hc.shape[0] != len(hashes): continue
        tx_attn_scores = np.zeros(len(hashes), dtype=np.float32)
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            cp_win = cp_ids[start:end]
            out_win = rec.get("is_outbound", np.ones(hc.shape[0], dtype=bool))[start:end]
            n = hc_win.shape[0]
            
            hc_t = torch.tensor(hc_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            cp_t = torch.tensor(cp_win, dtype=torch.long).unsqueeze(0).to(DEVICE)
            out_t = torch.tensor(out_win, dtype=torch.bool).unsqueeze(0).to(DEVICE)
            mask_t = torch.ones((1, n), dtype=torch.bool).to(DEVICE)
            
            with torch.no_grad():
                _, attn = model(hc_t, cp_t, mask_t, outbound_mask=out_t)
            
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
                
    if metrics['total'] == 0: return 0, 0, 0
    return metrics['h1']/metrics['total'], metrics['h5']/metrics['total'], metrics['h10']/metrics['total']

def run_rf_baseline(records, tc_gt, tx_history, record_dict):
    print("Training Feature-Only RF...")
    X_train, y_train = [], []
    for r in records:
        hc = r["hand_crafted"] # [N, 4]
        if hc.shape[0] > 0:
            X_train.append(np.max(hc, axis=0)) # Max pooling
            y_train.append(r["label"])
            
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating RF on Localization...")
    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict: continue
        
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"]
        if hc.shape[0] != len(hashes): continue
        
        # Instance-level score = prob(phisher)
        scores = rf.predict_proba(hc)[:, 1]
        
        ranked_idx = np.argsort(scores)[::-1]
        ranked_hashes = [hashes[i] for i in ranked_idx]
        
        for ch in gt_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total'] += 1
                if rank <= 1: metrics['h1'] += 1
                if rank <= 5: metrics['h5'] += 1
                if rank <= 10: metrics['h10'] += 1
                
    if metrics['total'] == 0: return 0, 0, 0
    return metrics['h1']/metrics['total'], metrics['h5']/metrics['total'], metrics['h10']/metrics['total']

def main():
    print("Loading data...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
        
    phishers = [r for r in records if r['label'] == 1]
    normals = [r for r in records if r['label'] == 0]
    train_records = phishers[:-500] + normals[:-5000]
    
    with open(RESULTS_DIR / 'step16_etherscan_tc_hits.json', 'r') as f:
        tc_hits = json.load(f)
    tc_gt = {addr.lower(): set(tx['hash'].lower() for tx in txs) for addr, txs in tc_hits.items()}
    target_accounts = set(tc_gt.keys())
    
    tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
    record_dict = {r['address']: r for r in records}
    
    # Run RF Baseline
    rf_h1, rf_h5, rf_h10 = run_rf_baseline(train_records, tc_gt, tx_history, record_dict)
    
    # Run Ablation Matrix
    conditions = [
        ("TMIL Base (No CP, No Mask)", False, False),
        ("TMIL + Mask", False, True),
        ("TMIL + CP", True, False),
        ("C-TMIL Full", True, True)
    ]
    
    results = {}
    epochs = 15
    train_ds = CounterpartyDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    for name, use_cp, use_mask in conditions:
        print(f"\n--- Training {name} ---")
        model = CounterpartyTMILETH(use_cp=use_cp, use_mask=use_mask).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for ep in tqdm(range(epochs), desc=f"Epochs for {name}"):
            train_epoch(model, train_loader, optimizer, DEVICE)
            
        h1, h5, h10 = evaluate_model(model, records, tc_gt, tx_history, record_dict)
        results[name] = (h1, h5, h10)

    # Print Final Markdown Table
    print("\n\n" + "="*60)
    print("### Reviewer Evaluation Table: Hit@K Localization Performance")
    print("| Model Variant | Hit@1 | Hit@5 | Hit@10 |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| Feature-Only (Random Forest) | {rf_h1*100:.2f}% | {rf_h5*100:.2f}% | {rf_h10*100:.2f}% |")
    for name, _, _ in conditions:
        h1, h5, h10 = results[name]
        print(f"| {name} | {h1*100:.2f}% | {h5*100:.2f}% | {h10*100:.2f}% |")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
