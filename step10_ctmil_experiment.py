"""
Step 15b: Counterparty-Aware TMIL-ETH (C-TMIL)
==============================================
Dual-stream architecture using both Handcrafted features and Counterparty Embeddings.
Solves the transaction-level context blindness of account-level BERT.
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utils import RESULTS_DIR
from step07_evaluate_baseline import load_transactions, PHISHER_TX_IN, PHISHER_TX_OUT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 50001
EMBED_DIM = 64

# --- Architecture ---

class CounterpartyTMILETH(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hc_dim=4):
        super().__init__()
        
        # 1. Counterparty Embedding Table (trainable!)
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
        """
        hc: [B, W, 4]
        cp_ids: [B, W] (int)
        mask: [B, W] (bool)
        outbound_mask: [B, W] (bool)
        """
        # Embeddings
        h_cp = self.cp_embed(cp_ids)        # [B, W, 64]
        h_hc = self.hc_proj(self.hc_norm(hc)) # [B, W, 64]
        
        # Dual-Stream Fusion (Additive)
        h_fused = self.attn_norm(h_cp + h_hc) # [B, W, 64]
        
        # Gated Attention
        A_V = torch.tanh(self.attn_V(h_fused))
        A_U = torch.sigmoid(self.attn_U(h_fused))
        attn_scores = self.attn_w(A_V * A_U).squeeze(-1)  # [B, W]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        if outbound_mask is not None:
            # Prevent attention on inbound txs, but avoid NaN if window has NO outbound txs
            has_outbound = outbound_mask.any(dim=1, keepdim=True)
            mask_cond = has_outbound & ~outbound_mask
            attn_scores = attn_scores.masked_fill(mask_cond, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)      # [B, W]
        
        # Aggregate
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), h_fused).squeeze(1) # [B, 64]
        
        # Classify
        logits = self.classifier(bag_rep).squeeze(-1)     # [B]
        
        return logits, attn_weights

# --- Data ---

class CounterpartyDataset(Dataset):
    def __init__(self, records, is_train=True):
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

    def __getitem__(self, idx):
        return self.samples[idx]

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
    
    for hc_b, cp_b, out_b, y_b, mask_b in tqdm(loader, desc="Train", leave=False):
        hc_b = hc_b.to(DEVICE)
        cp_b = cp_b.to(DEVICE)
        out_b = out_b.to(DEVICE)
        y_b = y_b.to(DEVICE)
        mask_b = mask_b.to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(hc_b, cp_b, mask_b, outbound_mask=out_b)
        loss = bce(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print("=" * 60)
    print("Step 15b: Train & Evaluate Counterparty-Aware TMIL")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading features...")
    with open(RESULTS_DIR / 'step02_features.pkl', 'rb') as f:
        records = pickle.load(f)
        
    phishers = [r for r in records if r['label'] == 1]
    normals = [r for r in records if r['label'] == 0]
    
    train_records = phishers[:-500] + normals[:-5000]
    
    train_ds = CounterpartyDataset(train_records)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # 2. Build Model
    model = CounterpartyTMILETH().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 3. Train
    epochs = 15
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")
        
    # 4. Evaluate Hit@1
    print("\nEvaluating Hit@1 on TC Cashouts...")
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
                
    print("\n==================================================")
    print("C-TMIL (COUNTERPARTY AWARE) RESULTS")
    print("==================================================")
    n = metrics['total']
    if n > 0:
        print(f"TC Cashout Txs Evaluated: {n}")
        print(f"Rank@1  : {metrics['h1']} ({(metrics['h1']/n)*100:.2f}%)")
        print(f"Rank@5  : {metrics['h5']} ({(metrics['h5']/n)*100:.2f}%)")
        print(f"Rank@10 : {metrics['h10']} ({(metrics['h10']/n)*100:.2f}%)")

if __name__ == "__main__":
    main()
