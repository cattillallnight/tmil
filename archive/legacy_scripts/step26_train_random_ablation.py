"""
Step 26: Ablation Study - Random Feature Control
================================================
Trains the TMIL Hybrid model, but replaces the 5th feature (PG-EGAE MSE)
with random noise. This proves that the *content* of the feature drives
the 32% improvement, not just the increased dimensionality (68 -> 69).
"""

import sys
import os
import json
import pickle
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
from collections import defaultdict
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
SEED = 42
BATCH_SIZE = 32
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 30
W = 200

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

class RandomFeatureDataset(Dataset):
    def __init__(self, records: list, W: int = 200):
        self.W = W
        self.items = []
        for rec in records:
            # Copy to avoid modifying original in memory
            hc = rec["hand_crafted"].copy()
            # Replace 5th dim with Random Uniform noise [0, 1]
            noise = np.random.uniform(0, 1, size=hc.shape[0])
            hc[:, 4] = noise
            
            bert = rec["bert_embedding"]
            y = rec["label"]
            wins = rec["windows"]

            for (start, end) in wins:
                hc_win = hc[start:end]
                n = hc_win.shape[0]
                if n < W:
                    pad = np.zeros((W - n, 5), dtype=np.float32)
                    hc_win = np.vstack([hc_win, pad])
                elif n > W:
                    hc_win = hc_win[:W]
                self.items.append((
                    hc_win.astype(np.float32),
                    bert.astype(np.float32),
                    y
                ))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        hc, bert, y = self.items[idx]
        return torch.tensor(hc, dtype=torch.float32), torch.tensor(bert, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    hc = torch.stack([b[0] for b in batch])
    bert = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    bert_bcast = bert.unsqueeze(1).expand(-1, hc.shape[1], -1)
    return hc, bert_bcast, labels

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for hc, bert, labels in loader:
        hc, bert, labels = hc.to(DEVICE), bert.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        p, _ = model(hc, bert)
        l, _ = loss_fn(p, labels)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += l.item()
    return total_loss / max(len(loader), 1)

def evaluate_epoch(model, loader, loss_fn):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for hc, bert, labels in loader:
            hc, bert, labels = hc.to(DEVICE), bert.to(DEVICE), labels.to(DEVICE)
            p, _ = model(hc, bert)
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    return auc

def run_seed(seed, records):
    print(f"\n==============================================")
    print(f"RUNNING RANDOM FEATURE ABLATION WITH SEED: {seed}")
    print(f"==============================================")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    labels = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(records, test_size=0.2, stratify=labels, random_state=seed)
    
    train_ds = RandomFeatureDataset(train_recs, W)
    val_ds = RandomFeatureDataset(val_recs, W)
    
    labels_train = [item[2] for item in train_ds]
    w_phish = 1.0 / sum(1 for l in labels_train if l == 1)
    w_normal = 1.0 / sum(1 for l in labels_train if l == 0)
    sample_weights = [w_phish if l == 1 else w_normal for l in labels_train]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    loss_fn = GatedCompoundLoss(lambda1=0.3)
    
    print(f"\nPhase 1: Warm-up (Seed {seed})")
    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    best_auc = 0
    for ep in range(1, PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, loss_fn, opt1)
        auc = evaluate_epoch(model, val_loader, loss_fn)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), RESULTS_DIR / "checkpoints" / f"tmil_random_p1_seed{seed}.pt")
            
    print(f"Phase 1 Best Val AUC: {best_auc:.4f}")
        
    print(f"\nPhase 2: Fine-tuning (Seed {seed})")
    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    best_auc = 0
    for ep in range(1, PHASE2_EPOCHS + 1):
        train_one_epoch(model, train_loader, loss_fn, opt2)
        auc = evaluate_epoch(model, val_loader, loss_fn)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), RESULTS_DIR / "checkpoints" / f"tmil_random_final_seed{seed}.pt")
            
    print(f"Phase 2 Best Val AUC: {best_auc:.4f}")
        
    # Evaluate
    model.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / f"tmil_random_final_seed{seed}.pt", weights_only=True))
    model.eval()
    evaluate_on_tornado_cash(model, records, seed)


def evaluate_on_tornado_cash(model, records, seed):
    print(f"\n--- Evaluating Random Feature Model on Tornado Cash (Seed {seed}) ---")
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
    all_c_attn, all_n_attn = [], []

    for addr, gt_hashes in tc_gt.items():
        if addr not in tx_history or addr not in record_dict: continue
        tx_list = tx_history[addr]
        hashes = [tx[4] for tx in tx_list]
        rec = record_dict[addr]
        
        hc = rec["hand_crafted"].copy()
        # Ensure deterministic noise for eval per seed
        np.random.seed(seed)
        noise = np.random.uniform(0, 1, size=hc.shape[0])
        hc[:, 4] = noise
        
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        
        n_expected = len(hashes)
        if hc.shape[0] != n_expected: continue
            
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
                
    n_v = metrics['total']
    print(f"Total TC Cashout Evaluated: {n_v}")
    if n_v > 0:
        print(f"Rank@1  : {metrics['h1']} ({(metrics['h1']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['h10']} ({(metrics['h10']/n_v)*100:.2f}%)")
        mean_c = np.mean(all_c_attn) if all_c_attn else 0
        mean_n = np.mean(all_n_attn) if all_n_attn else 0
        print(f"Mean TC Attn: {mean_c:.6f}")
        print(f"Mean Normal Attn: {mean_n:.6f}")
        if mean_n > 0:
            print(f"Enrichment Ratio: {mean_c/mean_n:.2f}x")

def main():
    print("Loading normalized features...")
    features_file = RESULTS_DIR / "step02d_features_hybrid_norm.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    for s in [44]:
        run_seed(s, records)

if __name__ == "__main__":
    main()

