import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from utils import RESULTS_DIR
from step06_train_baseline import evaluate_epoch, train_one_epoch
from step07_evaluate_baseline import load_transactions
from utils import PHISHER_TX_IN, PHISHER_TX_OUT, sliding_windows
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. FIXED MODEL WITH DECOUPLED PROJECTION ---
class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, N, D = x.shape
        if N == 1:
            z = x.squeeze(1)
            attn = torch.ones(B, 1, device=x.device)
            return z, attn

        tanh_V = torch.tanh(self.V(x))
        sigm_U = torch.sigmoid(self.U(x))
        gated_h = tanh_V * sigm_U
        
        scores = self.w(gated_h).squeeze(-1)
        
        # BUG FIX 1: Apply Mask to prevent padding dilution
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        
        # We don't multiply here because value is decoupled
        return attn

class DecoupledMaskedTMILETH(nn.Module):
    def __init__(self,
                 hand_crafted_dim: int = 4,
                 bert_dim: int = 64,
                 proj_dim: int = 64,
                 attn_hidden: int = 128,
                 mlp_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # BUG FIX 2: Decoupled Projection
        # Attention only looks at hand_crafted features
        self.attn_proj = nn.Sequential(
            nn.Linear(hand_crafted_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.attention = GatedAttentionMIL(input_dim=proj_dim, hidden_dim=attn_hidden)
        
        # Value (Classifier) only looks at BERT embedding
        self.value_proj = nn.Sequential(
            nn.Linear(bert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, hand_crafted: torch.Tensor, bert_embed: torch.Tensor, mask: torch.Tensor = None):
        # 1. Compute Attention from hand_crafted
        attn_h = self.attn_proj(hand_crafted)
        attn = self.attention(attn_h, mask) # (B, W)
        
        # 2. Extract Value from BERT
        # bert_embed is broadcasted to (B, W, 64)
        v = self.value_proj(bert_embed) # (B, W, proj_dim)
        
        # 3. Aggregate Value using Attention
        z = torch.bmm(attn.unsqueeze(1), v).squeeze(1) # (B, proj_dim)
        
        logit = self.classifier(z).squeeze(-1)
        p_window = torch.sigmoid(logit)
        return p_window, attn

    def freeze_bert(self):
        for param in self.value_proj.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


from step05_model_architecture import GatedCompoundLoss

# --- 2. DATASET & COLLATE FIX ---
class AccountWindowDatasetMasked(Dataset):
    def __init__(self, records: list, W: int = 200):
        self.items = []
        for rec in records:
            hc    = rec["hand_crafted"]      
            bert  = rec["bert_embedding"]    
            y     = rec["label"]
            wins  = rec["windows"]
            for (start, end) in wins:
                hc_win = hc[start:end]
                n = hc_win.shape[0]
                mask = np.ones(W, dtype=bool)
                if n < W:
                    pad = np.zeros((W - n, 4), dtype=np.float32)
                    hc_win = np.vstack([hc_win, pad])
                    mask[n:] = False
                elif n > W:
                    hc_win = hc_win[:W]
                self.items.append((
                    hc_win.astype(np.float32),
                    bert.astype(np.float32),
                    y,
                    mask
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        hc, bert, y, mask = self.items[idx]
        return (
            torch.tensor(hc, dtype=torch.float32),
            torch.tensor(bert, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool)
        )

def collate_fn_masked(batch):
    hc = torch.stack([b[0] for b in batch])
    bert = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    mask = torch.stack([b[3] for b in batch])
    
    W = hc.shape[1]
    bert_bcast = bert.unsqueeze(1).expand(-1, W, -1)
    return hc, bert_bcast, labels, mask

# Override train/eval to pass mask
def train_one_epoch_masked(model, loader, loss_fn, optimizer, device, clip_val=1.0):
    model.train()
    total_loss = 0.0
    for hc, bert, labels, mask in loader:
        hc, bert, labels, mask = hc.to(device), bert.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        p, _ = model(hc, bert, mask)
        l, _ = loss_fn(p, labels)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        total_loss += l.item()
    return total_loss / max(len(loader), 1)

def evaluate_epoch_masked(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for hc, bert, labels, mask in loader:
            hc, bert, labels, mask = hc.to(device), bert.to(device), labels.to(device), mask.to(device)
            p, _ = model(hc, bert, mask)
            l, _ = loss_fn(p, labels)
            total_loss += l.item()
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / max(len(loader), 1), np.array(all_preds), np.array(all_labels)

# --- 3. EVALUATION SCRIPT ---
def evaluate_on_tornado_masked(model, records, W, S):
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
            mask_np = np.ones(W, dtype=bool)
            if n < W:
                pad = np.zeros((W - n, 4), dtype=np.float32)
                hc_win_pad = np.vstack([hc_win, pad])
                mask_np[n:] = False
            else:
                hc_win_pad = hc_win[:W]
                
            hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, W, -1).to(DEVICE)
            mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                p, attn = model(hc_t, bert_t, mask_t)
                
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

def evaluate_on_scamsniffer_masked(model, W, S):
    dataset_file = RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json"
    if not dataset_file.exists():
        dataset_file = RESULTS_DIR / "scamsniffer_txlevel_dataset.json"
    with open(dataset_file, "r") as f:
        ss_data = json.load(f)
        
    embed_file = RESULTS_DIR / "step28_scamsniffer_bert_embeddings.pkl"
    if not embed_file.exists():
        print("Missing BERT embeddings for ScamSniffer Pilot. Skipping.")
        return
        
    with open(embed_file, "rb") as f:
        addr2emb = pickle.load(f)
        
    metrics = {'total': 0, 'h1': 0, 'h5': 0, 'h10': 0}
    
    for rec in ss_data:
        addr = rec.get('address', '').lower()
        if addr not in addr2emb: continue
        
        hashes = [h.lower() for h in rec.get('hashes', [])]
        victim_hashes = set(tx['hash'].lower() for tx in rec.get('victim_txs', []))
        if not victim_hashes: continue
        
        bert_emb = addr2emb[addr] # (N, 64)
        n_expected = len(hashes)
        
        hc = rec.get('hand_crafted', [])
        if len(hc) != n_expected: continue
        hc = np.array(hc, dtype=np.float32)
        
        wins = sliding_windows(n_expected, W=W, S=S)
        tx_attn_scores = np.zeros(n_expected, dtype=np.float32)
        
        for start, end in wins:
            hc_win = hc[start:end]
            bert_win = bert_emb[start:end]
            n = hc_win.shape[0]
            
            mask_np = np.ones(W, dtype=bool)
            if n < W:
                pad_hc = np.zeros((W - n, 4), dtype=np.float32)
                pad_bert = np.zeros((W - n, 64), dtype=np.float32)
                hc_win_pad = np.vstack([hc_win, pad_hc])
                bert_win_pad = np.vstack([bert_win, pad_bert])
                mask_np[n:] = False
            else:
                hc_win_pad = hc_win[:W]
                bert_win_pad = bert_win[:W]
                
            hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bert_t = torch.tensor(bert_win_pad, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                p, attn = model(hc_t, bert_t, mask_t)
                
            attn_scores = attn.squeeze(0).cpu().numpy()[:n]
            for i in range(n):
                tx_idx = start + i
                tx_attn_scores[tx_idx] = max(tx_attn_scores[tx_idx], attn_scores[i])
                
        if np.sum(tx_attn_scores) > 0:
            tx_attn_scores = tx_attn_scores / np.sum(tx_attn_scores)
            
        addr_tx_attn = {hashes[i]: tx_attn_scores[i] for i in range(n_expected)}
        ranked_txs = sorted(addr_tx_attn.items(), key=lambda x: x[1], reverse=True)
        ranked_hashes = [x[0] for x in ranked_txs]
        
        for ch in victim_hashes:
            if ch in ranked_hashes:
                rank = ranked_hashes.index(ch) + 1
                metrics['total'] += 1
                if rank <= 1: metrics['h1'] += 1
                if rank <= 5: metrics['h5'] += 1
                if rank <= 10: metrics['h10'] += 1

    print("\n==================================================")
    print("SCAMSNIFFER EVALUATION RESULTS")
    print("==================================================")
    n_v = metrics['total']
    if n_v > 0:
        print(f"ScamSniffer Victim Txs Evaluated: {n_v}")
        print(f"Rank@1  : {metrics['h1']} ({(metrics['h1']/n_v)*100:.2f}%)")
        print(f"Rank@5  : {metrics['h5']} ({(metrics['h5']/n_v)*100:.2f}%)")
        print(f"Rank@10 : {metrics['h10']} ({(metrics['h10']/n_v)*100:.2f}%)")
    else:
        print("No pilot accounts evaluated.")

def main():
    W = 200
    S = 50
    print(f"--- Step 11: Decoupled Projection + Fix Mask Padding (W={W}, S={S}) ---", flush=True)

    features_file = RESULTS_DIR / f"step02_features.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)

    labels_arr = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(records, test_size=0.2, stratify=labels_arr, random_state=42)

    train_ds = AccountWindowDatasetMasked(train_recs, W=W)
    val_ds   = AccountWindowDatasetMasked(val_recs,   W=W)

    labels_train = [item[2] for item in train_ds]
    n_phish  = sum(1 for l in labels_train if l == 1)
    n_normal = sum(1 for l in labels_train if l == 0)
    w_phish  = 1.0 / n_phish  if n_phish  > 0 else 1.0
    w_normal = 1.0 / n_normal if n_normal > 0 else 1.0
    sample_weights = [w_phish if l == 1 else w_normal for l in labels_train]

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=50000, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate_fn_masked, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn_masked, num_workers=0)

    model = DecoupledMaskedTMILETH(hand_crafted_dim=4, bert_dim=64, proj_dim=64, attn_hidden=128, mlp_hidden=256).to(DEVICE)
    loss_fn = GatedCompoundLoss(lambda1=0.3) 

    print("Phase 1: Warm-up", flush=True)
    model.freeze_bert()
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    best_val_auc = 0.0
    for epoch in range(1, 21):
        train_one_epoch_masked(model, train_loader, loss_fn, optimizer1, DEVICE)
        val_loss, val_preds, val_labels = evaluate_epoch_masked(model, val_loader, loss_fn, DEVICE)
        from sklearn.metrics import roc_auc_score
        try: auc = roc_auc_score(val_labels, val_preds)
        except: auc = 0.0
        if auc > best_val_auc: best_val_auc = auc
        print(f"Phase 1 - Epoch {epoch} complete. Val AUC: {auc:.4f}", flush=True)

    print("Phase 2: Fine-tuning", flush=True)
    model.unfreeze_all()
    optimizer2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    for epoch in range(1, 31):
        train_one_epoch_masked(model, train_loader, loss_fn, optimizer2, DEVICE)
        print(f"Phase 2 - Epoch {epoch} complete.", flush=True)

    model.eval()
    evaluate_on_tornado_masked(model, records, W, S)
    evaluate_on_scamsniffer_masked(model, W, S)

if __name__ == "__main__":
    main()
