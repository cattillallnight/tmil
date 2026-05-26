import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

from utils import RESULTS_DIR
from gated_tmil import GatedTMILETH, GatedCompoundLoss

class AccountBagDataset(Dataset):
    """
    Dataset that yields an entire account as a bag of windows.
    Precomputes the 68-dim window features (4 hand-crafted mean + 64 BERT).
    """
    def __init__(self, records):
        self.items = []
        for rec in records:
            hc = rec["hand_crafted"]
            bert = rec["bert_embedding"]
            wins = rec["windows"]
            
            bag_features = []
            for (start, end) in wins:
                hc_win = hc[start:end]
                # Pad to at least 1 tx if empty (shouldn't happen, but safe)
                if hc_win.shape[0] == 0:
                    hc_win_mean = np.zeros(4, dtype=np.float32)
                else:
                    hc_win_mean = np.mean(hc_win, axis=0)
                
                win_feat = np.concatenate([hc_win_mean, bert]).astype(np.float32)
                bag_features.append(win_feat)
                
            self.items.append({
                "address": rec["address"],
                "bag_features": np.array(bag_features), # (K, 68)
                "label": rec["label"]
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return torch.tensor(item["bag_features"]), torch.tensor(item["label"], dtype=torch.long)

def collate_bag_fn(batch):
    """
    Pads bags of varying lengths K to max_K in the batch.
    Returns:
        x_pad: (B, max_K, 68)
        mask: (B, max_K) boolean, True for real, False for pad
        labels: (B,)
    """
    labels = torch.stack([b[1] for b in batch])
    feats = [b[0] for b in batch]
    
    max_k = max(f.shape[0] for f in feats)
    B = len(feats)
    D = feats[0].shape[1]
    
    x_pad = torch.zeros((B, max_k, D), dtype=torch.float32)
    mask = torch.zeros((B, max_k), dtype=torch.bool)
    
    for i, f in enumerate(feats):
        k = f.shape[0]
        x_pad[i, :k, :] = f
        mask[i, :k] = True
        
    return x_pad, mask, labels

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_pad, mask, labels in loader:
        x_pad, mask, labels = x_pad.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        p, _ = model(x_pad, mask)
        l, _ = loss_fn(p, labels)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += l.item()
    return total_loss / max(len(loader), 1)

def eval_model(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x_pad, mask, labels in loader:
            x_pad, mask = x_pad.to(device), mask.to(device)
            p, _ = model(x_pad, mask)
            probs.extend(p.cpu().numpy().tolist())
            ys.extend(labels.numpy().tolist())
    
    auc = roc_auc_score(ys, probs)
    binary = (np.array(probs) >= 0.5).astype(int)
    f1 = f1_score(ys, binary, zero_division=0)
    return auc, f1

def get_bag_bucket(N):
    if N == 1: return 0
    elif N <= 5: return 1
    elif N <= 50: return 2
    else: return 3

def main():
    print("=" * 70)
    print("TMIL-ETH: Ilse et al. Architecture (Step 17)")
    print("Gated Attention, No Triple Pooling, Bag-Size Stratified CV")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    features_file = RESULTS_DIR / "step2_features.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    print(f"Total accounts loaded: {len(records)}")
    
    # Stratification by label and bag_size bucket
    strat_keys = []
    labels = []
    for rec in records:
        L = rec["label"]
        N = len(rec["windows"])
        bucket = get_bag_bucket(N)
        strat_keys.append(f"{L}_{bucket}")
        labels.append(L)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_aucs, all_f1s = [], []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(records, strat_keys)):
        print(f"\n--- Fold {fold_idx+1}/5 ---")
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        
        train_ds = AccountBagDataset(train_recs)
        val_ds = AccountBagDataset(val_recs)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_bag_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_bag_fn, num_workers=0)
        
        model = GatedTMILETH(proj_dim=64, attn_hidden=128, mlp_hidden=256).to(device)
        loss_fn = GatedCompoundLoss(lambda1=0.5)
        
        # Phase 1: Freeze BERT projection
        model.freeze_bert()
        opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        for ep in range(10):
            train_one_epoch(model, train_loader, loss_fn, opt1, device)
            if (ep+1) % 5 == 0:
                print(f"  Phase1 Ep {ep+1}/10")
                
        # Phase 2: Unfreeze all
        model.unfreeze_all()
        opt2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=40, eta_min=1e-6)
        
        best_auc = 0.0
        best_f1 = 0.0
        
        for ep in range(40):
            train_one_epoch(model, train_loader, loss_fn, opt2, device)
            sched.step()
            
            if (ep+1) % 10 == 0:
                auc, f1 = eval_model(model, val_loader, device)
                if auc > best_auc:
                    best_auc = auc
                    best_f1 = f1
                print(f"  Phase2 Ep {ep+1}/40 | Val AUC: {auc:.4f} (best: {best_auc:.4f})")
                
        print(f"  Fold {fold_idx+1} FINAL: AUC={best_auc:.4f}, F1={best_f1:.4f}")
        all_aucs.append(best_auc)
        all_f1s.append(best_f1)
        
        # Only dry-run 1 fold for verification
        break
        
    print(f"\nDry Run Complete. AUC={best_auc:.4f}, F1={best_f1:.4f}")

if __name__ == "__main__":
    main()
