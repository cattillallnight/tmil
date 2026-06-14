"""
TMIL-ETH — Step 7b: Two-Phase Training (Hybrid Model)
=====================================================
Uses PG-EGAE MSE + Hand-crafted + BERT4ETH features.

Phase 1 (Warm-up, 20 epochs):
  - Freeze BERT4ETH feature projector
  - Train only MIL head (LR=1e-3)

Phase 2 (Fine-tuning, 30 epochs):
  - Unfreeze all parameters
  - Cosine annealing LR: 5e-5 -> 1e-6

Saves: results/checkpoints/tmil_hybrid_final.pt
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import RESULTS_DIR, sliding_windows, sidak_threshold
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RESULTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = RESULTS_DIR / "step02d_features_hybrid_norm.pkl"

# ─── Hyperparameters ─────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 32
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 30
LR_PHASE1    = 1e-3
LR_PHASE2_START = 5e-5
LR_PHASE2_END   = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
LAMBDA1      = 0.3
LAMBDA2      = 0.2
W, S         = 200, 50
HAND_DIM     = 5  # <-- Changed from 4 to 5 for Hybrid
BERT_DIM     = 64


# ─── Dataset ─────────────────────────────────────────────────────────────────

class AccountWindowDataset(Dataset):
    def __init__(self, records: list, W: int = 200, augment: bool = False):
        self.W = W
        self.augment = augment
        self.items = []

        for rec in records:
            hc    = rec["hand_crafted"]      # (n_tx, 5)
            bert  = rec["bert_embedding"]    # (64,)
            y     = rec["label"]
            wins  = rec["windows"]

            for (start, end) in wins:
                hc_win = hc[start:end]       # (n, 5)
                # Pad or truncate to W
                n = hc_win.shape[0]
                if n < W:
                    pad = np.zeros((W - n, hc_win.shape[1]), dtype=np.float32) # Fixed hardcoded 4
                    hc_win = np.vstack([hc_win, pad])
                elif n > W:
                    hc_win = hc_win[:W]
                self.items.append((
                    hc_win.astype(np.float32),
                    bert.astype(np.float32),
                    y,
                    rec["address"],
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        hc, bert, y, addr = self.items[idx]
        return (
            torch.tensor(hc, dtype=torch.float32),
            torch.tensor(bert, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


def collate_fn(batch):
    hc = torch.stack([b[0] for b in batch])
    bert = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])

    W = hc.shape[1]
    bert_bcast = bert.unsqueeze(1).expand(-1, W, -1)
    return hc, bert_bcast, labels


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_epoch(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for hc, bert, labels in loader:
            hc, bert, labels = hc.to(device), bert.to(device), labels.to(device)
            p, _ = model(hc, bert)
            l, _ = loss_fn(p, labels)
            total_loss += l.item()
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / max(len(loader), 1), np.array(all_preds), np.array(all_labels)

def compute_metrics(preds, labels, tau=0.5):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = 0.0
    binary = (preds >= tau).astype(int)
    p  = precision_score(labels, binary, zero_division=0)
    r  = recall_score(labels, binary, zero_division=0)
    f1 = f1_score(labels, binary, zero_division=0)
    return {"auc": auc, "precision": p, "recall": r, "f1": f1}


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device, clip_val=1.0):
    model.train()
    total_loss = 0.0
    for hc, bert, labels in loader:
        hc, bert, labels = hc.to(device), bert.to(device), labels.to(device)
        optimizer.zero_grad()
        p, _ = model(hc, bert)
        l, _ = loss_fn(p, labels)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        total_loss += l.item()
    return total_loss / max(len(loader), 1)

def main():
    print("=" * 60)
    print("TMIL-ETH - Step 7b: Hybrid Training")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run Step 02b first.")
        return

    print(f"\nLoading features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)
    print(f"  Total records: {len(records):,}")

    labels_arr = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(
        records, test_size=0.2, stratify=labels_arr, random_state=SEED
    )
    print(f"  Train: {len(train_recs):,} | Val: {len(val_recs):,}")

    train_ds = AccountWindowDataset(train_recs, W=W)
    val_ds   = AccountWindowDataset(val_recs,   W=W)

    labels_train = [item[2] for item in train_ds]
    n_phish  = sum(1 for l in labels_train if l == 1)
    n_normal = sum(1 for l in labels_train if l == 0)
    w_phish  = 1.0 / n_phish  if n_phish  > 0 else 1.0
    w_normal = 1.0 / n_normal if n_normal > 0 else 1.0
    sample_weights = [w_phish if l == 1 else w_normal for l in labels_train]

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model + loss
    model = GatedTMILETH(hand_crafted_dim=HAND_DIM, bert_dim=BERT_DIM,
                    proj_dim=64, attn_hidden=128, mlp_hidden=256).to(device)
    loss_fn = GatedCompoundLoss(lambda1=LAMBDA1)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"phase1": {"train_loss": [], "val_loss": [], "val_auc": []},
               "phase2": {"train_loss": [], "val_loss": [], "val_auc": []}}

    # ── Phase 1: Warm-up
    print(f"\nPhase 1: Warm-up ({PHASE1_EPOCHS} epochs)")
    model.freeze_bert()
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)

    best_val_auc_p1 = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer1, device, GRAD_CLIP)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, loss_fn, device)
        metrics = compute_metrics(val_preds, val_labels)

        history["phase1"]["train_loss"].append(train_loss)
        history["phase1"]["val_loss"].append(val_loss)
        history["phase1"]["val_auc"].append(metrics["auc"])

        print(f"  [P1 Ep {epoch:02d}] t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  v_AUC={metrics['auc']:.4f}")
        if metrics["auc"] > best_val_auc_p1:
            best_val_auc_p1 = metrics["auc"]
            torch.save(model.state_dict(), CKPT_DIR / "tmil_hybrid_phase1_best.pt")

    # ── Phase 2: Fine-tuning
    print(f"\nPhase 2: Fine-tuning ({PHASE2_EPOCHS} epochs)")
    model.unfreeze_all()
    optimizer2 = optim.AdamW(model.parameters(), lr=LR_PHASE2_START, weight_decay=WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS, eta_min=LR_PHASE2_END)

    best_val_auc_p2 = 0.0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer2, device, GRAD_CLIP)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, loss_fn, device)
        metrics = compute_metrics(val_preds, val_labels)
        scheduler2.step()

        history["phase2"]["train_loss"].append(train_loss)
        history["phase2"]["val_loss"].append(val_loss)
        history["phase2"]["val_auc"].append(metrics["auc"])

        print(f"  [P2 Ep {epoch:02d}] t_loss={train_loss:.4f}  v_loss={val_loss:.4f}  v_AUC={metrics['auc']:.4f}")
        if metrics["auc"] > best_val_auc_p2:
            best_val_auc_p2 = metrics["auc"]
            torch.save(model.state_dict(), CKPT_DIR / "tmil_hybrid_final.pt")

    print(f"\nBest Val AUC Phase 1: {best_val_auc_p1:.4f}")
    print(f"Best Val AUC Phase 2: {best_val_auc_p2:.4f}")
    print("\n[OK] Step 7b complete.\n")

if __name__ == "__main__":
    main()
