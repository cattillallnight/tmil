"""
TMIL-ETH — Step 7: Two-Phase Training WITH TEMPORAL FEATURES
===============================================================
Extended from baseline to include temporal proximity features.

NEW: Uses GatedTMILETH_Temporal model with 4 temporal features:
  - exponential decay (24h half-life)
  - within_24h indicator
  - within_72h indicator  
  - log time gap

Input: step04b_features_with_temporal.pkl (output of step04b)
Model: GatedTMILETH_Temporal (from step05b_model_with_temporal.py)

Phase 1 (Warm-up, 20 epochs):
  - Freeze BERT4ETH feature projector
  - Train only MIL head (LR=1e-3)
  - Avoids catastrophic forgetting

Phase 2 (Fine-tuning, 30 epochs):
  - Unfreeze all parameters
  - Cosine annealing LR: 5e-5 -> 1e-6
  - Full end-to-end optimization

Optimizer: AdamW, weight_decay=1e-4, grad_clip=1.0
Batch size: 32 accounts

Saves: results/checkpoints/tmil_eth_final.pt
       results/figures/step04_train_baseline_curves.json
       results/figures/step04_train_baseline_curves.png
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
from tmil_architecture import GatedCompoundLoss
from step05b_model_with_temporal import GatedTMILETH_Temporal

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RESULTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Use features with temporal data from step04b
FEATURES_FILE = RESULTS_DIR / "step04b_features_with_temporal.pkl"

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
HAND_DIM     = 4
BERT_DIM     = 64


# ─── Dataset ─────────────────────────────────────────────────────────────────

class AccountWindowDataset(Dataset):
    """
    Each item is a single (window, label, account_label) triple.
    Window: fixed-size slice of hand_crafted + temporal + bert_embedding.
    Account label: y_A (phishing=1, normal=0).
    """
    def __init__(self, records: list, W: int = 200, augment: bool = False):
        self.W = W
        self.augment = augment
        self.items = []  # (hc_window, temp_window, bert_embed, y_A, addr)

        for rec in records:
            hc    = rec["hand_crafted"]         # (n_tx, 4)
            temp  = rec["temporal_features"]    # (n_tx, 4) - NEW!
            bert  = rec["bert_embedding"]       # (64,)
            y     = rec["label"]
            wins  = rec["windows"]

            for (start, end) in wins:
                hc_win = hc[start:end]          # (n, 4)
                temp_win = temp[start:end]      # (n, 4) - NEW!
                
                # Pad or truncate to W
                n = hc_win.shape[0]
                if n < W:
                    pad_hc = np.zeros((W - n, 4), dtype=np.float32)
                    pad_temp = np.zeros((W - n, 4), dtype=np.float32)  # NEW!
                    hc_win = np.vstack([hc_win, pad_hc])
                    temp_win = np.vstack([temp_win, pad_temp])         # NEW!
                elif n > W:
                    hc_win = hc_win[:W]
                    temp_win = temp_win[:W]                            # NEW!
                
                self.items.append((
                    hc_win.astype(np.float32),
                    temp_win.astype(np.float32),    # NEW!
                    bert.astype(np.float32),
                    y,
                    rec["address"],
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        hc, temp, bert, y, addr = self.items[idx]      # NEW: unpack temp
        return (
            torch.tensor(hc, dtype=torch.float32),      # (W, 4)
            torch.tensor(temp, dtype=torch.float32),    # (W, 4) - NEW!
            torch.tensor(bert, dtype=torch.float32),    # (64,)
            torch.tensor(y, dtype=torch.long),
        )


def collate_fn(batch):
    """
    hc_batch:   (B, W, 4)
    temp_batch: (B, W, 4) - NEW!
    bert_batch: (B, 64) -> broadcast to (B, W, 64)
    labels:     (B,)
    """
    hc = torch.stack([b[0] for b in batch])        # (B, W, 4)
    temp = torch.stack([b[1] for b in batch])      # (B, W, 4) - NEW!
    bert = torch.stack([b[2] for b in batch])      # (B, 64)
    labels = torch.stack([b[3] for b in batch])    # (B,)

    # Broadcast bert to (B, W, 64)
    W = hc.shape[1]
    bert_bcast = bert.unsqueeze(1).expand(-1, W, -1)  # (B, W, 64)
    return hc, temp, bert_bcast, labels               # NEW: return temp


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_epoch(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for hc, temp, bert, labels in loader:                              # NEW: unpack temp
            hc, temp, bert, labels = hc.to(device), temp.to(device), bert.to(device), labels.to(device)  # NEW: move temp to device
            p, _ = model(hc, temp, bert)                                   # NEW: pass temp to model
            l, _ = loss_fn(p, labels)
            total_loss += l.item()
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / max(len(loader), 1), np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, tau=0.5):
    from sklearn.metrics import (roc_auc_score, precision_score,
                                 recall_score, f1_score)
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
    for hc, temp, bert, labels in loader:                                  # NEW: unpack temp
        hc, temp, bert, labels = hc.to(device), temp.to(device), bert.to(device), labels.to(device)  # NEW: move temp to device
        optimizer.zero_grad()
        p, _ = model(hc, temp, bert)                                       # NEW: pass temp to model
        l, _ = loss_fn(p, labels)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        total_loss += l.item()
    return total_loss / max(len(loader), 1)


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 7: Two-Phase Training WITH TEMPORAL FEATURES")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load feature records
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run Step 2 first.")
        return

    print(f"\nLoading features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)
    print(f"  Total records: {len(records):,}")

    # Train/val split (80/20, stratified)
    labels_arr = [r["label"] for r in records]
    train_recs, val_recs = train_test_split(
        records, test_size=0.2, stratify=labels_arr, random_state=SEED
    )
    print(f"  Train: {len(train_recs):,} | Val: {len(val_recs):,}")
    print(f"  Train phishing: {sum(r['label'] for r in train_recs):,}")
    print(f"  Val phishing:   {sum(r['label'] for r in val_recs):,}")

    # Datasets
    train_ds = AccountWindowDataset(train_recs, W=W)
    val_ds   = AccountWindowDataset(val_recs,   W=W)

    # Class-weighted sampler for 1:4 imbalance
    labels_train = [item[2] for item in train_ds]
    n_phish  = sum(1 for l in labels_train if l == 1)
    n_normal = sum(1 for l in labels_train if l == 0)
    w_phish  = 1.0 / n_phish  if n_phish  > 0 else 1.0
    w_normal = 1.0 / n_normal if n_normal > 0 else 1.0
    sample_weights = [w_phish if l == 1 else w_normal for l in labels_train]

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"\n  Train windows: {len(train_ds):,}")
    print(f"  Val windows:   {len(val_ds):,}")

    # Model + loss
    model = GatedTMILETH_Temporal(                                         # NEW: use Temporal model
        hand_crafted_dim=HAND_DIM,
        temporal_dim=4,                                                    # NEW: temporal features
        bert_dim=BERT_DIM,
        proj_dim=64,
        attn_hidden=128,
        mlp_hidden=256
    ).to(device)
    loss_fn = GatedCompoundLoss(lambda1=LAMBDA1)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"phase1": {"train_loss": [], "val_loss": [], "val_auc": []},
               "phase2": {"train_loss": [], "val_loss": [], "val_auc": []}}

    # ── Phase 1: Warm-up (freeze feature projector) ───────────────
    print(f"\n{'='*40}")
    print(f"Phase 1: Warm-up ({PHASE1_EPOCHS} epochs, LR={LR_PHASE1})")
    print(f"  Freeze: feature_proj. Train: MIL head only.")
    print(f"{'='*40}")

    model.freeze_bert()
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
    )

    best_val_auc_p1 = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer1, device, GRAD_CLIP)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, loss_fn, device)
        metrics = compute_metrics(val_preds, val_labels)

        history["phase1"]["train_loss"].append(train_loss)
        history["phase1"]["val_loss"].append(val_loss)
        history["phase1"]["val_auc"].append(metrics["auc"])

        print(f"  [P1 Ep {epoch:02d}/{PHASE1_EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")

        if metrics["auc"] > best_val_auc_p1:
            best_val_auc_p1 = metrics["auc"]
            torch.save(model.state_dict(), CKPT_DIR / "tmil_phase1_best.pt")

    # ── Phase 2: Fine-tuning (unfreeze all) ───────────────────────
    print(f"\n{'='*40}")
    print(f"Phase 2: Fine-tuning ({PHASE2_EPOCHS} epochs, "
          f"LR cosine {LR_PHASE2_START:.0e} -> {LR_PHASE2_END:.0e})")
    print(f"  Unfreeze all parameters.")
    print(f"{'='*40}")

    model.unfreeze_all()
    optimizer2 = optim.AdamW(model.parameters(), lr=LR_PHASE2_START,
                              weight_decay=WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=PHASE2_EPOCHS, eta_min=LR_PHASE2_END
    )

    best_val_auc_p2 = 0.0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer2, device, GRAD_CLIP)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, loss_fn, device)
        metrics = compute_metrics(val_preds, val_labels)
        scheduler2.step()

        history["phase2"]["train_loss"].append(train_loss)
        history["phase2"]["val_loss"].append(val_loss)
        history["phase2"]["val_auc"].append(metrics["auc"])

        lr_now = scheduler2.get_last_lr()[0]
        print(f"  [P2 Ep {epoch:02d}/{PHASE2_EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  lr={lr_now:.2e}")

        if metrics["auc"] > best_val_auc_p2:
            best_val_auc_p2 = metrics["auc"]
            torch.save(model.state_dict(), CKPT_DIR / "tmil_eth_final.pt")

    print(f"\nBest Val AUC Phase 1: {best_val_auc_p1:.4f}")
    print(f"Best Val AUC Phase 2: {best_val_auc_p2:.4f}")

    # ── Plot training curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    p1_loss = history["phase1"]["train_loss"] + history["phase2"]["train_loss"]
    p1_ep   = list(range(1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1))
    axes[0].plot(range(1, PHASE1_EPOCHS+1), history["phase1"]["train_loss"],
                 "b-", label="Phase 1 Train Loss")
    axes[0].plot(range(1, PHASE1_EPOCHS+1), history["phase1"]["val_loss"],
                 "b--", label="Phase 1 Val Loss")
    axes[0].plot(range(PHASE1_EPOCHS+1, PHASE1_EPOCHS+PHASE2_EPOCHS+1),
                 history["phase2"]["train_loss"], "r-", label="Phase 2 Train Loss")
    axes[0].plot(range(PHASE1_EPOCHS+1, PHASE1_EPOCHS+PHASE2_EPOCHS+1),
                 history["phase2"]["val_loss"], "r--", label="Phase 2 Val Loss")
    axes[0].axvline(PHASE1_EPOCHS, color="gray", linestyle=":", alpha=0.7, label="Phase boundary")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("TMIL-ETH Training Loss")
    axes[0].legend(fontsize=8)

    # AUC
    axes[1].plot(range(1, PHASE1_EPOCHS+1), history["phase1"]["val_auc"],
                 "b-o", markersize=3, label="Phase 1 Val AUC")
    axes[1].plot(range(PHASE1_EPOCHS+1, PHASE1_EPOCHS+PHASE2_EPOCHS+1),
                 history["phase2"]["val_auc"], "r-o", markersize=3, label="Phase 2 Val AUC")
    axes[1].axvline(PHASE1_EPOCHS, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("TMIL-ETH Validation AUC")
    axes[1].legend()

    plt.tight_layout()
    curve_path = RESULTS_DIR / "step04_train_baseline_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nTraining curves saved: {curve_path.name}")

    # Save history
    hist_path = RESULTS_DIR / "step04_train_baseline_curves.json"
    training_summary = {
        "config": {
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "lr_phase1": LR_PHASE1,
            "lr_phase2_start": LR_PHASE2_START,
            "lr_phase2_end": LR_PHASE2_END,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "lambda1": LAMBDA1,
            "lambda2": LAMBDA2,
            "batch_size": BATCH_SIZE,
            "device": str(device),
        },
        "results": {
            "best_val_auc_phase1": best_val_auc_p1,
            "best_val_auc_phase2": best_val_auc_p2,
            "checkpoint": str(CKPT_DIR / "tmil_eth_final.pt"),
        },
        "history": history,
    }
    with open(hist_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training curves JSON: {hist_path.name}")

    print(f"\n[OK] Step 7 complete.\n")
    return training_summary


if __name__ == "__main__":
    main()
