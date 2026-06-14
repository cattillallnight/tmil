"""
Step 25c: Evaluate Hybrid & Original TMIL Account-Level Metrics
===============================================================
Calculates AUC-ROC and FPR@95%TPR for both models on the validation set.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step05_model_architecture import GatedTMILETH
from step07b_training_hybrid import AccountWindowDataset, collate_fn, W

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR = Path(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results")

def get_fpr_at_95_tpr(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return 1.0

def evaluate_model(model, loader, name):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for hc, bert, labels in loader:
            hc, bert, labels = hc.to(DEVICE), bert.to(DEVICE), labels.to(DEVICE)
            p, _ = model(hc, bert)
            all_preds.extend(p.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    auc = roc_auc_score(labels, preds)
    binary = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, binary, zero_division=0)
    prec = precision_score(labels, binary, zero_division=0)
    rec = recall_score(labels, binary, zero_division=0)
    fpr95 = get_fpr_at_95_tpr(labels, preds)

    print(f"\n[{name}]")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"FPR@95%TPR:  {fpr95:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")

def main():
    feat_orig = RESULTS_DIR / "step02_features.pkl"
    feat_hybr = RESULTS_DIR / "step02d_features_hybrid_norm.pkl"
    ckpt_orig = RESULTS_DIR / "checkpoints" / "tmil_eth_final.pt"
    ckpt_hybr = RESULTS_DIR / "checkpoints" / "tmil_hybrid_final.pt"

    print("Loading Original Validation Set...")
    with open(feat_orig, "rb") as f:
        recs_orig = pickle.load(f)
    labels_orig = [r["label"] for r in recs_orig]
    _, val_orig = train_test_split(recs_orig, test_size=0.2, stratify=labels_orig, random_state=42)
    loader_orig = DataLoader(AccountWindowDataset(val_orig, W=W), batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("Loading Hybrid Validation Set...")
    with open(feat_hybr, "rb") as f:
        recs_hybr = pickle.load(f)
    labels_hybr = [r["label"] for r in recs_hybr]
    _, val_hybr = train_test_split(recs_hybr, test_size=0.2, stratify=labels_hybr, random_state=42)
    loader_hybr = DataLoader(AccountWindowDataset(val_hybr, W=W), batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("\nEvaluating Original TMIL Model (68-dim)...")
    model_orig = GatedTMILETH(hand_crafted_dim=4, bert_dim=64).to(DEVICE)
    model_orig.load_state_dict(torch.load(ckpt_orig, map_location=DEVICE, weights_only=True))
    model_orig.eval()
    evaluate_model(model_orig, loader_orig, "Original TMIL")

    print("\nEvaluating Hybrid TMIL Model (69-dim)...")
    model_hybr = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model_hybr.load_state_dict(torch.load(ckpt_hybr, map_location=DEVICE, weights_only=True))
    model_hybr.eval()
    evaluate_model(model_hybr, loader_hybr, "Hybrid TMIL")

if __name__ == "__main__":
    main()
