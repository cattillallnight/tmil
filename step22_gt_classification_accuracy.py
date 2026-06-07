import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
step22_gt_classification_accuracy.py
=====================================
Kiểm tra: Với 2,483 phishing accounts có Ground Truth label,
model TMIL-ETH classify đúng bao nhiêu %?

Câu hỏi: Model có "phát hiện" được các GT accounts là phishing không?
Metrics:
  - Classification accuracy (p > 0.5 threshold)
  - AUC on GT subset vs same-size normal accounts
  - Per-tier breakdown (VERY_LARGE / LARGE / MEDIUM)
"""

import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from utils import RESULTS_DIR
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
from step07_training import AccountWindowDataset, collate_fn, train_one_epoch
from torch.utils.data import DataLoader

TMIL_DIR   = Path(__file__).parent
GT_FILE    = TMIL_DIR / "ground_truth" / "time_aware_ground_truth.json"
FEATURES   = RESULTS_DIR / "step2_features.pkl"
VAL_CSV    = TMIL_DIR / "validation" / "full_automated_validation.csv"

def main():
    print("=" * 70)
    print("Step 22: GT Account Classification Accuracy")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GT
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_addrs = {item["account_address"].lower() for item in gt_data}
    print(f"\n[1] GT accounts loaded: {len(gt_addrs)}")

    # Load validation CSV to get tier info
    df_val = pd.read_csv(VAL_CSV)
    df_val["phisher_address"] = df_val["phisher_address"].str.lower()
    tier_map = dict(zip(df_val["phisher_address"], df_val["tier"]))

    # Load features
    print("[2] Loading features...")
    with open(FEATURES, "rb") as f:
        all_records = pickle.load(f)

    gt_recs     = [r for r in all_records if r["address"].lower() in gt_addrs and r["label"] == 1]
    non_gt_phish= [r for r in all_records if r["address"].lower() not in gt_addrs and r["label"] == 1]
    normal_recs = [r for r in all_records if r["label"] == 0]

    print(f"  GT phishing records    : {len(gt_recs)}")
    print(f"  Non-GT phishing records: {len(non_gt_phish)}")
    print(f"  Normal records         : {len(normal_recs)}")

    # Train on non-GT phishers + normals (isolated from GT eval set)
    rng = np.random.RandomState(42)
    n_train_phish = min(2000, len(non_gt_phish))
    n_train_norm  = min(8000, len(normal_recs))

    train_phish = rng.choice(non_gt_phish, n_train_phish, replace=False).tolist()
    eval_normals = [r for r in normal_recs if r not in train_phish[:n_train_phish]]
    train_norm  = rng.choice(normal_recs, n_train_norm, replace=False).tolist()
    test_normals= rng.choice(normal_recs, min(len(gt_recs), len(normal_recs)), replace=False).tolist()

    train_recs = train_phish + train_norm
    print(f"\n[3] Training on {len(train_recs)} accounts ({n_train_phish} phish + {n_train_norm} normal)...")
    print("    GT accounts are COMPLETELY HELD OUT from training.")

    model = GatedTMILETH(4, 64).to(device)
    loss_fn = GatedCompoundLoss(lambda1=0.3)

    ds = AccountWindowDataset(train_recs, W=200)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # Phase 1
    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for ep in range(10):
        loss = train_one_epoch(model, loader, loss_fn, opt1, device, 1.0)
        if (ep+1) % 5 == 0:
            print(f"  Phase 1 Ep {ep+1}/10 | loss={loss:.4f}")

    # Phase 2
    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-6)
    for ep in range(15):
        loss = train_one_epoch(model, loader, loss_fn, opt2, device, 1.0)
        scheduler.step()
        if (ep+1) % 5 == 0:
            print(f"  Phase 2 Ep {ep+1}/15 | loss={loss:.4f}")

    # Evaluate on GT accounts
    print(f"\n[4] Evaluating on {len(gt_recs)} GT phishing accounts...")
    model.eval()

    def get_pred(rec):
        hc = torch.tensor(rec["hand_crafted"], dtype=torch.float32).to(device)
        bert = torch.tensor(rec["bert_embedding"], dtype=torch.float32).to(device)
        wins = rec["windows"]

        best_p = -1
        for start, end in wins:
            n = end - start
            hc_win = hc[start:end]
            if n < 200:
                pad = torch.zeros(200 - n, 4, device=device)
                hc_win = torch.cat([hc_win, pad], dim=0)
            else:
                hc_win = hc_win[:200]

            bert_exp = bert.unsqueeze(0).expand(200, -1)
            hc_b   = hc_win.unsqueeze(0)
            bert_b = bert_exp.unsqueeze(0)

            with torch.no_grad():
                p, _ = model(hc_b, bert_b)
            if p.item() > best_p:
                best_p = p.item()
        return best_p

    # GT phishing scores
    gt_scores = []
    for i, account_rec in enumerate(gt_recs):
        p = get_pred(account_rec)
        gt_scores.append(p)
        if (i+1) % 200 == 0:
            print(f"  GT progress: {i+1}/{len(gt_recs)}")

    # Normal scores (same size)
    norm_scores = []
    for i, account_rec in enumerate(test_normals):
        p = get_pred(account_rec)
        norm_scores.append(p)
        if (i+1) % 200 == 0:
            print(f"  Normal progress: {i+1}/{len(test_normals)}")

    # ── Metrics ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GT ACCOUNT CLASSIFICATION ACCURACY")
    print("=" * 70)

    y_true = [1] * len(gt_scores) + [0] * len(norm_scores)
    y_score= gt_scores + norm_scores
    y_pred = [1 if s > 0.5 else 0 for s in y_score]

    auc = roc_auc_score(y_true, y_score)
    f1  = f1_score(y_true, y_pred)
    prec= precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    acc = np.mean(np.array(y_pred) == np.array(y_true))

    gt_detected_pct = np.mean(np.array(gt_scores) > 0.5) * 100
    mean_gt_score   = np.mean(gt_scores)
    mean_norm_score = np.mean(norm_scores)

    print(f"\n  GT accounts (N={len(gt_scores)})")
    print(f"    % Classified as phishing (p>0.5) : {gt_detected_pct:.1f}%")
    print(f"    Mean phishing probability          : {mean_gt_score:.3f}")
    print(f"\n  Normal accounts (N={len(norm_scores)})")
    print(f"    Mean phishing probability          : {mean_norm_score:.3f}")

    print(f"\n  Combined AUC  : {auc:.4f}")
    print(f"  F1            : {f1:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  Accuracy      : {acc:.4f}")

    # Per-tier breakdown
    print(f"\n  [Per-Tier Breakdown on GT Phishing Accounts]")
    tier_results = {}
    for rec, score in zip(gt_recs, gt_scores):
        addr = rec["address"].lower()
        tier = tier_map.get(addr, "UNKNOWN")
        if tier not in tier_results:
            tier_results[tier] = []
        tier_results[tier].append(score)

    for tier in ["VERY_LARGE", "LARGE", "MEDIUM", "SMALL", "MICRO", "NO_DATA", "UNKNOWN"]:
        if tier in tier_results:
            scores_t = tier_results[tier]
            det_pct  = np.mean(np.array(scores_t) > 0.5) * 100
            mean_s   = np.mean(scores_t)
            print(f"  {tier:12} N={len(scores_t):4d} | Detected={det_pct:.1f}% | mean_p={mean_s:.3f}")

    print("=" * 70)

    # Save
    results_out = {
        "gt_n": len(gt_scores),
        "normal_n": len(norm_scores),
        "gt_detected_pct": round(gt_detected_pct, 2),
        "mean_gt_score": round(mean_gt_score, 4),
        "mean_norm_score": round(mean_norm_score, 4),
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "per_tier": {k: {"n": len(v), "detected_pct": round(np.mean(np.array(v)>0.5)*100, 1), "mean_p": round(np.mean(v), 3)} for k, v in tier_results.items()}
    }
    import json as jmod
    out_path = RESULTS_DIR / "step22_gt_classification_accuracy.json"
    with open(out_path, "w") as f:
        jmod.dump(results_out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
