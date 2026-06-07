"""
step23_gt_targeted_cv.py  (v2 - fixed)
=========================
Nested CV với GT accounts được đưa vào training.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json, pickle, numpy as np, torch, torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader

from utils import RESULTS_DIR
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
from step07_training import AccountWindowDataset, collate_fn, train_one_epoch

FEATURES_FILE  = RESULTS_DIR / "step2_features.pkl"
GT_FILE        = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
VAL_CSV        = Path(__file__).parent / "validation" / "full_automated_validation.csv"

SEED         = 42
OUTER_FOLDS  = 5
BATCH_SIZE   = 64
PHASE1_EP    = 15
PHASE2_EP    = 20
LR1, LR2, LR2_MIN = 1e-3, 5e-5, 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
W            = 200
LAMBDA1_BEST = 0.3   # Pre-selected from step9

def fpr_at_95tpr(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 1.0
    fpr_a, tpr_a, _ = roc_curve(y_true, y_score)
    return float(fpr_a[np.argmin(np.abs(tpr_a - 0.95))])

def qmetrics(y_true, y_score, tau=0.5):
    yt = np.array(y_true)
    ys = np.array(y_score)
    try:
        auc = float(roc_auc_score(yt, ys))
    except:
        auc = 0.0
    b = (ys >= tau).astype(int)
    return {
        "auc": auc,
        "f1": float(f1_score(yt, b, zero_division=0)),
        "precision": float(precision_score(yt, b, zero_division=0)),
        "recall": float(recall_score(yt, b, zero_division=0)),
        "fpr_at_95tpr": fpr_at_95tpr(yt.tolist(), ys.tolist()),
    }

def train_model(tr_recs, lambda1, device):
    loss_fn = GatedCompoundLoss(lambda1=lambda1)
    model   = GatedTMILETH(4, 64).to(device)
    tr_ds   = AccountWindowDataset(tr_recs, W=W)
    tr_ld   = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR1, weight_decay=WEIGHT_DECAY)
    for ep in range(PHASE1_EP):
        train_one_epoch(model, tr_ld, loss_fn, opt1, device, GRAD_CLIP)
        if (ep+1) % 5 == 0:
            print(f"    Phase1 ep {ep+1}/{PHASE1_EP}")

    model.unfreeze_all()
    opt2  = optim.AdamW(model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=LR2_MIN)
    for ep in range(PHASE2_EP):
        train_one_epoch(model, tr_ld, loss_fn, opt2, device, GRAD_CLIP)
        sched.step()
        if (ep+1) % 5 == 0:
            print(f"    Phase2 ep {ep+1}/{PHASE2_EP}")
    return model

def get_score(model, rec, device):
    """Max phishing score across all windows."""
    model.eval()
    hc   = torch.tensor(rec["hand_crafted"], dtype=torch.float32).to(device)
    bert = torch.tensor(rec["bert_embedding"], dtype=torch.float32).to(device)
    wins = rec["windows"]
    best_p = 0.0
    with torch.no_grad():
        for start, end in wins:
            n = end - start
            hc_win = hc[start:end]
            if n < W:
                pad = torch.zeros(W - n, 4, device=device)
                hc_win = torch.cat([hc_win, pad], dim=0)
            else:
                hc_win = hc_win[:W]
            bert_exp = bert.unsqueeze(0).expand(W, -1)
            p, _ = model(hc_win.unsqueeze(0), bert_exp.unsqueeze(0))
            if p.item() > best_p:
                best_p = p.item()
    return best_p

def main():
    print("=" * 70)
    print("Step 23: GT-Targeted Nested CV (v2 - fixed)")
    print(f"  {OUTER_FOLDS}-fold outer | Phase1={PHASE1_EP}ep | Phase2={PHASE2_EP}ep")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GT addresses
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_addrs = {item["account_address"].lower() for item in gt_data}
    print(f"\n[1] GT addresses: {len(gt_addrs)}")

    # Load tier map
    import pandas as pd
    df_val = pd.read_csv(VAL_CSV)
    df_val["phisher_address"] = df_val["phisher_address"].str.lower()
    tier_map = dict(zip(df_val["phisher_address"], df_val["tier"]))

    # Load features
    print("[2] Loading features...")
    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)

    ph_all = [r for r in records if r["label"] == 1]
    nm_all = [r for r in records if r["label"] == 0]

    # Mark each record with is_gt flag
    for r in ph_all:
        r["is_gt"] = (r["address"].lower() in gt_addrs)

    gt_in_pool = sum(1 for r in ph_all if r["is_gt"])
    print(f"  Phishing pool: {len(ph_all)} ({gt_in_pool} GT, {len(ph_all)-gt_in_pool} non-GT)")
    print(f"  Normal pool  : {len(nm_all)}")

    # Sample 1:4 balanced dataset
    rng = np.random.RandomState(SEED)
    n_ph = min(4361, len(ph_all))
    n_nm = min(4 * n_ph, len(nm_all))
    ph_idx = rng.choice(len(ph_all), n_ph, replace=False)
    nm_idx = rng.choice(len(nm_all), n_nm, replace=False)
    ph_sel = [ph_all[i] for i in ph_idx]
    nm_sel = [nm_all[i] for i in nm_idx]

    gt_selected = sum(1 for r in ph_sel if r["is_gt"])
    print(f"  Dataset (1:4): {n_ph} phishing ({gt_selected} GT) + {n_nm} normal")

    # Build combined list with metadata
    all_recs = ph_sel + nm_sel
    all_lbs  = np.array([r["label"] for r in all_recs])
    all_isgt = np.array([1 if r.get("is_gt") else 0 for r in all_recs])

    # 5-fold CV
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)

    # Storage for aggregated results
    fold_results = []

    for fi, (tri, tei) in enumerate(outer.split(np.zeros(len(all_recs)), all_lbs)):
        print(f"\n{'='*60}")
        print(f"[Fold {fi+1}/{OUTER_FOLDS}]")

        tr_recs = [all_recs[i] for i in tri]
        te_recs = [all_recs[i] for i in tei]
        te_lbs  = all_lbs[tei]
        te_isgt = all_isgt[tei]

        # Count
        n_te_ph = int(np.sum(te_lbs == 1))
        n_te_gt = int(np.sum((te_lbs == 1) & (te_isgt == 1)))
        n_te_nm = int(np.sum(te_lbs == 0))
        print(f"  Train={len(tr_recs)} | Test={len(te_recs)} "
              f"(phishing={n_te_ph}, GT={n_te_gt}, normal={n_te_nm})")

        # Train
        print(f"  Training with λ1={LAMBDA1_BEST}...")
        model = train_model(tr_recs, LAMBDA1_BEST, device)

        # Score ALL test records
        print(f"  Scoring {len(te_recs)} accounts...")
        te_scores = []
        for k, account_rec in enumerate(te_recs):
            s = get_score(model, account_rec, device)
            te_scores.append(s)
            if (k+1) % 500 == 0:
                print(f"    Scored {k+1}/{len(te_recs)}")
        te_scores = np.array(te_scores)

        # Overall fold metrics
        fold_m = qmetrics(te_lbs.tolist(), te_scores.tolist())

        # GT-specific metrics
        gt_mask  = (te_lbs == 1) & (te_isgt == 1)
        ngt_mask = (te_lbs == 1) & (te_isgt == 0)
        nm_mask  = (te_lbs == 0)

        gt_s  = te_scores[gt_mask]
        ngt_s = te_scores[ngt_mask]
        nm_s  = te_scores[nm_mask]

        gt_det_05  = float(np.mean(gt_s > 0.5) * 100) if len(gt_s) > 0 else 0.0
        ngt_det_05 = float(np.mean(ngt_s > 0.5) * 100) if len(ngt_s) > 0 else 0.0

        # AUC: GT vs Normal (isolated comparison)
        if len(gt_s) > 0 and len(nm_s) > 0:
            gt_nm_lbs = [1]*len(gt_s) + [0]*len(nm_s)
            gt_nm_sc  = gt_s.tolist() + nm_s.tolist()
            try:
                gt_auc = roc_auc_score(gt_nm_lbs, gt_nm_sc)
            except:
                gt_auc = 0.0
        else:
            gt_auc = 0.0

        print(f"\n  Fold {fi+1} Results:")
        print(f"    Overall AUC      : {fold_m['auc']:.4f}  F1={fold_m['f1']:.4f}  "
              f"Precision={fold_m['precision']:.4f}  Recall={fold_m['recall']:.4f}")
        print(f"    GT AUC vs Normal : {gt_auc:.4f}")
        print(f"    GT detected @0.5 : {gt_det_05:.1f}%  (N={len(gt_s)})")
        print(f"    non-GT detected  : {ngt_det_05:.1f}%  (N={len(ngt_s)})")
        print(f"    GT mean score    : {np.mean(gt_s):.3f}  |  Normal mean: {np.mean(nm_s):.3f}")

        # Per-tier for this fold's GT accounts
        tier_fold = {}
        for i, account_rec in enumerate(te_recs):
            if te_lbs[i] == 1 and te_isgt[i] == 1:
                addr = account_rec["address"].lower()
                tier = tier_map.get(addr, "UNKNOWN")
                if tier not in tier_fold:
                    tier_fold[tier] = []
                tier_fold[tier].append(te_scores[i])

        fold_results.append({
            "fold": fi+1,
            "metrics": fold_m,
            "gt_n": int(len(gt_s)),
            "gt_auc": round(gt_auc, 4),
            "gt_detected_pct": round(gt_det_05, 2),
            "gt_mean_score": round(float(np.mean(gt_s)), 4) if len(gt_s) > 0 else 0,
            "ngt_detected_pct": round(ngt_det_05, 2),
            "tier_results": {t: {"n": len(v), "det_pct": round(float(np.mean(np.array(v)>0.5)*100), 1), "mean_p": round(float(np.mean(v)), 3)} for t, v in tier_fold.items()}
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL AGGREGATED RESULTS — Step 23 GT-Targeted CV")
    print("=" * 70)

    fold_aucs    = [r["metrics"]["auc"] for r in fold_results]
    fold_f1s     = [r["metrics"]["f1"] for r in fold_results]
    fold_precs   = [r["metrics"]["precision"] for r in fold_results]
    fold_recs    = [r["metrics"]["recall"] for r in fold_results]
    fold_gt_auc  = [r["gt_auc"] for r in fold_results]
    fold_gt_det  = [r["gt_detected_pct"] for r in fold_results]
    fold_gt_mean = [r["gt_mean_score"] for r in fold_results]

    print(f"\n  Overall Classification (all phishing vs normal):")
    print(f"    AUC       : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"    F1        : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"    Precision : {np.mean(fold_precs):.4f} ± {np.std(fold_precs):.4f}")
    print(f"    Recall    : {np.mean(fold_recs):.4f} ± {np.std(fold_recs):.4f}")

    print(f"\n  GT Phishing Accounts Specifically:")
    print(f"    GT AUC vs Normal : {np.mean(fold_gt_auc):.4f} ± {np.std(fold_gt_auc):.4f}")
    print(f"    GT Detected @0.5 : {np.mean(fold_gt_det):.1f}% ± {np.std(fold_gt_det):.1f}%")
    print(f"    GT Mean Score    : {np.mean(fold_gt_mean):.4f}")

    # Aggregate per-tier across folds
    print(f"\n  Per-Tier Detection (aggregated across folds):")
    tier_agg = {}
    for fr in fold_results:
        for tier, info in fr["tier_results"].items():
            if tier not in tier_agg:
                tier_agg[tier] = {"n_total": 0, "det_sum": 0.0, "mean_p_sum": 0.0, "fold_count": 0}
            tier_agg[tier]["n_total"] += info["n"]
            tier_agg[tier]["det_sum"] += info["det_pct"] * info["n"]
            tier_agg[tier]["mean_p_sum"] += info["mean_p"] * info["n"]
            tier_agg[tier]["fold_count"] += 1

    tier_out = {}
    for tier in ["VERY_LARGE", "LARGE", "MEDIUM", "SMALL", "MICRO", "NO_DATA", "UNKNOWN"]:
        if tier in tier_agg:
            ta = tier_agg[tier]
            weighted_det  = ta["det_sum"]  / ta["n_total"]
            weighted_mean = ta["mean_p_sum"] / ta["n_total"]
            print(f"    {tier:12} N={ta['n_total']:4d} | Detected={weighted_det:5.1f}% | mean_p={weighted_mean:.3f}")
            tier_out[tier] = {"n": ta["n_total"], "detected_pct": round(weighted_det, 1), "mean_p": round(weighted_mean, 3)}

    print("=" * 70)

    # Save
    out = {
        "overall": {
            "auc_mean": round(float(np.mean(fold_aucs)), 4),
            "auc_std":  round(float(np.std(fold_aucs)), 4),
            "f1_mean":  round(float(np.mean(fold_f1s)), 4),
            "precision_mean": round(float(np.mean(fold_precs)), 4),
            "recall_mean": round(float(np.mean(fold_recs)), 4),
        },
        "gt_metrics": {
            "gt_auc_mean": round(float(np.mean(fold_gt_auc)), 4),
            "gt_auc_std":  round(float(np.std(fold_gt_auc)), 4),
            "gt_detected_pct_mean": round(float(np.mean(fold_gt_det)), 2),
            "gt_detected_pct_std":  round(float(np.std(fold_gt_det)), 2),
            "gt_mean_score": round(float(np.mean(fold_gt_mean)), 4),
        },
        "per_tier": tier_out,
        "per_fold": fold_results,
    }
    out_path = RESULTS_DIR / "step23_gt_targeted_cv.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    print("[OK] Step 23 complete.\n")

if __name__ == "__main__":
    main()
