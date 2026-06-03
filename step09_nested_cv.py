"""
TMIL-ETH — Step 9: Nested Stratified CV (Full Data — GPU Version)
==================================================================
Protocol (paper §4.2):
  - Outer: 5-fold stratified CV
  - Inner: 3-fold stratified CV (lambda selection)
  - Lambda grid: lambda1 in {0.1, 0.3, 0.5} x lambda2 in {0.1, 0.2, 0.3} = 9 combos
  - FPR@95%TPR <= 0.08 hard constraint
  - Imbalance ratios: 1:4, 1:10, 1:20
  - Full dataset: 4,361 phishing + up to 87,220 normal accounts
  - GPU recommended: RTX A6000 or equivalent

Outputs: results/step9_nested_cv_results.json
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json, pickle, random, numpy as np, torch, torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_score, recall_score
from torch.utils.data import DataLoader

from utils import RESULTS_DIR
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
from step07_training import (AccountWindowDataset, collate_fn,
                                       train_one_epoch, evaluate_epoch)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"

SEED = 42
OUTER_FOLDS  = 5
INNER_FOLDS  = 3
BATCH_SIZE   = 128     # GPU: use 128+; CPU: reduce to 32–64
PHASE1_EP    = 20      # paper protocol: 20 epochs warmup
PHASE2_EP    = 30      # paper protocol: 30 epochs fine-tune
LR1, LR2, LR2_MIN = 1e-3, 5e-5, 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
W = 200

# Full dataset: 4361 phishing accounts (set lower only for quick debug)
N_PHISH_SAMPLE = 4361  # full phishing set; paper uses all available
LAMBDA1_GRID = [0.1, 0.3, 0.5]
LAMBDA2_GRID = [0.1, 0.2, 0.3]
FPR_CONSTRAINT = 0.08
IMBALANCE_RATIOS = ["1:4", "1:10", "1:20"]


def fpr_at_95tpr(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 1.0
    fpr_a, tpr_a, _ = roc_curve(y_true, y_score)
    return float(fpr_a[np.argmin(np.abs(tpr_a - 0.95))])


def qmetrics(y_true, y_score, tau=0.5):
    try: auc = float(roc_auc_score(y_true, y_score))
    except: auc = 0.0
    b = (y_score >= tau).astype(int)
    return {
        "auc": auc,
        "f1":  float(f1_score(y_true, b, zero_division=0)),
        "precision": float(precision_score(y_true, b, zero_division=0)),
        "recall":    float(recall_score(y_true, b, zero_division=0)),
        "fpr_at_95tpr": fpr_at_95tpr(y_true, y_score),
    }


def train_eval(tr, va, l1, l2, device):
    loss_fn = GatedCompoundLoss(lambda1=l1)
    model   = GatedTMILETH(4, 64).to(device)
    tr_ds = AccountWindowDataset(tr, W=W)
    va_ds = AccountWindowDataset(va, W=W)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR1, weight_decay=WEIGHT_DECAY)
    for _ in range(PHASE1_EP):
        train_one_epoch(model, tr_ld, loss_fn, opt1, device, GRAD_CLIP)

    model.unfreeze_all()
    opt2  = optim.AdamW(model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=LR2_MIN)
    for _ in range(PHASE2_EP):
        train_one_epoch(model, tr_ld, loss_fn, opt2, device, GRAD_CLIP)
        sched.step()

    _, p, l = evaluate_epoch(model, va_ld, loss_fn, device)
    return qmetrics(l, p)


def run_cv(ph_all, nm_all, ratio, device):
    rng = np.random.RandomState(SEED)
    mults = {"1:4": 4, "1:10": 10, "1:20": 20}
    n_ph = min(N_PHISH_SAMPLE, len(ph_all))
    n_nm = min(mults[ratio] * n_ph, len(nm_all))
    ph  = [ph_all[i] for i in rng.choice(len(ph_all), n_ph, replace=False)]
    nm  = [nm_all[i] for i in rng.choice(len(nm_all), n_nm, replace=False)]
    cv  = ph + nm
    lbs = np.array([r["label"] for r in cv])
    print(f"\n  Ratio {ratio}: n={len(cv):,} (phish={n_ph:,}, normal={n_nm:,})")

    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fi, (tri, tei) in enumerate(outer.split(np.zeros(len(cv)), lbs)):
        tr_recs = [cv[i] for i in tri]
        te_recs = [cv[i] for i in tei]
        tr_lbs  = np.array([r["label"] for r in tr_recs])
        inner   = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)

        best_l, best_auc = (0.3, 0.2), -1.0
        for l1 in LAMBDA1_GRID:
            for l2 in LAMBDA2_GRID:
                in_aucs, in_fprs = [], []
                for ini, vali in inner.split(np.zeros(len(tr_recs)), tr_lbs):
                    try:
                        m = train_eval([tr_recs[i] for i in ini],
                                       [tr_recs[i] for i in vali], l1, l2, device)
                        in_aucs.append(m["auc"]); in_fprs.append(m["fpr_at_95tpr"])
                    except:
                        in_aucs.append(0.0); in_fprs.append(1.0)
                if np.mean(in_fprs) <= FPR_CONSTRAINT and np.mean(in_aucs) > best_auc:
                    best_auc = np.mean(in_aucs); best_l = (l1, l2)

        try:
            tm = train_eval(tr_recs, te_recs, best_l[0], best_l[1], device)
        except Exception as e:
            print(f"    Outer test failed: {e}")
            tm = {"auc":0.0,"f1":0.0,"fpr_at_95tpr":1.0,"precision":0.0,"recall":0.0}

        print(f"    Fold {fi+1}/{OUTER_FOLDS}: AUC={tm['auc']:.4f}  F1={tm['f1']:.4f}  "
              f"FPR@95%={tm['fpr_at_95tpr']:.4f}  lambda=({best_l[0]},{best_l[1]})")
        fold_results.append({"fold": fi+1, "best_lambda": list(best_l), "metrics": tm})

    aucs = [r["metrics"]["auc"]          for r in fold_results]
    f1s  = [r["metrics"]["f1"]           for r in fold_results]
    fprs = [r["metrics"]["fpr_at_95tpr"] for r in fold_results]
    ps   = [r["metrics"]["precision"]    for r in fold_results]
    rs   = [r["metrics"]["recall"]       for r in fold_results]
    agg  = {
        "auc":          {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "f1":           {"mean": float(np.mean(f1s)),  "std": float(np.std(f1s))},
        "fpr_at_95tpr": {"mean": float(np.mean(fprs)), "std": float(np.std(fprs))},
        "precision":    {"mean": float(np.mean(ps)),   "std": float(np.std(ps))},
        "recall":       {"mean": float(np.mean(rs)),   "std": float(np.std(rs))},
    }
    meets = agg["fpr_at_95tpr"]["mean"] <= FPR_CONSTRAINT
    print(f"  --- {ratio}: AUC={agg['auc']['mean']:.4f}+/-{agg['auc']['std']:.4f}  "
          f"F1={agg['f1']['mean']:.4f}  FPR@95%={agg['fpr_at_95tpr']['mean']:.4f}  "
          f"Constraint={'PASS' if meets else 'FAIL'}")
    return {"ratio": ratio, "aggregate": agg, "per_fold": fold_results,
            "meets_fpr_constraint": meets, "n_phishing": n_ph, "n_normal": n_nm}


def main():
    print("="*60)
    print("TMIL-ETH - Step 9: Nested CV (CPU-optimised)")
    print(f"  Outer={OUTER_FOLDS}-fold  Inner={INNER_FOLDS}-fold")
    print(f"  Lambda grid: {len(LAMBDA1_GRID)*len(LAMBDA2_GRID)} combos")
    print(f"  Phishing subsample: {N_PHISH_SAMPLE} (adjust for GPU)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records):,} records")

    ph_all = [r for r in records if r["label"] == 1]
    nm_all = [r for r in records if r["label"] == 0]

    all_results = {}
    for ratio in IMBALANCE_RATIOS:
        all_results[ratio] = run_cv(ph_all, nm_all, ratio, device)

    print("\n" + "="*60)
    print("FINAL RESULTS — TMIL-ETH Nested CV")
    print("="*60)
    print(f"  {'Ratio':>6} | {'AUC':>14} | {'F1':>10} | {'FPR@95%':>10} | {'OK?'}")
    print(f"  {'-'*6} | {'-'*14} | {'-'*10} | {'-'*10} | {'-'*5}")
    for ratio, res in all_results.items():
        a = res["aggregate"]
        ok = "PASS" if res["meets_fpr_constraint"] else "FAIL"
        print(f"  {ratio:>6} | "
              f"{a['auc']['mean']:.4f}+/-{a['auc']['std']:.4f} | "
              f"{a['f1']['mean']:.4f}+/-{a['f1']['std']:.4f} | "
              f"{a['fpr_at_95tpr']['mean']:.4f}+/-{a['fpr_at_95tpr']['std']:.4f} | {ok}")

    out = {
        "config": {
            "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
            "lambda1_grid": LAMBDA1_GRID, "lambda2_grid": LAMBDA2_GRID,
            "fpr_constraint": FPR_CONSTRAINT,
            "phase1_ep": PHASE1_EP, "phase2_ep": PHASE2_EP,
            "batch_size": BATCH_SIZE,
            "n_phish_sample": N_PHISH_SAMPLE,
            "mode": "CPU-optimised (reduced subsample+epochs)",
        },
        "results_by_ratio": all_results,
    }
    out_path = RESULTS_DIR / "step9_nested_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\n[OK] Step 9 complete.\n")


if __name__ == "__main__":
    main()
