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

Outputs: results/figures/step09_nested_cv_results.json
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
FEATURES_FILE = RESULTS_DIR / "step02_features.pkl"

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
    out_path = RESULTS_DIR / "step09_nested_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\n[OK] Step 9 complete.\n")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GT-Targeted Nested CV
# (formerly step09_gt_targeted_cv.py)
# Usage: python -c "from step09_nested_cv import run_gt_targeted_cv; run_gt_targeted_cv()"
# ══════════════════════════════════════════════════════════════════════════════

def run_gt_targeted_cv():
    """
    Nested 5-Fold CV variant that explicitly tracks Ground Truth (GT) accounts.
    Reports per-tier detection rates in addition to overall classification metrics.
    Saves: results/figures/step09_gt_targeted_cv.json
    """
    import pickle as _pk23, pandas as _pd23
    from sklearn.model_selection import StratifiedKFold as _SKF23
    from sklearn.metrics import roc_auc_score as _auc23, f1_score as _f1_23
    from sklearn.metrics import precision_score as _prec23, recall_score as _rec23
    from sklearn.metrics import roc_curve as _roc_curve23
    import torch as _torch23, torch.optim as _optim23
    from torch.utils.data import DataLoader as _DL23
    from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
    from step07_training import AccountWindowDataset, collate_fn, train_one_epoch

    GT_FILE23  = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
    VAL_CSV23  = Path(__file__).parent / "validation" / "full_automated_validation.csv"

    print("=" * 70)
    print("Step 9b: GT-Targeted Nested CV")
    print("=" * 70)
    device = _torch23.device("cuda" if _torch23.cuda.is_available() else "cpu")
    if not GT_FILE23.exists():
        print("[SKIP] GT file not found."); return

    with open(GT_FILE23, "r", encoding="utf-8") as f:
        import json as _j23; gt_data = _j23.load(f)
    gt_addrs23 = {item["account_address"].lower() for item in gt_data}

    with open(RESULTS_DIR / "step02_features.pkl", "rb") as f:
        records23 = _pk23.load(f)

    ph_all = [r for r in records23 if r["label"] == 1]
    nm_all = [r for r in records23 if r["label"] == 0]
    for r in ph_all: r["is_gt"] = (r["address"].lower() in gt_addrs23)

    rng23 = _np.random.RandomState(42)
    n_ph = min(4361, len(ph_all))
    n_nm = min(4 * n_ph, len(nm_all))
    ph_sel = [ph_all[i] for i in rng23.choice(len(ph_all), n_ph, replace=False)]
    nm_sel = [nm_all[i] for i in rng23.choice(len(nm_all), n_nm, replace=False)]
    all_r23 = ph_sel + nm_sel
    all_l23 = _np.array([r["label"] for r in all_r23])
    all_isgt = _np.array([1 if r.get("is_gt") else 0 for r in all_r23])
    print(f"Dataset: {n_ph} phishing ({sum(1 for r in ph_sel if r['is_gt'])} GT) + {n_nm} normal")

    outer23 = _SKF23(5, shuffle=True, random_state=42)
    fold_results23 = []

    for fi, (tri, tei) in enumerate(outer23.split(_np.zeros(len(all_r23)), all_l23)):
        print(f"\n[Fold {fi+1}/5]")
        tr_recs23 = [all_r23[i] for i in tri]
        te_recs23 = [all_r23[i] for i in tei]
        te_lbs23  = all_l23[tei]
        te_isgt23 = all_isgt[tei]

        model23 = GatedTMILETH(4, 64).to(device)
        loss_fn23 = GatedCompoundLoss(lambda1=0.3)
        ds23 = AccountWindowDataset(tr_recs23, W=200)
        ld23 = _DL23(ds23, 64, shuffle=True, collate_fn=collate_fn, num_workers=0)
        model23.freeze_bert()
        o1 = _optim23.AdamW(filter(lambda p: p.requires_grad, model23.parameters()), lr=1e-3)
        for _ in range(20): train_one_epoch(model23, ld23, loss_fn23, o1, device, 1.0)
        model23.unfreeze_all()
        o2 = _optim23.AdamW(model23.parameters(), lr=5e-5)
        sch = _torch23.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=30, eta_min=1e-6)
        for _ in range(30):
            train_one_epoch(model23, ld23, loss_fn23, o2, device, 1.0); sch.step()

        model23.eval()
        scores23 = []
        preds23 = []
        from utils import sidak_threshold as _sidak23
        with _torch23.no_grad():
            for rec in te_recs23:
                hc = _torch23.tensor(rec["hand_crafted"], dtype=_torch23.float32).to(device)
                bert = _torch23.tensor(rec["bert_embedding"], dtype=_torch23.float32).to(device)
                best = 0.0
                for st, en in rec["windows"]:
                    n = en - st; hw = hc[st:en]
                    if n < 200: hw = _torch23.cat([hw, _torch23.zeros(200-n, 4, device=device)])
                    else: hw = hw[:200]
                    be = bert.unsqueeze(0).expand(200, -1)
                    p, _ = model23(hw.unsqueeze(0), be.unsqueeze(0))
                    if p.item() > best: best = p.item()
                scores23.append(best)
                # Sidak correction for binary prediction
                tau_eff = _sidak23(0.5, len(rec["windows"]))
                preds23.append(1 if best >= tau_eff else 0)
        scores23 = _np.array(scores23)
        b23 = _np.array(preds23)

        try:
            fold_auc = float(_auc23(te_lbs23, scores23))
        except: fold_auc = 0.0
        gt_mask23 = (te_lbs23 == 1) & (te_isgt23 == 1)
        gt_det = float(_np.mean(scores23[gt_mask23] > 0.5) * 100) if gt_mask23.sum() > 0 else 0.0
        print(f"  AUC={fold_auc:.4f}  F1={_f1_23(te_lbs23, b23):.4f}  "
              f"GT_detected={gt_det:.1f}% (N={gt_mask23.sum()})")
        fold_results23.append({"fold": fi+1, "auc": round(fold_auc, 4),
                               "f1": round(float(_f1_23(te_lbs23, b23)), 4),
                               "gt_detected_pct": round(gt_det, 2)})

    mean_auc23 = float(_np.mean([r["auc"] for r in fold_results23]))
    mean_f1_23 = float(_np.mean([r["f1"] for r in fold_results23]))
    print(f"\n  FINAL: AUC={mean_auc23:.4f} ± {_np.std([r['auc'] for r in fold_results23]):.4f}  "
          f"F1={mean_f1_23:.4f}")
    out23 = RESULTS_DIR / "step09_gt_targeted_cv.json"
    with open(out23, "w", encoding="utf-8") as f:
        import json as _j23b; _j23b.dump({"overall": {"auc_mean": round(mean_auc23, 4),
                                                       "f1_mean": round(mean_f1_23, 4)},
                                          "per_fold": fold_results23}, f, indent=2)
    print(f"  Saved: {out23}")
    print("[OK] GT-Targeted CV complete.\n")
