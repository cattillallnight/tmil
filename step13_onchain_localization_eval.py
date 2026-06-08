import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import RESULTS_DIR
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
from step07_training import AccountWindowDataset, collate_fn, train_one_epoch

def calculate_iou(pred_set, gt_set):
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    return intersection / union if union > 0 else 0

def main():
    print("="*70)
    print("TMIL-ETH - Step 13: Forensic Localization Eval (Time-Aware GT)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_file = RESULTS_DIR / "step2_features.pkl"
    # === THE ULTIMATE GT: Time-Aware Cross-Reference & Tornado Cash ===
    # Built by step12_time_aware_gt_builder.py
    # 1. Sender IN Normal AND Receiver IN Phisher
    # 2. Clustered by Time (Removes Outliers/Temporal Leakage)
    # 3. Endpoint mapped to Tornado Cash
    TMIL_DIR = Path(__file__).parent
    gt_file = TMIL_DIR / "ground_truth" / "time_aware_ground_truth.json"
    
    if not Path(gt_file).exists():
        print(f"Error: {gt_file} not found. Run step12_time_aware_gt_builder.py first.")
        return
        
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    print(f"\n[1] Loading dataset and Time-Aware Ground Truth ({len(gt_data)} accounts)...")
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    # Tách tập Test (Các ví do người dán nhãn) ra khỏi tập Train
    eval_addrs = {item["account_address"].lower() for item in gt_data}
    
    test_recs = []
    train_pool_phish = []
    train_pool_norm = []
    
    for r in records:
        if r["address"].lower() in eval_addrs:
            test_recs.append(r)
        else:
            if r["label"] == 1:
                train_pool_phish.append(r)
            else:
                train_pool_norm.append(r)
                
    # Lấy 1 lượng nhỏ (ví dụ 100 phish, 400 normal) để train thật nhanh mô hình 
    # (Vì mục đích chỉ là lấy Attention Score, không cần mô hình hoàn hảo 100%)
    rng = np.random.RandomState(42)
    n_train_phish = min(100, len(train_pool_phish))
    n_train_norm = min(400, len(train_pool_norm))
    
    train_recs = rng.choice(train_pool_phish, n_train_phish, replace=False).tolist() + \
                 rng.choice(train_pool_norm, n_train_norm, replace=False).tolist()
                 
    print(f"  Tập Train cách ly (Isolated Train Set): {len(train_recs)} accounts.")
    print(f"  Tập Test ẩn (Hidden Eval Set)         : {len(test_recs)} accounts.")
    
    print("\n[2] Training model (10 epochs, GPU-accelerated) to extract Attention scores...")
    model = GatedTMILETH(4, 64).to(device)
    loss_fn = GatedCompoundLoss(lambda1=0.3)
    
    ds = AccountWindowDataset(train_recs, W=200)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Phase 1: Freeze BERT, only train MIL head (5 epochs on GPU)
    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for ep in range(5):
        train_one_epoch(model, loader, loss_fn, opt1, device, 1.0)
        print(f"  Phase 1 Epoch {ep+1}/5 done.")
        
    # Phase 2: Unfreeze all, fine-tune end-to-end (5 epochs on GPU)
    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=1e-4)
    for ep in range(5):
        train_one_epoch(model, loader, loss_fn, opt2, device, 1.0)
        print(f"  Phase 2 Epoch {ep+1}/5 done.")

    print("\n[3] Bắt đầu chấm điểm (Evaluation) dựa trên Time-Aware Algorithmic Ground Truth...")
    
    hit_at_1_count = 0
    hit_at_3_count = 0
    hit_at_5_count = 0
    ious = []
    results_list = []
    
    # Tạo dict để truy xuất nhanh Ground Truth
    gt_dict = {item["account_address"].lower(): item for item in gt_data}
    # Duyệt qua từng tài khoản Test (Chỉ đánh giá 1 lần cho 1 địa chỉ để tránh lặp)
    seen_addrs = set()
    for rec in test_recs:
        addr = rec["address"].lower()
        if addr not in eval_addrs:
            continue
        if addr in seen_addrs:
            continue
        seen_addrs.add(addr)
            
        # === ULTIMATE GT: Key is "time_aware_gt_bursts" ===
        gt_bursts = gt_dict[addr].get("time_aware_gt_bursts", [])
        if not gt_bursts:
            continue
        gt = gt_bursts[0]
        gt_start = gt["start_tx_idx"]
        gt_end   = gt["end_tx_idx"]
        gt_set   = set(range(gt_start, gt_end + 1))
        
        # Exact victim indices from the dense temporal cluster
        victim_indices = set(gt_dict[addr].get("victim_tx_indices", []))
        
        hc = rec["hand_crafted"]  
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        
        best_attn_scores = None
        best_p = -1
        best_start = 0
        
        for win_idx, (start, end) in enumerate(wins):
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < 200:
                pad = np.zeros((200 - n, 4), dtype=np.float32)
                hc_win_pad = np.vstack([hc_win, pad])
            else:
                hc_win_pad = hc_win[:200]
                
            hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(device)
            bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 200, -1).to(device)
            
            with torch.no_grad():
                p, attn = model(hc_t, bert_t)
                
            if p.item() > best_p:
                best_p = p.item()
                best_attn_scores = attn.squeeze(0).cpu().numpy()[:n]
                best_start = start
                
        if best_attn_scores is None:
            continue
            
        # Top-k metrics
        top_k_indices = np.argsort(best_attn_scores)[::-1]
        
        hit_1 = 1 if (best_start + top_k_indices[0]) in gt_set else 0
        hit_3 = 1 if any((best_start + i) in gt_set for i in top_k_indices[:3]) else 0
        hit_5 = 1 if any((best_start + i) in gt_set for i in top_k_indices[:5]) else 0
        
        hit_at_1_count += hit_1
        hit_at_3_count += hit_3
        hit_at_5_count += hit_5
        
        # Lấy Top-3 Attention để tính IoU
        top3_local = top_k_indices[:3]
        pred_set = set([best_start + i for i in top3_local])
        
        iou = calculate_iou(pred_set, gt_set)
        ious.append(iou)
        
        results_list.append({
            "account": addr,
            "auto_gt_burst": f"{gt_start}-{gt_end}",
            "ai_max_attn_idx": best_start + top_k_indices[0],
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "iou": round(iou, 3)
        })

    if len(results_list) == 0:
        print("Không có kết quả nào để đánh giá.")
        return
        
    hit_1_rate = (hit_at_1_count / len(results_list)) * 100
    hit_3_rate = (hit_at_3_count / len(results_list)) * 100
    hit_5_rate = (hit_at_5_count / len(results_list)) * 100
    mean_iou = np.mean(ious) * 100
    
    print("\n" + "="*70)
    print("   FORENSIC LOCALIZATION EVALUATION — TIME-AWARE GROUND TRUTH")
    print("="*70)
    print(f"  Accounts evaluated (Time-Aware clustered)  : {len(results_list)}")
    print(f"  Pointing Game  Hit@1                           : {hit_1_rate:.2f}%")
    print(f"  Pointing Game  Hit@3                           : {hit_3_rate:.2f}%")
    print(f"  Pointing Game  Hit@5                           : {hit_5_rate:.2f}%")
    print(f"  Temporal Overlap  Mean IoU                     : {mean_iou:.2f}%")
    print("="*70)
    print("\n* Ground Truth: Cross-Reference + Temporal Clustering + Tornado Cash")
    print("* ZERO assumptions: No temporal leakage, endpoints verified via Smart Contracts")
    
    df = pd.DataFrame(results_list)
    out_path = RESULTS_DIR / "step13_time_aware_localization_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")

if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Full Automated Validation (2,483 GT accounts, no API)
# (formerly step21_full_automated_validation.py)
# To run: call run_full_automated_validation() below
# ══════════════════════════════════════════════════════════════════════════════

import math as _math_s21
from datetime import datetime as _datetime_s21, timezone as _timezone_s21

_DATA_DIR_S21    = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
_GT_FILE_S21     = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
_CEX_CSV_S21     = Path(__file__).parent / "results" / "cex_address_sources.csv"
_VAL_DIR_S21     = Path(__file__).parent / "validation"
_PRIMARY_THRESH  = 0.5   # ETH
_PROBABLE_THRESH = 5.0   # ETH


def _wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * _math_s21.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0.0, center - margin), min(1.0, center + margin)


def run_full_automated_validation():
    """Full automated validation on all 2,483 GT accounts (no API required)."""
    import pandas as _pd_s21
    print("=" * 70)
    print("Step 13b: Full Automated Validation (2,483 accounts, no API)")
    print("=" * 70)
    _VAL_DIR_S21.mkdir(exist_ok=True)
    if not _CEX_CSV_S21.exists() or not _GT_FILE_S21.exists():
        print("[SKIP] CEX CSV or GT file not found.")
        return
    df_cex = _pd_s21.read_csv(_CEX_CSV_S21)
    cex_set = set(df_cex["address"].str.lower().str.strip().tolist())
    cex_label = {row["address"].lower().strip(): row["label"] for _, row in df_cex.iterrows()}
    with open(_GT_FILE_S21, "r", encoding="utf-8") as f:
        import json as _j21; gt_data = _j21.load(f)
    if not _DATA_DIR_S21.exists():
        print(f"[SKIP] Data directory not found: {_DATA_DIR_S21}")
        return
    df_out = _pd_s21.read_csv(_DATA_DIR_S21 / "phisher_transaction_out.csv",
                              header=None, dtype=str, low_memory=False)
    df_out.columns = list(range(len(df_out.columns)))
    df_out[5]  = df_out[5].str.lower().str.strip().fillna("")
    df_out[6]  = df_out[6].str.lower().str.strip().fillna("")
    df_out[7]  = _pd_s21.to_numeric(df_out[7], errors="coerce").fillna(0)
    df_out[11] = _pd_s21.to_numeric(df_out[11], errors="coerce").fillna(0)
    out_grouped = df_out.groupby(5)
    records_out = []
    for acc in gt_data:
        addr = acc["account_address"].lower().strip()
        if addr not in out_grouped.groups:
            records_out.append({"phisher_address": addr, "cashout_value_eth": 0.0,
                                "tier": "NO_DATA", "stratum": "NO_DATA",
                                "confirmation_status": "NO_CASHOUT", "cex_label": ""})
            continue
        phisher_out = out_grouped.get_group(addr).sort_values(11).reset_index(drop=True)
        if len(phisher_out) == 0:
            records_out.append({"phisher_address": addr, "cashout_value_eth": 0.0,
                                "tier": "NO_DATA", "stratum": "NO_DATA",
                                "confirmation_status": "NO_CASHOUT", "cex_label": ""})
            continue
        max_row = phisher_out.loc[phisher_out[7].idxmax()]
        val_eth = float(max_row[7]) / 1e18
        to_addr = str(max_row[6]).lower().strip()
        tier = ("VERY_LARGE" if val_eth > 50 else "LARGE" if val_eth >= 5
                else "MEDIUM" if val_eth >= 0.5 else "SMALL" if val_eth >= 0.05 else "MICRO")
        stratum = "PRIMARY" if val_eth >= _PRIMARY_THRESH else "SMALL_MICRO"
        if not to_addr or to_addr in cex_set:
            status = "CONFIRMED_CEX" if to_addr in cex_set else "NO_CASHOUT"
            clabel = cex_label.get(to_addr, "Known CEX/Mixer") if to_addr in cex_set else ""
        elif stratum == "SMALL_MICRO":
            status, clabel = "SMALL_MICRO_EXCLUDED", ""
        elif val_eth >= _PROBABLE_THRESH:
            status, clabel = "PROBABLE", ""
        else:
            status, clabel = "UNCERTAIN", ""
        records_out.append({"phisher_address": addr, "cashout_value_eth": round(val_eth, 6),
                            "cashout_to_address": to_addr, "tier": tier, "stratum": stratum,
                            "confirmation_status": status, "cex_label": clabel})
    df_val = _pd_s21.DataFrame(records_out)
    out_csv = _VAL_DIR_S21 / "full_automated_validation.csv"
    df_val.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_p = df_val[df_val["stratum"] == "PRIMARY"]
    n_p = len(df_p)
    n_conf = (df_p["confirmation_status"] == "CONFIRMED_CEX").sum()
    n_prob = (df_p["confirmation_status"] == "PROBABLE").sum()
    ci_lo, ci_hi = _wilson_ci(int(n_conf), n_p)
    print(f"  PRIMARY accounts: {n_p} | CONFIRMED_CEX: {n_conf} ({n_conf/n_p*100:.1f}%)")
    print(f"  95% Wilson CI (strict): [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    print(f"  CONFIRMED+PROBABLE: {n_conf+n_prob} ({(n_conf+n_prob)/n_p*100:.1f}%)")
    print(f"  Saved: {out_csv}")
    print("[OK] Full Automated Validation complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: GT Account Classification Accuracy
# (formerly step22_gt_classification_accuracy.py)
# To run: call run_gt_classification_accuracy() below
# ══════════════════════════════════════════════════════════════════════════════

def run_gt_classification_accuracy():
    """
    Evaluates TMIL-ETH on 2,483 Ground Truth phishing accounts.
    Trains on non-GT phishers, evaluates on GT accounts held out.
    """
    import pickle as _pk22
    import numpy as _np22
    import pandas as _pd22
    from sklearn.metrics import roc_auc_score as _roc22, f1_score as _f1_22
    from sklearn.metrics import precision_score as _prec22, recall_score as _rec22
    from step05_model_architecture import GatedTMILETH, GatedCompoundLoss
    from step07_training import AccountWindowDataset, collate_fn, train_one_epoch
    from torch.utils.data import DataLoader as _DL22
    import torch as _torch22, torch.optim as _optim22

    print("=" * 70)
    print("Step 13c: GT Account Classification Accuracy")
    print("=" * 70)
    device = _torch22.device("cuda" if _torch22.cuda.is_available() else "cpu")
    gt_file = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
    val_csv = Path(__file__).parent / "validation" / "full_automated_validation.csv"
    if not gt_file.exists() or not val_csv.exists():
        print("[SKIP] GT or validation CSV not found.")
        return
    with open(gt_file, "r", encoding="utf-8") as f:
        import json as _j22; gt_addrs = {it["account_address"].lower() for it in _j22.load(f)}
    df_val22 = _pd22.read_csv(val_csv)
    df_val22["phisher_address"] = df_val22["phisher_address"].str.lower()
    tier_map22 = dict(zip(df_val22["phisher_address"], df_val22["tier"]))
    with open(RESULTS_DIR / "step2_features.pkl", "rb") as f:
        all_recs = _pk22.load(f)
    gt_recs   = [r for r in all_recs if r["address"].lower() in gt_addrs and r["label"] == 1]
    train_ph  = [r for r in all_recs if r["address"].lower() not in gt_addrs and r["label"] == 1]
    norm_recs = [r for r in all_recs if r["label"] == 0]
    rng22 = _np22.random.RandomState(42)
    t_ph  = rng22.choice(train_ph, min(2000, len(train_ph)), replace=False).tolist()
    t_nm  = rng22.choice(norm_recs, min(8000, len(norm_recs)), replace=False).tolist()
    train_recs22 = t_ph + t_nm
    model22 = GatedTMILETH(4, 64).to(device)
    loss_fn22 = GatedCompoundLoss(lambda1=0.3)
    ds22 = AccountWindowDataset(train_recs22, W=200)
    loader22 = _DL22(ds22, batch_size=64, shuffle=True, collate_fn=collate_fn)
    model22.freeze_bert()
    opt1_22 = _optim22.AdamW(filter(lambda p: p.requires_grad, model22.parameters()), lr=1e-3)
    for ep in range(10): train_one_epoch(model22, loader22, loss_fn22, opt1_22, device, 1.0)
    model22.unfreeze_all()
    opt2_22 = _optim22.AdamW(model22.parameters(), lr=5e-5)
    sched22 = _torch22.optim.lr_scheduler.CosineAnnealingLR(opt2_22, T_max=15, eta_min=1e-6)
    for ep in range(15):
        train_one_epoch(model22, loader22, loss_fn22, opt2_22, device, 1.0); sched22.step()

    def _pred22(rec):
        hc = _torch22.tensor(rec["hand_crafted"], dtype=_torch22.float32).to(device)
        bert = _torch22.tensor(rec["bert_embedding"], dtype=_torch22.float32).to(device)
        best = -1
        for start, end in rec["windows"]:
            n = end - start
            hw = hc[start:end]
            if n < 200: hw = _torch22.cat([hw, _torch22.zeros(200-n, 4, device=device)])
            else: hw = hw[:200]
            be = bert.unsqueeze(0).expand(200, -1)
            with _torch22.no_grad():
                p, _ = model22(hw.unsqueeze(0), be.unsqueeze(0))
            if p.item() > best: best = p.item()
        return best

    model22.eval()
    gt_scores22   = [_pred22(r) for r in gt_recs]
    test_normals22 = rng22.choice(norm_recs, min(len(gt_recs), len(norm_recs)), replace=False).tolist()
    norm_scores22  = [_pred22(r) for r in test_normals22]
    y_true22 = [1]*len(gt_scores22) + [0]*len(norm_scores22)
    y_score22 = gt_scores22 + norm_scores22
    y_pred22  = [1 if s > 0.5 else 0 for s in y_score22]
    auc22 = _roc22(y_true22, y_score22)
    print(f"  GT Detected @0.5: {_np22.mean(_np22.array(gt_scores22)>0.5)*100:.1f}%  "
          f"AUC={auc22:.4f}  F1={_f1_22(y_true22, y_pred22):.4f}  "
          f"Prec={_prec22(y_true22, y_pred22, zero_division=0):.4f}")
    import json as _j22b
    out22 = RESULTS_DIR / "step22_gt_classification_accuracy.json"
    with open(out22, "w") as f:
        _j22b.dump({"gt_n": len(gt_scores22), "auc": round(auc22, 4),
                    "gt_detected_pct": round(float(_np22.mean(_np22.array(gt_scores22)>0.5)*100), 2)}, f, indent=2)
    print(f"  Saved: {out22}")
    print("[OK] GT Classification Accuracy complete.\n")
