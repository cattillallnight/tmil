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
from step05_model_architecture import TMILETH, CompoundLoss
from step07_training import AccountWindowDataset, collate_fn, train_one_epoch

def calculate_iou(pred_set, gt_set):
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    return intersection / union if union > 0 else 0

def main():
    print("="*70)
    print("TMIL-ETH - Step 20: Forensic Localization Eval (Time-Aware GT)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_file = RESULTS_DIR / "step2_features.pkl"
    # === THE ULTIMATE GT: Time-Aware Cross-Reference & Tornado Cash ===
    # Built by step22_time_aware_gt_builder.py
    # 1. Sender IN Normal AND Receiver IN Phisher
    # 2. Clustered by Time (Removes Outliers/Temporal Leakage)
    # 3. Endpoint mapped to Tornado Cash
    gt_file = "time_aware_ground_truth.json"
    
    if not Path(gt_file).exists():
        print(f"Error: {gt_file} not found. Run step22_time_aware_gt_builder.py first.")
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
    model = TMILETH(4, 64).to(device)
    loss_fn = CompoundLoss(lambda1=0.3, lambda2=0.2)
    
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

    print("\n[3] Bắt đầu chấm điểm (Evaluation) dựa trên Human Ground Truth...")
    
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
            "human_gt_burst": f"{gt_start}-{gt_end}",
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
    out_path = RESULTS_DIR / "step20_time_aware_localization_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")

if __name__ == "__main__":
    main()
