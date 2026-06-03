import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
import numpy as np
import pandas as pd
import torch
import itertools
from pathlib import Path

from utils import RESULTS_DIR, sliding_windows
from step05_model_architecture import GatedTMILETH

def calculate_iou(pred_set, gt_set):
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    return intersection / union if union > 0 else 0

def inject_anomalies(hc_original, n_inject, position, cluster, seed):
    rng = np.random.RandomState(seed)
    n_tx = hc_original.shape[0]
    
    if position == "early":
        start_range, end_range = 0, int(0.3 * n_tx)
    elif position == "mid":
        start_range, end_range = int(0.3 * n_tx), int(0.7 * n_tx)
    else: # late
        start_range, end_range = int(0.7 * n_tx), n_tx - n_inject
        
    end_range = max(start_range + 1, end_range)
    
    indices = []
    if cluster:
        start_idx = rng.randint(start_range, end_range)
        indices = list(range(start_idx, min(start_idx + n_inject, n_tx)))
    else:
        # scattered
        possible_indices = list(range(start_range, end_range))
        if len(possible_indices) >= n_inject:
            indices = sorted(rng.choice(possible_indices, n_inject, replace=False))
        else:
            indices = possible_indices
            
    # Modify
    hc_mod = hc_original.copy()
    # Anomaly vector representing Tornado Cash cashout
    # High z_amount (5.0), high density (0.8), high novelty (1), high value_ratio (1)
    anomaly_vec = np.array([5.0, 0.8, 1.0, 1.0], dtype=np.float32)
    
    for idx in indices:
        hc_mod[idx] = anomaly_vec
        
    return hc_mod, indices

def main():
    print("="*70)
    print("TMIL-ETH - Step 23: Synthetic Injection Benchmark")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_file = RESULTS_DIR / "step2_features.pkl"
    
    if not features_file.exists():
        print(f"Error: {features_file} not found.")
        return
        
    print("[1] Loading step2_features.pkl...")
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    # Lọc ra N tài khoản Normal (label == 0) có n_tx >= 50
    normal_recs = [r for r in records if r["label"] == 0 and r["n_tx"] >= 50]
    
    # Lấy 500 tài khoản dài nhất làm Host Sequences để chạy cho nhanh
    normal_recs = sorted(normal_recs, key=lambda x: x["n_tx"], reverse=True)[:500]
    print(f"  Selected {len(normal_recs)} Host Sequences (Normal Accounts).")
    
    print("\n[2] Loading trained model (GatedTMILETH)...")
    # Chúng ta sử dụng model vừa được train trong step20 hoặc step12 
    # Nhưng nếu không có checkpoint trực tiếp, ta có thể khởi tạo và mô phỏng 
    # Tuy nhiên, trong thực tế model đã lưu ở results/checkpoints/tmil_eth_final.pt
    model_path = RESULTS_DIR / "checkpoints" / "tmil_eth_final.pt"
    model = GatedTMILETH(4, 64).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded checkpoint from {model_path}")
    else:
        print(f"  WARNING: {model_path} not found. Using untrained weights!")
        
    model.eval()
    
    # Các kịch bản chèn
    n_injects = [1, 3, 5]
    positions = ["early", "mid", "late"]
    clusters = [True, False]
    
    scenarios = []
    for n, p, c in itertools.product(n_injects, positions, clusters):
        if n == 1 and not c: 
            continue # n=1 thì cluster hay scattered như nhau
        scenarios.append({"n_inject": n, "position": p, "cluster": c})
        
    print(f"\n[3] Running Synthetic Benchmark on {len(scenarios)} scenarios...")
    
    results_summary = []
    
    for i, s in enumerate(scenarios):
        hit1 = 0
        hit3 = 0
        hit5 = 0
        ious = []
        
        n_inj = s["n_inject"]
        pos = s["position"]
        clust = s["cluster"]
        
        for r_idx, rec in enumerate(normal_recs):
            # Inject
            seed = 42 + i * 1000 + r_idx
            hc_mod, gt_indices = inject_anomalies(rec["hand_crafted"], n_inj, pos, clust, seed)
            if len(gt_indices) == 0:
                continue
                
            gt_set = set(gt_indices)
            bert = rec["bert_embedding"]
            wins = rec["windows"]
            
            best_attn_scores = None
            best_p = -1
            best_start = 0
            
            for win_idx, (start, end) in enumerate(wins):
                hc_win = hc_mod[start:end]
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
                
            top_k_indices = np.argsort(best_attn_scores)[::-1]
            top1 = best_start + top_k_indices[0]
            top3 = [best_start + x for x in top_k_indices[:3]]
            top5 = [best_start + x for x in top_k_indices[:5]]
            
            if top1 in gt_set: hit1 += 1
            if any(x in gt_set for x in top3): hit3 += 1
            if any(x in gt_set for x in top5): hit5 += 1
            
            pred_set = set(top3)
            ious.append(calculate_iou(pred_set, gt_set))
            
        N = len(normal_recs)
        h1_pct = (hit1 / N) * 100
        h3_pct = (hit3 / N) * 100
        h5_pct = (hit5 / N) * 100
        m_iou = np.mean(ious) * 100
        
        c_str = "Clustered" if clust else "Scattered"
        name = f"N={n_inj}, {pos.capitalize()}, {c_str}"
        
        results_summary.append({
            "Scenario": name,
            "Hit@1": h1_pct,
            "Hit@3": h3_pct,
            "Hit@5": h5_pct,
            "IoU": m_iou
        })
        print(f"  ✓ {name:<30} -> Hit@1: {h1_pct:.2f}% | IoU: {m_iou:.2f}%")
        
    df = pd.DataFrame(results_summary)
    out_path = RESULTS_DIR / "step23_synthetic_benchmark_results.csv"
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*70)
    print("  SYNTHETIC INJECTION BENCHMARK — RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    print(f"Saved detailed results to: {out_path}")

if __name__ == "__main__":
    main()
