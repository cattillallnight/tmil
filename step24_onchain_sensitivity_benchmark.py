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

def evaluate_gt_file(gt_file, model, test_recs, eval_addrs, device):
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    gt_dict = {item["account_address"].lower(): item for item in gt_data}
    
    hit1, hit3, hit5 = 0, 0, 0
    ious = []
    total = 0
    
    seen_addrs = set()
    for rec in test_recs:
        addr = rec["address"].lower()
        if addr not in eval_addrs or addr in seen_addrs or addr not in gt_dict:
            continue
        seen_addrs.add(addr)
            
        gt_bursts = gt_dict[addr].get("time_aware_gt_bursts", [])
        if not gt_bursts: continue
        
        gt = gt_bursts[0]
        gt_start = gt["start_tx_idx"]
        gt_end   = gt["end_tx_idx"]
        gt_set   = set(range(gt_start, gt_end + 1))
        if len(gt_set) == 0: continue
        
        total += 1
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
            
        top_k_indices = np.argsort(best_attn_scores)[::-1]
        top1 = best_start + top_k_indices[0]
        top3 = [best_start + x for x in top_k_indices[:3]]
        top5 = [best_start + x for x in top_k_indices[:5]]
        
        if top1 in gt_set: hit1 += 1
        if any(x in gt_set for x in top3): hit3 += 1
        if any(x in gt_set for x in top5): hit5 += 1
        
        pred_set = set(top3)
        ious.append(calculate_iou(pred_set, gt_set))

    if total == 0: return 0, 0, 0, 0
    return (hit1/total)*100, (hit3/total)*100, (hit5/total)*100, np.mean(ious)*100

def main():
    print("="*70)
    print("TMIL-ETH - Step 24: On-Chain Sensitivity Benchmark")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_file = RESULTS_DIR / "step2_features.pkl"
    
    TMIL_DIR = Path(__file__).parent
    
    # Check what GT files exist
    gap_days = [1, 3, 7, 14, 30]
    gt_files = {d: TMIL_DIR / f"time_aware_ground_truth_{d}d.json" for d in gap_days}
    
    if not all(f.exists() for f in gt_files.values()):
        print("Error: Missing some Ground Truth files. Run step22_time_aware_gt_builder.py first.")
        return
        
    print("\n[1] Preparing dataset...")
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    # We evaluate on the accounts present in the 30d file as a baseline
    with open(gt_files[30], "r", encoding="utf-8") as f:
        eval_addrs = {item["account_address"].lower() for item in json.load(f)}
        
    test_recs = [r for r in records if r["address"].lower() in eval_addrs]
    train_pool_phish = [r for r in records if r["address"].lower() not in eval_addrs and r["label"] == 1]
    train_pool_norm = [r for r in records if r["address"].lower() not in eval_addrs and r["label"] == 0]
    
    rng = np.random.RandomState(42)
    train_recs = rng.choice(train_pool_phish, min(100, len(train_pool_phish)), replace=False).tolist() + \
                 rng.choice(train_pool_norm, min(400, len(train_pool_norm)), replace=False).tolist()
                 
    print(f"  Test Set: {len(test_recs)} accounts.")
    
    print("\n[2] Training model (10 epochs) to extract Attention scores...")
    model = GatedTMILETH(4, 64).to(device)
    loss_fn = GatedCompoundLoss(lambda1=0.3)
    
    ds = AccountWindowDataset(train_recs, W=200)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for _ in range(5): train_one_epoch(model, loader, loss_fn, opt1, device, 1.0)
        
    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(5): train_one_epoch(model, loader, loss_fn, opt2, device, 1.0)

    print("\n[3] Running Sensitivity Benchmark across Temporal Gaps...")
    
    results = []
    for d in gap_days:
        h1, h3, h5, iou = evaluate_gt_file(gt_files[d], model, test_recs, eval_addrs, device)
        results.append({
            "Gap_Threshold": f"{d} days",
            "Hit@1": h1,
            "Hit@3": h3,
            "Hit@5": h5,
            "IoU": iou
        })
        print(f"  ✓ Gap {d} days -> Hit@1: {h1:.2f}% | IoU: {iou:.2f}%")
        
    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "step24_sensitivity_benchmark_results.csv"
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*70)
    print("  ON-CHAIN FORENSIC SENSITIVITY BENCHMARK")
    print("="*70)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*70)
    print(f"Detailed results saved to: {out_path}")

if __name__ == "__main__":
    main()
