import json
import pickle
import numpy as np
from pathlib import Path
from utils import RESULTS_DIR

def calculate_iou(pred_set, gt_set):
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    return intersection / union if union > 0 else 0

def main():
    print("="*60)
    print("TMIL-ETH: Track B Heuristic Baselines (Step 18)")
    print("="*60)

    gt_file = "human_ground_truth.json"
    features_file = RESULTS_DIR / "step2_features.pkl"

    if not Path(gt_file).exists():
        print(f"Error: {gt_file} not found.")
        return

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(features_file, "rb") as f:
        records = pickle.load(f)

    # Fast lookup for records
    record_dict = {r["address"].lower(): r for r in records}
    
    total_accs = 0
    hit1_first = 0
    hit1_max_val = 0
    
    # Random baseline variables
    sum_inv_N = 0.0
    
    # Restricted >= 5 windows variables
    total_ge5 = 0
    sum_inv_N_ge5 = 0.0

    print("Evaluating heuristics...")
    for gt in gt_data:
        addr = gt["account_address"].lower()
        if addr not in record_dict:
            continue
            
        rec = record_dict[addr]
        wins = rec["windows"]
        hc = rec["hand_crafted"] # (seq_len, 4)
        
        gt_set = set(gt["ground_truth_indices"])
        if len(gt_set) == 0 or len(wins) == 0:
            continue
            
        N = len(wins)
        total_accs += 1
        sum_inv_N += 1.0 / N
        
        if N >= 5:
            total_ge5 += 1
            sum_inv_N_ge5 += 1.0 / N

        # 1. First Window Heuristic
        first_win_start, first_win_end = wins[0]
        first_pred_set = set(range(first_win_start, first_win_end))
        if calculate_iou(first_pred_set, gt_set) > 0:
            hit1_first += 1

        # 2. Max Value Window Heuristic
        # We calculate the total z_amount (idx 0) for each window
        win_values = []
        for (start, end) in wins:
            # sum of transaction values in this window
            val = np.sum(hc[start:end, 0])
            win_values.append(val)
            
        max_idx = np.argmax(win_values)
        max_win_start, max_win_end = wins[max_idx]
        max_pred_set = set(range(max_win_start, max_win_end))
        if calculate_iou(max_pred_set, gt_set) > 0:
            hit1_max_val += 1

    print(f"\nTotal Evaluated Accounts: {total_accs}")
    print(f"Total Accounts with >= 5 windows: {total_ge5}")
    
    print("\n--- Hit@1 Results ---")
    print(f"Random Baseline (Expected): {(sum_inv_N / total_accs) * 100:.2f}%")
    print(f"First Window Heuristic:     {(hit1_first / total_accs) * 100:.2f}%")
    print(f"Max Value Heuristic:        {(hit1_max_val / total_accs) * 100:.2f}%")
    
    print("\n--- Restricted (N >= 5) ---")
    print(f"Random Baseline (N>=5):     {(sum_inv_N_ge5 / total_ge5) * 100:.2f}%")

if __name__ == "__main__":
    main()
