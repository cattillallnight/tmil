"""
Step 02d: Normalize PG-EGAE MSE Feature (Log1p)
=================================================
Apply log1p normalization to the 5th feature (PG-EGAE MSE) in the hybrid
feature set. This handles the heavy-tailed distribution.

Input:  step02b_features_hybrid.pkl  (raw MSE, feature[:, 4])
Output: step02d_features_hybrid_norm.pkl  (log1p(MSE) normalized)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

RESULTS_DIR = Path(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results")

def main():
    in_path  = RESULTS_DIR / 'step02b_features_hybrid.pkl'
    out_path = RESULTS_DIR / 'step02d_features_hybrid_norm.pkl'
    
    print(f"Loading {in_path.name}...")
    with open(in_path, 'rb') as f:
        records = pickle.load(f)
    print(f"Loaded {len(records):,} records.")

    # --- Collect global stats on raw MSE for reporting ---
    all_raw_mse = []
    for r in records:
        hc = r['hand_crafted']
        if hc.shape[1] >= 5:
            all_raw_mse.extend(hc[:, 4].tolist())
    all_raw_mse = np.array(all_raw_mse, dtype=np.float32)
    
    print(f"\n[Raw MSE Stats]")
    print(f"  Mean: {np.mean(all_raw_mse):.4f}")
    print(f"  Max:  {np.max(all_raw_mse):.4f}")
    print(f"  Phisher separability already computed in step02c.")

    # --- Apply log1p normalization to feature[:, 4] ---
    print("\nApplying log1p normalization to PG-EGAE MSE feature...")
    norm_records = []
    for r in tqdm(records):
        hc = r['hand_crafted']
        if hc.shape[1] >= 5:
            hc = hc.copy()
            hc[:, 4] = np.log1p(hc[:, 4])
        r['hand_crafted'] = hc
        norm_records.append(r)

    # --- Verify normalized distribution ---
    all_log_mse = []
    phisher_log = []
    normal_log  = []
    for r in norm_records:
        hc = r['hand_crafted']
        if hc.shape[1] >= 5:
            vals = hc[:, 4].tolist()
            all_log_mse.extend(vals)
            if r['label'] == 1:
                phisher_log.extend(vals)
            else:
                normal_log.extend(vals)
    
    phisher_log = np.array(phisher_log)
    normal_log  = np.array(normal_log)
    
    print(f"\n[Log1p MSE Stats After Normalization]")
    print(f"  Global Mean:    {np.mean(all_log_mse):.4f}")
    print(f"  Global Max:     {np.max(all_log_mse):.4f}")
    print(f"  Phisher Mean:   {np.mean(phisher_log):.4f}")
    print(f"  Normal Mean:    {np.mean(normal_log):.4f}")
    ratio = np.mean(phisher_log) / (np.mean(normal_log) + 1e-9)
    print(f"  Separation Ratio (log): {ratio:.2f}x")

    # --- Save ---
    print(f"\nSaving {len(norm_records):,} records to {out_path.name}...")
    with open(out_path, 'wb') as f:
        pickle.dump(norm_records, f)

    print("\n[OK] Step 02d Complete.")

if __name__ == '__main__':
    main()
