"""
Step 02c: Analyze PG-EGAE MSE Distribution
==========================================
Analyzes the 5th feature (PG-EGAE MSE) across the entire TMIL dataset
to determine the optimal normalization strategy (Raw vs Z-score vs Log-norm).
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results")

def main():
    print("Loading step02b_features_hybrid.pkl...")
    file_path = RESULTS_DIR / 'step02b_features_hybrid.pkl'
    if not file_path.exists():
        print("File not found! Wait for step02b to finish.")
        return
        
    with open(file_path, 'rb') as f:
        records = pickle.load(f)
        
    all_mses = []
    phisher_mses = []
    normal_mses = []
    
    # Also collect per-account stats to see if Z-score makes sense
    intra_account_stds = []
    
    for r in records:
        hc = r['hand_crafted']
        if hc.shape[1] < 5:
            continue
            
        mse_col = hc[:, 4]
        
        all_mses.extend(mse_col.tolist())
        if r['label'] == 1:
            phisher_mses.extend(mse_col.tolist())
        else:
            normal_mses.extend(mse_col.tolist())
            
        if len(mse_col) > 1:
            intra_account_stds.append(np.std(mse_col))

    all_mses = np.array(all_mses)
    phisher_mses = np.array(phisher_mses)
    normal_mses = np.array(normal_mses)
    intra_account_stds = np.array(intra_account_stds)
    
    print("\n" + "="*50)
    print("PG-EGAE MSE DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Total Transactions: {len(all_mses):,}")
    print(f"  Phisher Txs:      {len(phisher_mses):,}")
    print(f"  Normal Txs:       {len(normal_mses):,}")
    
    print("\n[GLOBAL DISTRIBUTION]")
    print(f"  Min:    {np.min(all_mses):.4f}")
    print(f"  Max:    {np.max(all_mses):.4f}")
    print(f"  Mean:   {np.mean(all_mses):.4f}")
    print(f"  Median: {np.median(all_mses):.4f}")
    print(f"  StdDev: {np.std(all_mses):.4f}")
    
    print("\n[PERCENTILES]")
    print(f"  p75:    {np.percentile(all_mses, 75):.4f}")
    print(f"  p90:    {np.percentile(all_mses, 90):.4f}")
    print(f"  p95:    {np.percentile(all_mses, 95):.4f}")
    print(f"  p99:    {np.percentile(all_mses, 99):.4f}")
    print(f"  p99.9:  {np.percentile(all_mses, 99.9):.4f}")
    
    print("\n[CLASS SEPARABILITY (RAW)]")
    print(f"  Mean Phisher MSE: {np.mean(phisher_mses):.4f}")
    print(f"  Mean Normal MSE:  {np.mean(normal_mses):.4f}")
    
    print("\n[INTRA-ACCOUNT VARIANCE]")
    print(f"  Median Account StdDev: {np.median(intra_account_stds):.4f}")
    zero_var = np.sum(intra_account_stds == 0)
    print(f"  Accounts with 0 StdDev: {zero_var} ({(zero_var/len(intra_account_stds))*100:.1f}%)")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    # 1. Raw Histogram (log scale)
    plt.subplot(1, 2, 1)
    plt.hist(all_mses, bins=50, log=True, color='blue', alpha=0.7)
    plt.title("Raw MSE Distribution (Log-Y)")
    plt.xlabel("MSE Score")
    
    # 2. Log1p Normalized Histogram
    plt.subplot(1, 2, 2)
    log_mses = np.log1p(all_mses)
    plt.hist(log_mses, bins=50, log=True, color='green', alpha=0.7)
    plt.title("Log1p(MSE) Distribution (Log-Y)")
    plt.xlabel("log(1 + MSE)")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'step02c_mse_distribution.png')
    print(f"\nSaved distribution plot to: {RESULTS_DIR / 'step02c_mse_distribution.png'}")
    
if __name__ == "__main__":
    main()
