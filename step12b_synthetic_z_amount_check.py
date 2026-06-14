import pickle
import numpy as np
from scipy import stats
import math
from utils import RESULTS_DIR

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0.0
    pool_sd = math.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pool_sd == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd

def main():
    print("--- Phân tích Circularity: Synthetic Injection Check ---")
    
    # 1. Load Step 02 features (we only need values from normal accounts to inject)
    # Actually step02 features has already computed z_amount.
    # To do a synthetic injection, we need the raw values.
    # Let's just create 1000 synthetic normal accounts
    
    synthetic_burst_z = []
    synthetic_normal_z = []
    
    np.random.seed(42)
    for _ in range(1000):
        # Mô phỏng một account bình thường: 100 giao dịch, giá trị trung bình 0.5 ETH, std=0.2
        values = np.random.normal(loc=0.5, scale=0.2, size=100)
        values = np.clip(values, 0.01, None)
        
        # Inject Synthetic Burst: Rửa tiền 100 ETH x 3 giao dịch
        burst_indices = [50, 51, 52]
        values[burst_indices] = 100.0
        
        # Tính lại z_amount cho account này theo đúng công thức của step04
        mu = values.mean()
        sigma = values.std() + 1e-9
        z_amount = np.clip((values - mu) / sigma, -3.0, 3.0) / 3.0
        
        for i in range(100):
            if i in burst_indices:
                synthetic_burst_z.append(z_amount[i])
            else:
                synthetic_normal_z.append(z_amount[i])
                
    burst_vals = np.array(synthetic_burst_z)
    normal_vals = np.array(synthetic_normal_z)
    
    print(f"Total Synthetic Burst Txs: {len(burst_vals)}")
    print(f"Total Synthetic Normal Txs: {len(normal_vals)}")
    print(f"Z-amount Burst Mean  : {np.mean(burst_vals):.4f}")
    print(f"Z-amount Normal Mean : {np.mean(normal_vals):.4f}")
    
    d_val = cohens_d(burst_vals, normal_vals)
    u_stat, p_val = stats.mannwhitneyu(burst_vals, normal_vals, alternative='two-sided')
    
    print(f"\n[Statistical Test on SYNTHETIC DATA]")
    print(f"Cohen's d = {d_val:.4f} | p-value = {p_val:.2e}")
    
if __name__ == "__main__":
    main()
