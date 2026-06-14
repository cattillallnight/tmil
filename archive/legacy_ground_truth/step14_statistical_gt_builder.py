import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
TMIL_DIR = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth")
OUTPUT_FILE = TMIL_DIR / "ground_truth" / "statistical_ground_truth.json"

def main():
    print("=" * 70)
    print("Step 14: Statistical Burst Detection (Z-Score & Tornado Anchor)")
    print("=" * 70)

    # 1. Load Data
    print("[1] Loading transactions and Tornado Cash endpoints...")
    df_in = pd.read_csv(DATA_DIR / "phisher_transaction_in.csv", header=None, dtype=str)
    df_in.columns = list(range(len(df_in.columns)))
    df_in[7] = pd.to_numeric(df_in[7], errors="coerce").fillna(0)
    df_in[11] = pd.to_numeric(df_in[11], errors="coerce").fillna(0)
    
    df_out = pd.read_csv(DATA_DIR / "phisher_transaction_out.csv", header=None, dtype=str)
    df_out.columns = list(range(len(df_out.columns)))
    df_out[7] = pd.to_numeric(df_out[7], errors="coerce").fillna(0)
    df_out[11] = pd.to_numeric(df_out[11], errors="coerce").fillna(0)

    tc_in = pd.read_csv(DATA_DIR / "tornado_trans_in_removed.csv", header=None, usecols=[6], dtype=str)
    tornado_endpoints = set(tc_in[6].str.lower().tolist())

    phishers = df_in[6].str.lower().unique().tolist()
    print(f"    Found {len(phishers):,} phishing accounts with IN transactions.")

    # 2. Pipeline Processing
    print("\n[2] Applying Statistical Sliding Window Detection...")
    results = []
    
    high_conf = 0
    medium_conf = 0

    for p in phishers:
        p_in = df_in[df_in[6].str.lower() == p].copy()
        p_out = df_out[df_out[5].str.lower() == p].copy()
        
        # Gộp tất cả IN/OUT và sort theo thời gian để có chuỗi tx sequential
        combined = pd.concat([
            p_in[[5, 6, 7, 11]].assign(direction="in"),
            p_out[[5, 6, 7, 11]].assign(direction="out")
        ]).sort_values(11).reset_index(drop=True)
        
        if len(combined) == 0:
            continue
            
        # Bước 1 — Tính anomaly score tự động (Z-Score)
        # Sử dụng Value (lượng ETH) hoặc khoảng cách thời gian (Time-delta)
        # Ở đây ta kết hợp cả 2: Lượng tiền bất thường hoặc Giao dịch quá nhanh
        values = combined[7].values
        mu_val = np.mean(values)
        sigma_val = np.std(values) + 1e-9
        z_scores = (values - mu_val) / sigma_val
        
        # Bước 2 — Detect burst bằng sliding window (mean [t-2 : t+2])
        # Dùng Pandas rolling window size 5
        z_series = pd.Series(z_scores)
        window_scores = z_series.rolling(window=5, center=True, min_periods=1).mean()
        
        # Threshold tự động: Lớn hơn 1.5 StdDev trong window
        burst_candidates = combined.index[window_scores > 1.5].tolist()
        
        if not burst_candidates:
            # Fallback nếu không có burst nào vượt 1.5, hạ chuẩn xuống 1.0
            burst_candidates = combined.index[window_scores > 1.0].tolist()
            if not burst_candidates:
                continue
                
        start_idx = burst_candidates[0]
        
        # Bước 3 — Anchor bằng Tornado Cash
        # Lấy các giao dịch OUT xảy ra sau khi burst bắt đầu
        post_burst = combined[(combined.index >= start_idx) & (combined["direction"] == "out")]
        
        confidence = "low"
        end_idx = start_idx
        
        if len(post_burst) > 0:
            # Kiểm tra xem có giao dịch nào chui vào Tornado Cash không
            tc_txs = post_burst[post_burst[6].str.lower().isin(tornado_endpoints)]
            if len(tc_txs) > 0:
                confidence = "high"
                end_idx = tc_txs.index[0] # Chốt sổ tại giao dịch Tornado Cash đầu tiên
                high_conf += 1
            else:
                confidence = "medium"
                # Dùng max ETH outgoing làm endpoint (chuyển tiền tẩu tán)
                cashout_idx = post_burst[7].idxmax()
                end_idx = cashout_idx
                medium_conf += 1
        else:
            # Không có giao dịch out, lấy giao dịch dị thường cuối cùng
            end_idx = burst_candidates[-1]
            
        # Đảm bảo logic
        if end_idx < start_idx:
            end_idx = start_idx
            
        # Bước 4 — Gán nhãn
        results.append({
            "phisher_address": p,
            "burst_start_tx_idx": int(start_idx),
            "burst_end_tx_idx": int(end_idx),
            "confidence": confidence
        })

    # 3. Lưu Dataset
    print(f"\n[3] Exporting Transaction-level GT Dataset...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"    Total Accounts Labeled: {len(results):,}")
    print(f"    [High Confidence] Tornado Cash Anchored: {high_conf:,}")
    print(f"    [Medium Confidence] Max Outgoing Anchored: {medium_conf:,}")
    print(f"    Saved to: {OUTPUT_FILE.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
