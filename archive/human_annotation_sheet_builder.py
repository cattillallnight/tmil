import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
step17_annotation_sheet_builder.py (v2)
========================================
Tạo Annotation Sheet sạch để bạn spot-check thủ công 100+ cases.

Với mỗi case, sheet cung cấp:
  - Link trực tiếp đến trang Etherscan của Phisher
  - Link trực tiếp đến giao dịch Max Outgoing (cashout_tx)
  - Giá trị ETH của giao dịch
  - Địa chỉ nhận (recipient)
  - Thời gian giao dịch
  - Cột trống để bạn điền: CONFIRMED / REJECTED / UNCERTAIN

Hướng dẫn spot-check (cho 1 case, mất ~1-2 phút):
  1. Click link 'cashout_tx_link' → Xem giao dịch trên Etherscan
  2. Kiểm tra:
     a) Đây có phải giao dịch OUT lớn nhất không?
     b) Thời gian có nằm gần sau đợt nhận tiền lừa đảo không?
     c) Giá trị có lớn bất thường so với các giao dịch xung quanh không?
  3. Điền CONFIRMED nếu giao dịch này rõ ràng là "dọn sạch ví" (cashout/consolidation)
     Điền REJECTED nếu giao dịch này có vẻ bình thường
     Điền UNCERTAIN nếu không chắc chắn
"""

import pandas as pd
import numpy as np
from pathlib import Path

TMIL_DIR    = Path(__file__).parent
RESULTS_DIR = TMIL_DIR / "results"
INPUT_CSV   = RESULTS_DIR / "step16_etherscan_labels.csv"
OUTPUT_CSV  = RESULTS_DIR / "step17_annotation_sheet.csv"


def classify_value(val_eth: float) -> str:
    """
    Auto-classify dựa trên giá trị ETH của Max Outgoing transaction.
    Đây là heuristic đơn giản: giao dịch càng lớn, càng có khả năng là cashout.
    """
    if val_eth <= 0:
        return "NO_TX"
    elif val_eth < 0.05:
        return "MICRO (<0.05 ETH)"
    elif val_eth < 0.5:
        return "SMALL (0.05-0.5 ETH)"
    elif val_eth < 5.0:
        return "MEDIUM (0.5-5 ETH)"
    elif val_eth < 50.0:
        return "LARGE (5-50 ETH)"
    else:
        return "VERY LARGE (>50 ETH)"


def main():
    print("=" * 70)
    print("Step 17 (v2): Annotation Sheet Builder for Manual Spot-Check")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} not found. Run step16 first!")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"\n[1] Loaded {len(df)} cases from step16.")

    # Lọc bỏ các case không có giao dịch
    df_valid = df[df["cashout_value_eth"] > 0].copy()
    df_invalid = df[df["cashout_value_eth"] <= 0].copy()
    print(f"    Valid cases (has cashout tx): {len(df_valid)}")
    print(f"    Invalid (no out tx found)   : {len(df_invalid)}")

    # Tạo các cột bổ sung
    df_valid["value_category"]       = df_valid["cashout_value_eth"].apply(classify_value)
    df_valid["annotator_1_verdict"]  = ""   # CONFIRMED / REJECTED / UNCERTAIN
    df_valid["annotator_1_notes"]    = ""

    # Sắp xếp theo giá trị (lớn nhất trước để dễ spot-check)
    df_valid = df_valid.sort_values("cashout_value_eth", ascending=False).reset_index(drop=True)

    # Xây dựng annotation sheet sạch
    sheet_cols = [
        "account_address",
        "etherscan_link",
        "cashout_tx_link",
        "cashout_value_eth",
        "value_category",
        "cashout_recipient",
        "cashout_timestamp",
        "etherscan_tx_count",
        "victim_campaign_start_idx",
        "cashout_tx_idx_gt",
        "annotator_1_verdict",   # <-- BẠN ĐIỀN VÀO ĐÂY
        "annotator_1_notes",
    ]
    existing_cols = [c for c in sheet_cols if c in df_valid.columns]
    df_out = df_valid[existing_cols].copy()

    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # Thống kê phân phối giá trị
    print("\n[2] Value Distribution of Cashout Transactions:")
    print("-" * 50)
    cats = df_valid["value_category"].value_counts().sort_index()
    for cat, cnt in cats.items():
        print(f"    {cat:30} : {cnt:4d} cases ({cnt/len(df_valid)*100:.1f}%)")

    total_eth = df_valid["cashout_value_eth"].sum()
    mean_eth  = df_valid["cashout_value_eth"].mean()
    median_eth= df_valid["cashout_value_eth"].median()

    print(f"\n    Total ETH in cashout txs : {total_eth:,.2f} ETH")
    print(f"    Mean cashout value       : {mean_eth:.2f} ETH")
    print(f"    Median cashout value     : {median_eth:.2f} ETH")

    print("\n" + "=" * 70)
    print("  STEP 17 — ANNOTATION SHEET READY")
    print("=" * 70)
    print(f"  Total annotatable cases  : {len(df_out)}")
    print(f"  Saved to: {OUTPUT_CSV}")
    print("""
  [HƯỚNG DẪN SPOT-CHECK cho Bạn (Annotator 1)]:
  ─────────────────────────────────────────────
  1. Mở file CSV bằng Excel hoặc Google Sheets.
  2. Với mỗi hàng (đặc biệt các hàng LARGE/VERY LARGE):
     a) Click cột 'cashout_tx_link' để xem giao dịch trên Etherscan.
     b) Click cột 'etherscan_link' để xem toàn bộ lịch sử ví.
     c) Kiểm tra: giao dịch này có phải là "dọn sạch ví" (gom tiền và xả đi)?
        → Dấu hiệu: số round (1 ETH, 10 ETH, 50 ETH) hoặc xả gần hết số dư.
  3. Điền cột 'annotator_1_verdict':
     → CONFIRMED  : Rõ ràng là cashout/consolidation
     → REJECTED   : Giao dịch bình thường, không liên quan
     → UNCERTAIN  : Không chắc chắn
  4. Sau khi annotate xong ≥ 50 rows, chạy: python step18_cohen_kappa.py
  ─────────────────────────────────────────────
  Tip: Ưu tiên annotate các hàng có value lớn (LARGE/VERY LARGE) trước!
  """)


if __name__ == "__main__":
    main()
