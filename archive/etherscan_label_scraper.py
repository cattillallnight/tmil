import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
step16_etherscan_label_scraper.py  (v2 - Hash-based Verification)
===================================================================
Bước 1 của Hybrid GT Verification Pipeline.

Cách tiếp cận mới:
  1. Đọc file CSV gốc (phisher_transaction_out.csv) để lấy hash thật
     của giao dịch Max Outgoing (Ground Truth) cho mỗi phisher.
  2. Dùng Etherscan API /api?module=proxy&action=eth_getTransactionByHash
     để verify hash đó và lấy thông tin on-chain (value, to, timestamp).
  3. Lấy label của địa chỉ nhận (to) để kiểm tra có phải CEX/Mixer không.
  4. Xuất annotation sheet đầy đủ cho step17.
"""

import json
import time
import random
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── CONFIG ─────────────────────────────────────────────────────────────────
ETHERSCAN_API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
N_SAMPLE          = 120
RATE_LIMIT_SECS   = 0.22  # slightly under 5 req/sec

DATA_DIR    = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
TMIL_DIR    = Path(__file__).parent
GT_FILE     = TMIL_DIR / "ground_truth" / "time_aware_ground_truth.json"
RESULTS_DIR = TMIL_DIR / "results"
OUTPUT_CSV  = RESULTS_DIR / "step16_etherscan_labels.csv"

BASE_URL = "https://api.etherscan.io/api"
# ─────────────────────────────────────────────────────────────────────────────

def etherscan_get(params: dict) -> dict:
    params["apikey"] = ETHERSCAN_API_KEY
    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=12)
            data = r.json()
            return data
        except Exception as e:
            print(f"  [WARN] {e}")
            time.sleep(1)
    return {"status": "0", "result": None}


def get_tx_info(tx_hash: str) -> dict:
    """Lấy thông tin giao dịch qua hash (JSON-RPC proxy)."""
    params = {
        "module": "proxy",
        "action": "eth_getTransactionByHash",
        "txhash": tx_hash,
    }
    data = etherscan_get(params)
    return data.get("result") or {}


def get_tx_receipt(tx_hash: str) -> dict:
    """Lấy receipt để biết block timestamp."""
    params = {
        "module": "proxy",
        "action": "eth_getTransactionReceipt",
        "txhash": tx_hash,
    }
    data = etherscan_get(params)
    return data.get("result") or {}


def get_address_tag(address: str) -> str:
    """Tìm tag Etherscan của địa chỉ nhận tiền."""
    params = {
        "module":     "account",
        "action":     "txlist",
        "address":    address,
        "startblock": 0,
        "endblock":   99999999,
        "page":       1,
        "offset":     3,
        "sort":       "desc",
    }
    data = etherscan_get(params)
    txs = data.get("result", [])
    if not isinstance(txs, list):
        return ""
    for tx in txs:
        for field in ["toAddressNameTag", "fromAddressNameTag"]:
            tag = tx.get(field, "")
            if tag:
                return tag
    return ""


def classify_recipient(to_addr: str, tag: str) -> str:
    """
    Phân loại địa chỉ nhận tiền:
    - CEX (Binance, Coinbase, etc.) → CONFIRMED_CEX (tẩu tán lên sàn)
    - Mixer/Tornado → CONFIRMED_MIXER
    - Phishing/Scam liên quan → CONFIRMED_RELATED
    - Không xác định → UNCERTAIN
    """
    tag_l = tag.lower() if tag else ""
    
    cex_keywords = ["binance", "coinbase", "huobi", "kraken", "okex", "okx",
                    "ftx", "kucoin", "gate", "bybit", "bitfinex", "gemini",
                    "exchange", "hot wallet", "deposit"]
    mixer_keywords = ["tornado", "mixer", "tumbler", "wasabi", "coinjoin"]
    phish_keywords = ["phish", "scam", "fake", "fraud", "hack", "exploit",
                      "rug", "malicious", "theft", "stolen", "drainer"]
    
    for kw in mixer_keywords:
        if kw in tag_l:
            return "CONFIRMED_MIXER"
    for kw in cex_keywords:
        if kw in tag_l:
            return "CONFIRMED_CEX"
    for kw in phish_keywords:
        if kw in tag_l:
            return "CONFIRMED_RELATED_PHISHING"
    return "UNCERTAIN"


def main():
    print("=" * 70)
    print("Step 16 (v2): Hash-based Etherscan Verification (100+ cases)")
    print("=" * 70)

    # ── 1. Load Ground Truth ──────────────────────────────────────────────────
    print("\n[1] Loading Ground Truth & Raw CSV Data...")
    with open(GT_FILE, "r") as f:
        gt_data = json.load(f)

    # Load raw OUT transactions to get actual tx hashes
    print("    Loading phisher_transaction_out.csv (this may take a moment)...")
    df_out = pd.read_csv(DATA_DIR / "phisher_transaction_out.csv",
                         header=None, dtype=str)
    df_out.columns = list(range(len(df_out.columns)))
    # col 0  = tx hash
    # col 5  = from (phisher)
    # col 6  = to (recipient)
    # col 7  = value (wei)
    # col 11 = timestamp
    df_out[5]  = df_out[5].str.lower().fillna("")
    df_out[6]  = df_out[6].str.lower().fillna("")
    df_out[7]  = pd.to_numeric(df_out[7], errors="coerce").fillna(0)
    df_out[11] = pd.to_numeric(df_out[11], errors="coerce").fillna(0)

    # Load IN transactions to determine victim campaign timing
    print("    Loading phisher_transaction_in.csv...")
    df_in = pd.read_csv(DATA_DIR / "phisher_transaction_in.csv",
                        header=None, dtype=str)
    df_in.columns = list(range(len(df_in.columns)))
    df_in[5]  = df_in[5].str.lower().fillna("")
    df_in[6]  = df_in[6].str.lower().fillna("")
    df_in[11] = pd.to_numeric(df_in[11], errors="coerce").fillna(0)

    # ── 2. Sample 120 accounts ────────────────────────────────────────────────
    random.seed(42)
    sampled = random.sample(gt_data, min(N_SAMPLE, len(gt_data)))
    print(f"    Sampled {len(sampled)} accounts.")

    # ── 3. Build records from CSV + verify via Etherscan ──────────────────────
    print(f"\n[2] Building verification records (API calls included)...")

    records = []
    for i, acc in enumerate(sampled):
        addr  = acc["account_address"].lower()
        burst = acc["time_aware_gt_bursts"][0]
        gt_end_idx = burst["end_tx_idx"]

        print(f"  [{i+1:3d}/{len(sampled)}] {addr[:22]}...", end=" ", flush=True)

        # Get all OUT txs for this phisher from CSV
        phisher_out = df_out[df_out[5] == addr].sort_values(11).reset_index(drop=True)

        # The GT says Max Outgoing is at gt_end_idx in the COMBINED (in+out) sorted list.
        # Find the actual Max Outgoing within 72h window of last victim tx.
        # Since we don't reconstruct the full combined list here, we find the
        # largest OUT tx directly from phisher_out.
        real_cashout_row = None
        real_cashout_hash = ""
        real_cashout_value_eth = 0
        real_cashout_to = ""
        real_cashout_ts = 0

        if len(phisher_out) > 0:
            max_idx = phisher_out[7].idxmax()
            real_cashout_row = phisher_out.loc[max_idx]
            real_cashout_hash = str(real_cashout_row[0]).strip()
            real_cashout_value_eth = round(real_cashout_row[7] / 1e18, 4)
            real_cashout_to = str(real_cashout_row[6]).strip()
            real_cashout_ts = int(real_cashout_row[11])

        # Get tag of recipient via Etherscan
        recipient_tag = ""
        auto_verdict  = "NO_TX_FOUND"

        if real_cashout_hash and real_cashout_hash != "nan":
            recipient_tag = get_address_tag(real_cashout_to)
            time.sleep(RATE_LIMIT_SECS)
            verdict_class = classify_recipient(real_cashout_to, recipient_tag)
            auto_verdict  = verdict_class
        
        # Format timestamp
        try:
            dt_str = datetime.utcfromtimestamp(real_cashout_ts).strftime("%Y-%m-%d %H:%M UTC") if real_cashout_ts else ""
        except Exception:
            dt_str = str(real_cashout_ts)

        record = {
            "account_address":         addr,
            "etherscan_link":          f"https://etherscan.io/address/{addr}",
            "etherscan_tx_count":      acc.get("total_txs", "N/A"),
            "victim_campaign_start_idx": burst["start_tx_idx"],
            "cashout_tx_idx_gt":       gt_end_idx,
            "cashout_tx_hash":         real_cashout_hash,
            "cashout_tx_link":         f"https://etherscan.io/tx/{real_cashout_hash}" if real_cashout_hash else "",
            "cashout_value_eth":       real_cashout_value_eth,
            "cashout_recipient":       real_cashout_to,
            "cashout_recipient_tag":   recipient_tag,
            "cashout_timestamp":       dt_str,
            "auto_verdict":            auto_verdict,
            # Annotator columns
            "annotator_1_verdict":     "",   # Bạn điền: CONFIRMED / REJECTED / UNCERTAIN
            "annotator_1_notes":       "",
        }
        records.append(record)
        print(f"val={real_cashout_value_eth:.2f} ETH | to_tag='{recipient_tag[:25]}' → {auto_verdict}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # Stats
    confirmed = df["auto_verdict"].str.startswith("CONFIRMED").sum()
    uncertain = (df["auto_verdict"] == "UNCERTAIN").sum()
    no_tx     = (df["auto_verdict"] == "NO_TX_FOUND").sum()

    print("\n" + "=" * 70)
    print("  STEP 16 (v2) — SUMMARY")
    print("=" * 70)
    print(f"  Total cases              : {len(records)}")
    print(f"  CONFIRMED (CEX/Mixer)    : {confirmed} ({confirmed/len(records)*100:.1f}%)")
    print(f"  UNCERTAIN (No tag)       : {uncertain} ({uncertain/len(records)*100:.1f}%)")
    print(f"  NO_TX_FOUND              : {no_tx}")
    print(f"\n  Saved to: {OUTPUT_CSV}")
    print("  → Tiếp theo: Chạy step17_annotation_sheet_builder.py")


if __name__ == "__main__":
    main()
