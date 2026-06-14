"""
Step 18: Chainabuse Ground Truth Reconstruction Pipeline (TxPhishScope Methodology)
===================================================================================
1. Loads the 3,921 phisher addresses from our local dataset.
2. Queries Chainabuse API for each address.
3. If reports exist, extracts the earliest report timestamp.
4. Uses Etherscan to find IN transactions up to 48 hours before that report.
5. Outputs `transaction_level_gt_chainabuse.json`.
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from dateutil import parser
from pathlib import Path
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data')
os.makedirs(RESULTS_DIR, exist_ok=True)

CHAINABUSE_API_KEY = "ca_dG96OUlqV0ZOY0lCTlI4Q3ZvbG9Td3JWLmN0eW82SDlrSnVpSHM2WldsaUhYN1E9PQ" 
ETHERSCAN_API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"

CHAINABUSE_ENDPOINT = "https://api.chainabuse.com/v0/reports"
ETHERSCAN_ENDPOINT = "https://api.etherscan.io/api"
LOOKBACK_HOURS = 48 

def load_phishers():
    df_in = pd.read_csv(DATA_DIR / 'phisher_transaction_in.csv', header=None, usecols=[6], dtype=str, low_memory=False)
    df_out = pd.read_csv(DATA_DIR / 'phisher_transaction_out.csv', header=None, usecols=[5], dtype=str, low_memory=False)
    phishers = set(df_in[6].dropna().str.lower().tolist()) | set(df_out[5].dropna().str.lower().tolist())
    return list(phishers)

def run_pipeline():
    phishers = load_phishers()
    print(f"[*] Loaded {len(phishers)} phisher addresses from local dataset.")
    
    final_gt = []
    total_victims = 0
    scammers_with_reports = 0
    
    auth = HTTPBasicAuth(CHAINABUSE_API_KEY, '')
    
    for i, addr in enumerate(phishers):
        # 1. Query Chainabuse
        try:
            ca_resp = requests.get(CHAINABUSE_ENDPOINT, auth=auth, params={'address': addr}, timeout=10)
            if ca_resp.status_code == 401:
                print("\n[!] API Key is invalid or expired. Exiting.")
                break
            
            if ca_resp.status_code == 200:
                reports = ca_resp.json()
                if reports and isinstance(reports, list):
                    scammers_with_reports += 1
                    
                    # Find earliest report
                    earliest_ts = float('inf')
                    earliest_iso = ""
                    for r in reports:
                        created_at = r.get("createdAt")
                        if created_at:
                            dt = parser.parse(created_at)
                            ts = int(dt.timestamp())
                            if ts < earliest_ts:
                                earliest_ts = ts
                                earliest_iso = created_at
                                
                    if earliest_ts == float('inf'):
                        continue
                        
                    # 2. Query Etherscan
                    lookback_ts = earliest_ts - (LOOKBACK_HOURS * 3600)
                    eth_params = {
                        "module": "account",
                        "action": "txlist",
                        "address": addr,
                        "startblock": 0,
                        "endblock": 99999999,
                        "page": 1,
                        "offset": 10000,
                        "sort": "asc",
                        "apikey": ETHERSCAN_API_KEY
                    }
                    
                    eth_resp = requests.get(ETHERSCAN_ENDPOINT, params=eth_params, timeout=10)
                    eth_data = eth_resp.json()
                    
                    if eth_data["status"] == "1" and isinstance(eth_data["result"], list):
                        txs = eth_data["result"]
                        victim_txs = []
                        
                        for idx, tx in enumerate(txs):
                            to_addr = str(tx.get("to", "")).lower()
                            val = int(tx.get("value", 0))
                            tx_ts = int(tx.get("timeStamp", 0))
                            is_error = tx.get("isError", "0")
                            
                            if is_error == "0" and to_addr == addr and val > 0:
                                if lookback_ts <= tx_ts <= earliest_ts:
                                    victim_txs.append({
                                        "tx_index": idx,
                                        "hash": tx.get("hash"),
                                        "from": tx.get("from"),
                                        "value_eth": val / 1e18,
                                        "timestamp": tx_ts,
                                        "hours_before_report": round((earliest_ts - tx_ts) / 3600, 2)
                                    })
                                    
                        if victim_txs:
                            final_gt.append({
                                "scammer_address": addr,
                                "earliest_report_iso": earliest_iso,
                                "report_count": len(reports),
                                "inferred_victim_txs": victim_txs
                            })
                            total_victims += len(victim_txs)
                            sys.stdout.write(f"\r[Hit] {addr}: {len(victim_txs)} victims ")
                            sys.stdout.flush()
        except Exception as e:
            pass
            
        # Rate limit to be safe for both APIs (Etherscan limit is 5/sec, Chainabuse limit unknown but play safe)
        time.sleep(0.3)
        
        if (i+1) % 100 == 0:
            print(f"\n[*] Progress: {i+1}/{len(phishers)} | Accounts with Reports: {scammers_with_reports} | Txs Extracted: {total_victims}")
            # Save checkpoint
            out_file = RESULTS_DIR / "transaction_level_gt_chainabuse.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_gt, f, indent=2)

    print(f"\n\n[+] Pipeline Complete!")
    print(f"    - Total Scammers with Reports: {scammers_with_reports}")
    print(f"    - Scammers with Extracted Victim Txs: {len(final_gt)}")
    print(f"    - Total Confirmed Victim Txs: {total_victims}")
    
    out_file = RESULTS_DIR / "transaction_level_gt_chainabuse.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_gt, f, indent=2)
    print(f"[+] Saved Ground Truth to: {out_file}")

if __name__ == "__main__":
    run_pipeline()
