"""
Step 18: Chainabuse Ground Truth Reconstruction
===================================================
1. Reads chainabuse_reports.json (from browser subagent scraping).
2. Queries Etherscan API for each reported scammer address.
3. Infers victim transactions by looking at IN transactions 
   that occurred up to 48 hours BEFORE the report timestamp.
"""

import os
import sys
import json
import time
import requests
import dateparser
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
REPORTS_FILE = RESULTS_DIR / "chainabuse_reports.json"
OUT_FILE = RESULTS_DIR / "chainabuse_victim_txs.json"

API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
ENDPOINT = "https://api.etherscan.io/api"

def parse_time(time_str):
    """Parse various time formats to a Unix timestamp."""
    try:
        # e.g., '2 hours ago', 'Oct 15, 2023'
        dt = dateparser.parse(time_str)
        if dt:
            # Ensure timezone awareness
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
    return int(time.time()) # fallback to now

def reconstruct_victims():
    print("========== Phase 2 & 3: Etherscan Sync & Victim Inference ==========")
    if not REPORTS_FILE.exists():
        print(f"Error: {REPORTS_FILE} not found. Waiting for scraper.")
        return
        
    with open(REPORTS_FILE, "r", encoding="utf-8") as f:
        reports = json.load(f)
        
    print(f"Loaded {len(reports)} reports from Chainabuse.")
    
    inferred_data = []
    total_victims = 0
    
    for idx, report in enumerate(reports):
        address = str(report.get("scammer_address", "")).strip().lower()
        time_str = report.get("report_time", "")
        desc = report.get("description", "")
        
        if not address or not address.startswith("0x"):
            continue
            
        report_ts = parse_time(time_str)
        # Lookback window: 48 hours before report
        lookback_ts = report_ts - (48 * 3600)
        
        print(f"[{idx+1}/{len(reports)}] Querying Etherscan for {address}...")
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 10000,
            "sort": "asc",
            "apikey": API_KEY
        }
        
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=10)
            data = resp.json()
            
            if data["status"] == "1" and isinstance(data["result"], list):
                txs = data["result"]
                victim_txs = []
                
                for tx in txs:
                    to_addr = str(tx.get("to", "")).lower()
                    val = int(tx.get("value", 0))
                    tx_ts = int(tx.get("timeStamp", 0))
                    
                    # IN transaction (money sent TO scammer) AND > 0 value
                    if to_addr == address and val > 0:
                        # Check if within the 48h lookback window
                        if lookback_ts <= tx_ts <= report_ts:
                            val_eth = val / 1e18
                            victim_txs.append({
                                "hash": tx.get("hash"),
                                "from": tx.get("from"),
                                "value_eth": val_eth,
                                "timestamp": tx_ts,
                                "time_diff_hours": (report_ts - tx_ts) / 3600
                            })
                            
                if victim_txs:
                    inferred_data.append({
                        "scammer_address": address,
                        "report_time": time_str,
                        "report_ts": report_ts,
                        "description": desc,
                        "inferred_victim_txs": victim_txs
                    })
                    total_victims += len(victim_txs)
                    print(f"  -> Found {len(victim_txs)} likely victim transactions!")
                else:
                    print(f"  -> No IN txs found within 48h before {time_str}")
                    
        except Exception as e:
            print(f"  Error querying {address}: {e}")
            
        time.sleep(0.25) # Rate limit
        
    print("\n========== Reconstruction Complete ==========")
    print(f"Total Reports Processed: {len(reports)}")
    print(f"Total Scammer Accounts with Inferred Txs: {len(inferred_data)}")
    print(f"Total Confirmed Victim Txs (Ground Truth): {total_victims}")
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(inferred_data, f, indent=2)
        
    print(f"Saved dataset to {OUT_FILE}")

if __name__ == "__main__":
    reconstruct_victims()
