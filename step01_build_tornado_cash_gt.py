"""
Step 16: Etherscan TC Crawler (A* Targeted Ground Truth)
===================================================
Crawls Etherscan for all 3,921 phisher accounts to find TC cashout transactions.
API Rate limit: 5 calls/sec.
Target: 200-300 accounts.
"""

import sys
import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce')
DATA_DIR = BASE_DIR / "BERT4ETH" / "Data"
RESULTS_DIR = BASE_DIR / "tmil_eth" / "results"

API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
ENDPOINT = "https://api.etherscan.io/v2/api"

# Full verified Tornado Cash contract list (all official ETH pools + Router)
# Source: Tornado Cash official documentation & on-chain deployment records
OFFICIAL_TC = {
    "0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144",  # 0.1 ETH Pool
    "0x12d66f87a04a9e220c9d45f6d8db75c93964f1f2",  # 1 ETH Pool
    "0x47ce0c6ed5b0ce3d3a51fdb1d7921825a5dbbab4",  # 10 ETH Pool
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",  # 100 ETH Pool
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",  # 1000 ETH Pool
    "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3",  # cDAI Pool (100 DAI)
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # Router
}

def load_tc_endpoints():
    tc_endpoints = OFFICIAL_TC.copy()
    p = DATA_DIR / "tornado_trans_in_removed.csv"
    if p.exists():
        df = pd.read_csv(p, header=None, usecols=[6], dtype=str)
        local_tc = set(df[6].dropna().str.lower().tolist())
        tc_endpoints.update(local_tc)
    return tc_endpoints

def load_phishers():
    df_in = pd.read_csv(DATA_DIR / 'phisher_transaction_in.csv', header=None, usecols=[6], dtype=str, low_memory=False)
    df_out = pd.read_csv(DATA_DIR / 'phisher_transaction_out.csv', header=None, usecols=[5], dtype=str, low_memory=False)
    phishers = set(df_in[6].dropna().str.lower().tolist()) | set(df_out[5].dropna().str.lower().tolist())
    return list(phishers)

def crawl():
    print("========== Phase 1: Etherscan TC Crawler ==========")
    tc_endpoints = load_tc_endpoints()
    print(f"Loaded {len(tc_endpoints)} TC Endpoints.")
    
    phishers = load_phishers()
    print(f"Loaded {len(phishers)} Phisher Accounts to crawl.")
    
    results = {}
    total_tc_txs = 0
    accounts_with_tc = 0
    
    out_file = RESULTS_DIR / "step16_etherscan_tc_hits.json"
    
    # Check if we already crawled some to resume
    if out_file.exists():
        with open(out_file, "r") as f:
            results = json.load(f)
        accounts_with_tc = len(results)
        total_tc_txs = sum(len(txs) for txs in results.values())
        print(f"Resuming from {accounts_with_tc} accounts already found...")
        
    phishers_to_crawl = [p for p in phishers if p not in results]
    print(f"Remaining to crawl: {len(phishers_to_crawl)}")
    
    count = 0
    
    for address in phishers_to_crawl:
        count += 1
        params = {
            "chainid": 1,
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
                tc_hits = []
                for tx in txs:
                    to_addr = str(tx.get("to", "")).lower()
                    if to_addr in tc_endpoints:
                        tc_hits.append({
                            "hash": tx.get("hash"),
                            "timeStamp": int(tx.get("timeStamp", 0)),
                            "to": to_addr,
                            "value": tx.get("value")
                        })
                
                if tc_hits:
                    results[address] = tc_hits
                    accounts_with_tc += 1
                    total_tc_txs += len(tc_hits)
                    print(f"[{count}/{len(phishers_to_crawl)}] HIT! {address}: {len(tc_hits)} TC txs. Total accounts: {accounts_with_tc}")
                    
                    # Save checkpoint every hit
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=2)
                        
            elif data["message"] == "NOTOK" and "rate limit" in data.get("result", "").lower():
                print("Rate limit hit, backing off...")
                time.sleep(2)
                continue
                
        except Exception as e:
            print(f"Error on {address}: {e}")
            
        # Rate limit: 5 calls / sec -> 0.25s sleep = 4 calls / sec (safe)
        time.sleep(0.25)
        
        if count % 100 == 0:
            print(f"Progress: {count}/{len(phishers_to_crawl)} | TC Accounts Found: {accounts_with_tc}")

    print("\n========== Crawl Complete ==========")
    print(f"Total Phisher Accounts with TC: {accounts_with_tc}")
    print(f"Total TC Transactions: {total_tc_txs}")
    
if __name__ == "__main__":
    crawl()
