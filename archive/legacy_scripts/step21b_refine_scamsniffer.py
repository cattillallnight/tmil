"""
Step 21b: Refine ScamSniffer Ground Truth (Single Threaded, Rate-Limit Safe)
=============================================================================
"""

import sys
import json
import time
import requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
IN_FILE = RESULTS_DIR / 'scamsniffer_txlevel_dataset.json'
OUT_FILE = RESULTS_DIR / 'scamsniffer_txlevel_dataset_refined.json'

ETHERSCAN_API = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"

def get_tx_from_address(tx_hash):
    resp = requests.get('https://api.etherscan.io/v2/api', params={
        'chainid': 1,
        'module': 'proxy',
        'action': 'eth_getTransactionByHash',
        'txhash': tx_hash,
        'apikey': ETHERSCAN_API
    }, timeout=15)
    data = resp.json()
    if data and 'result' in data:
        res = data['result']
        if isinstance(res, dict):
            return res.get('from', '').lower()
        else:
            raise Exception("Rate limit or invalid response: " + str(res))
    return None

def is_contract(address):
    resp = requests.get('https://api.etherscan.io/v2/api', params={
        'chainid': 1,
        'module': 'proxy',
        'action': 'eth_getCode',
        'address': address,
        'tag': 'latest',
        'apikey': ETHERSCAN_API
    }, timeout=15)
    data = resp.json()
    if data and 'result' in data:
        code = data['result']
        if isinstance(code, str) and code.startswith('0x'):
            return code != '0x' and code != ''
        else:
            raise Exception("Rate limit or invalid response: " + str(code))
    return False

def main():
    print("Loading original dataset...")
    with open(IN_FILE, 'r') as f:
        dataset = json.load(f)
        
    total_original = sum(d.get('victim_tx_count', 0) for d in dataset)
    print(f"Original victim txs: {total_original}")
    
    refined_dataset = []
    total_refined = 0
    
    contract_cache = {}
    
    for i, item in enumerate(dataset):
        new_victims = []
        for v in item.get('victim_txs', []):
            if v['value_eth'] < 0.01:
                continue
                
            tx_hash = v['hash']
            
            # Retry loop for rate limits
            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    time.sleep(0.35) # Global rate limiter (max 3 req/s)
                    from_addr = get_tx_from_address(tx_hash)
                    if not from_addr:
                        success = True # Skip silently
                        break 
                        
                    if from_addr not in contract_cache:
                        time.sleep(0.35) # Global rate limiter
                        contract_cache[from_addr] = is_contract(from_addr)
                        
                    if contract_cache[from_addr]:
                        success = True # It's a contract, skip
                        break 
                        
                    new_victims.append(v)
                    success = True
                    break 
                except Exception as e:
                    time.sleep(1.0 * (attempt + 1))
                    
            if not success:
                print(f"Failed {tx_hash} completely after retries.")
                    
        if new_victims:
            new_item = item.copy()
            new_item['victim_txs'] = new_victims
            new_item['victim_tx_count'] = len(new_victims)
            refined_dataset.append(new_item)
            total_refined += len(new_victims)
            
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(dataset)} addresses. Valid victims so far: {total_refined}")
            
    with open(OUT_FILE, 'w') as f:
        json.dump(refined_dataset, f, indent=2)
        
    print("\n========================================")
    print("REFINEMENT COMPLETE")
    print(f"Total victim txs before : {total_original}")
    print(f"Total victim txs after  : {total_refined}")
    print(f"Saved to {OUT_FILE}")
    print("========================================")

if __name__ == '__main__':
    main()
