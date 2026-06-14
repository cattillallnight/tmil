"""
Step 19: Free Phishing Transaction-Level Ground Truth Collection
================================================================
Sources (all FREE):
1. ScamSniffer scam-database (GitHub) - Confirmed phishing addresses
2. EtherScamDB/CryptoScamDB (API) - Community-reported phishing addresses
3. Etherscan labeled addresses (API) - Addresses tagged "phishing" by Etherscan

For each confirmed phishing address that overlaps with our dataset:
- ALL inbound ETH transactions = confirmed victim transactions (Ground Truth)
- These have Real Reports backing them (ScamSniffer = professional Web3 security firm)
"""

import sys, requests, json, time
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data')
RESULTS_DIR.mkdir(exist_ok=True)

ETHERSCAN_API = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"

def load_our_phishers():
    df_in = pd.read_csv(DATA_DIR / 'phisher_transaction_in.csv', header=None, usecols=[6], dtype=str)
    df_out = pd.read_csv(DATA_DIR / 'phisher_transaction_out.csv', header=None, usecols=[5], dtype=str)
    return set(df_in[6].dropna().str.lower()) | set(df_out[5].dropna().str.lower())

def source_scamsniffer():
    """ScamSniffer scam-database - free, maintained by professional Web3 security firm."""
    print("[*] Fetching ScamSniffer blacklist...")
    resp = requests.get(
        'https://raw.githubusercontent.com/scamsniffer/scam-database/main/blacklist/address.json',
        timeout=15
    )
    return set(a.lower() for a in resp.json())

def source_cryptoscamdb():
    """CryptoScamDB - free community-maintained database with API."""
    print("[*] Fetching CryptoScamDB addresses...")
    try:
        resp = requests.get('https://api.cryptoscamdb.org/v1/addresses', timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Their API returns {result: {address: [{type, url, ...}]}}
            if 'result' in data:
                return set(a.lower() for a in data['result'].keys())
    except Exception as e:
        print(f"  CryptoScamDB failed: {e}")
    return set()

def source_etherscan_phishing_labels():
    """
    Etherscan labels their own 'phish/hack' accounts.
    We use the 'account' module to check each of our phishers against their label API.
    Note: Etherscan doesn't have a bulk label API on free tier, but we can use
    the 'tokenlist' workaround or simply check account tags.
    """
    # Alternatively, use the known open Etherscan phishing list from GitHub:
    # https://github.com/MrLuit/EtherScamDB
    print("[*] Fetching EtherScamDB addresses...")
    try:
        resp = requests.get(
            'https://raw.githubusercontent.com/MrLuit/EtherScamDB/master/data/addresses.json',
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            return set(a.lower() for a in data)
    except Exception as e:
        print(f"  EtherScamDB failed: {e}")
    return set()

def get_victim_txs(phishing_addr):
    """
    Get all ETH-transfer IN transactions for a confirmed phishing address.
    Since the address is confirmed phishing, ALL inbound transfers = victim transactions.
    """
    resp = requests.get('https://api.etherscan.io/api', params={
        'module': 'account',
        'action': 'txlist', 
        'address': phishing_addr,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API
    }, timeout=15)
    
    data = resp.json()
    if data['status'] != '1' or not isinstance(data['result'], list):
        return []
    
    txs = data['result']
    victim_txs = []
    for idx, tx in enumerate(txs):
        to_addr = str(tx.get('to', '')).lower()
        val = int(tx.get('value', 0))
        is_error = tx.get('isError', '0')
        
        if is_error == '0' and to_addr == phishing_addr and val > 0:
            victim_txs.append({
                'tx_index_in_sequence': idx,
                'hash': tx.get('hash'),
                'from_addr': tx.get('from'),
                'value_eth': val / 1e18,
                'timestamp': int(tx.get('timeStamp', 0)),
                'block_number': tx.get('blockNumber')
            })
    
    return victim_txs

def run():
    print("=" * 60)
    print("Free Phishing Transaction-Level Ground Truth Collection")
    print("=" * 60)
    
    our_phishers = load_our_phishers()
    print(f"[*] Our dataset: {len(our_phishers)} phisher addresses")
    
    # Collect from all free sources
    all_confirmed = set()
    
    ss = source_scamsniffer()
    print(f"    ScamSniffer: {len(ss)} confirmed phishing addresses")
    all_confirmed |= ss
    
    # Skip CryptoScamDB and EtherScamDB for now - focusing on ScamSniffer quality
    # cdb = source_cryptoscamdb()
    # edb = source_etherscan_phishing_labels()
    
    # Find overlap
    overlap = our_phishers & all_confirmed
    print(f"\n[+] Overlap (in our dataset AND confirmed phishing): {len(overlap)} addresses")
    
    if not overlap:
        print("[-] No overlap found. Try broader sources.")
        return
    
    # Collect victim transactions for each overlapping address
    final_gt = []
    total_victim_txs = 0
    
    print(f"\n[*] Fetching victim transactions for {len(overlap)} confirmed phishing addresses...")
    for addr in overlap:
        print(f"  Querying {addr}...")
        victim_txs = get_victim_txs(addr)
        
        if victim_txs:
            entry = {
                'phishing_address': addr,
                'source': 'ScamSniffer-scam-database',
                'report_url': 'https://github.com/scamsniffer/scam-database',
                'victim_txs': victim_txs,
                'victim_tx_count': len(victim_txs)
            }
            final_gt.append(entry)
            total_victim_txs += len(victim_txs)
            print(f"    -> {len(victim_txs)} victim transactions found")
        else:
            print(f"    -> No victim transactions (or address had no inbound txs)")
        
        time.sleep(0.3)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Confirmed Phishing Addresses with Victim Txs: {len(final_gt)}")
    print(f"  Total Transaction-Level Ground Truth Txs: {total_victim_txs}")
    print(f"{'=' * 60}")
    
    # Save
    out_path = RESULTS_DIR / 'transaction_level_gt_scamsniffer.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_gt, f, indent=2)
    print(f"\n[+] Saved to: {out_path}")
    
    # Print summary table
    print("\nDetailed Summary:")
    print(f"{'Address':<45} {'Victim Txs':>10}")
    print("-" * 57)
    for entry in final_gt:
        print(f"{entry['phishing_address']:<45} {entry['victim_tx_count']:>10}")

if __name__ == '__main__':
    run()
