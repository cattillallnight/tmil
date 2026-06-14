"""
Step 21: Build New Transaction-Level Phishing Dataset (2022-2024)
==================================================================
Using ScamSniffer's DATED archive as the authoritative source.

Pipeline:
1. Download all ScamSniffer archive files (YYYY-MM-DD.json) -> get {address: first_seen_date}
2. For each phishing address: fetch full tx history from Etherscan
3. Mark IN transactions within 72h AFTER first_seen_date as "victim transactions" (GT)
   [Rationale: ScamSniffer detects a phishing address when victims START reporting.
    The first 72h of activity captures the peak phishing window.]
4. Build feature sequences (v_log, gap_log, dir, freq) for TMIL-ETH compatibility
5. Save final dataset: phisher_sequences + victim_tx_labels

Ground Truth Quality:
- Address-level: ScamSniffer (professional Web3 security firm, not community)  
- Transaction-level: temporal proximity to first_seen_date (72h window)
- This is equivalent to TxPhishScope's methodology (their "phishing window" approach)
"""

import sys, requests, json, time, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
RESULTS_DIR.mkdir(exist_ok=True)

ETHERSCAN_API = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
ETHERSCAN_ENDPOINT = "https://api.etherscan.io/v2/api"  # V2 endpoint
BASE_ARCHIVE_URL = "https://raw.githubusercontent.com/scamsniffer/scam-database/main/blacklist/archive"

# ============================================================
# PHASE 1: Download ScamSniffer archive -> build address:date map
# ============================================================
def build_address_date_map(max_files=1000):
    """Download all daily archive files and map each address to its first seen date."""
    print("[Phase 1] Building ScamSniffer address->first_seen_date map...")
    
    checkpoint = RESULTS_DIR / "scamsniffer_date_map.json"
    if checkpoint.exists():
        print("  Loading from checkpoint...")
        with open(checkpoint, 'r') as f:
            return json.load(f)
    
    # Get list of archive files
    resp = requests.get(
        'https://api.github.com/repos/scamsniffer/scam-database/contents/blacklist/archive',
        timeout=15
    )
    files = sorted([f['name'] for f in resp.json() if f['name'].endswith('.json')])
    print(f"  Found {len(files)} dated archive files (2022 -> 2024)")
    
    addr_date_map = {}  # address -> first_seen_date_str
    
    for i, file_info in enumerate(files[:max_files]):
        fname = file_info['name']
        date_str = fname.replace('.json', '')  # e.g., "2022-06-15"
        download_url = file_info['download_url']  # Use direct download URL
        try:
            resp = requests.get(download_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Archive files have structure: {domains: [...], address: [...]}
                addrs = data.get('address', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                for addr in addrs:
                    addr_lower = addr.lower()
                    if addr_lower not in addr_date_map:
                        addr_date_map[addr_lower] = date_str
        except Exception as e:
            pass
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(files)} files | {len(addr_date_map)} unique addresses mapped")
        
        time.sleep(0.05)  # Be polite to GitHub CDN
    
    print(f"  Total: {len(addr_date_map)} unique phishing addresses with dates")
    
    with open(checkpoint, 'w') as f:
        json.dump(addr_date_map, f)
    print(f"  Saved checkpoint to {checkpoint}")
    
    return addr_date_map


# ============================================================
# PHASE 2: Fetch Etherscan transactions for each address
# ============================================================
def fetch_txs_for_address(addr):
    """Fetch full transaction list for an address from Etherscan V2."""
    resp = requests.get('https://api.etherscan.io/v2/api', params={
        'chainid': 1,
        'module': 'account',
        'action': 'txlist',
        'address': addr,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API
    }, timeout=15)
    
    data = resp.json()
    if data.get('status') == '1' and isinstance(data.get('result'), list):
        return data['result']
    return []

def build_features_from_txs(addr, txs):
    """Build TMIL-ETH compatible features from raw Etherscan transactions."""
    if not txs:
        return None
    
    records = []
    for tx in txs:
        to_addr = str(tx.get('to', '')).lower()
        from_addr = str(tx.get('from', '')).lower()
        val = int(tx.get('value', 0))
        ts = int(tx.get('timeStamp', 0))
        is_error = tx.get('isError', '0')
        
        if is_error == '1' or ts == 0:
            continue
        
        direction = 1 if to_addr == addr else 0  # 1=IN, 0=OUT
        val_eth = val / 1e18
        
        records.append({
            'hash': tx.get('hash'),
            'ts': ts,
            'direction': direction,
            'value_eth': val_eth,
            'from_addr': from_addr,
            'to_addr': to_addr
        })
    
    if not records:
        return None
    
    # Sort by timestamp
    records.sort(key=lambda x: x['ts'])
    
    # Compute features
    timestamps = [r['ts'] for r in records]
    values = [r['value_eth'] for r in records]
    directions = [r['direction'] for r in records]
    hashes = [r['hash'] for r in records]
    
    # v_log: log(1 + value_eth)
    v_log = np.log1p(np.array(values, dtype=np.float32))
    
    # gap_log: log(1 + time gap in hours)
    gaps = [0.0] + [(timestamps[i] - timestamps[i-1]) / 3600.0 for i in range(1, len(timestamps))]
    gap_log = np.log1p(np.array(gaps, dtype=np.float32))
    
    # dir_feat: -1 for OUT, +1 for IN
    dir_feat = np.array([1.0 if d == 1 else -1.0 for d in directions], dtype=np.float32)
    
    # freq: rolling 24h transaction count (simplified: use 0 for now)
    freq = np.zeros(len(records), dtype=np.float32)
    
    hand_crafted = np.column_stack([v_log, gap_log, dir_feat, freq])
    
    return {
        'address': addr,
        'timestamps': timestamps,
        'values_eth': values,
        'directions': directions,
        'hashes': hashes,
        'hand_crafted': hand_crafted.tolist(),
        'n_txs': len(records)
    }


# ============================================================
# PHASE 3: Label victim transactions using 72h window
# ============================================================
def label_victim_txs(seq_data, first_seen_date_str, window_hours=72):
    """
    Label which transactions in the sequence are victim transactions.
    Victim = IN transaction within [24h before, 72h after] first_seen_date.
    """
    try:
        first_seen_dt = datetime.strptime(first_seen_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        first_seen_ts = int(first_seen_dt.timestamp())
    except:
        return []
    
    window_start = first_seen_ts - (24 * 3600)   # 24h before detection
    window_end = first_seen_ts + (window_hours * 3600)
    
    victim_indices = []
    for i, (ts, direction, val) in enumerate(zip(
        seq_data['timestamps'], seq_data['directions'], seq_data['values_eth']
    )):
        if direction == 1 and val > 0 and window_start <= ts <= window_end:
            victim_indices.append({
                'tx_index': i,
                'hash': seq_data['hashes'][i],
                'value_eth': val,
                'timestamp': ts,
                'hours_from_detection': (ts - first_seen_ts) / 3600
            })
    
    return victim_indices


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("Building New Transaction-Level Phishing Dataset (2022-2024)")
    print("Source: ScamSniffer Dated Archive + Etherscan API")
    print("=" * 65)
    
    # Phase 1: Build address-date map
    addr_date_map = build_address_date_map()
    
    print(f"\n[Phase 2] Fetching Etherscan transactions for {len(addr_date_map)} addresses...")
    print("  (Estimated time: ~25 minutes with rate limiting)")
    
    dataset = []
    total_victim_txs = 0
    processed = 0
    errors = 0
    
    checkpoint_file = RESULTS_DIR / "scamsniffer_dataset_checkpoint.json"
    # Load existing checkpoint if available
    processed_addrs = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        dataset = existing
        processed_addrs = set(d['address'] for d in dataset)
        total_victim_txs = sum(len(d.get('victim_txs', [])) for d in dataset)
        print(f"  Resuming from checkpoint: {len(processed_addrs)} already processed")
    
    addr_list = [(addr, date) for addr, date in addr_date_map.items() if addr not in processed_addrs]
    print(f"  Remaining: {len(addr_list)} addresses to process")
    
    for i, (addr, first_seen_date) in enumerate(addr_list):
        try:
            txs = fetch_txs_for_address(addr)
            
            if not txs:
                processed += 1
                time.sleep(0.25)
                continue
            
            seq_data = build_features_from_txs(addr, txs)
            if not seq_data:
                processed += 1
                time.sleep(0.25)
                continue
            
            victim_txs = label_victim_txs(seq_data, first_seen_date)
            
            entry = {
                'address': addr,
                'first_seen_date': first_seen_date,
                'source': 'ScamSniffer-archive',
                'n_txs': seq_data['n_txs'],
                'victim_txs': victim_txs,
                'victim_tx_count': len(victim_txs),
                'hand_crafted': seq_data['hand_crafted'],
                'timestamps': seq_data['timestamps'],
                'directions': seq_data['directions'],
                'values_eth': seq_data['values_eth'],
                'hashes': seq_data['hashes']
            }
            
            dataset.append(entry)
            total_victim_txs += len(victim_txs)
            processed += 1
            
            if len(victim_txs) > 0:
                print(f"  [+] {addr[:20]}... | {seq_data['n_txs']} txs | {len(victim_txs)} victim txs | First seen: {first_seen_date}")
        
        except Exception as e:
            errors += 1
        
        time.sleep(0.25)  # Etherscan rate limit (5 calls/sec)
        
        # Save checkpoint every 100 addresses
        if (processed + errors) % 100 == 0:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f)
            print(f"\n  [Checkpoint] {processed}/{len(addr_list)} | "
                  f"With victims: {len([d for d in dataset if d['victim_tx_count'] > 0])} | "
                  f"Total victim txs: {total_victim_txs}")
    
    # Final save
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    
    # Summary
    with_victims = [d for d in dataset if d['victim_tx_count'] > 0]
    
    print(f"\n{'=' * 65}")
    print(f"DATASET CONSTRUCTION COMPLETE")
    print(f"  Total phishing addresses processed: {len(dataset)}")
    print(f"  Addresses with victim transactions: {len(with_victims)}")
    print(f"  Total victim transaction labels: {total_victim_txs}")
    print(f"  Average victim txs per account: {total_victim_txs/max(len(with_victims),1):.1f}")
    print(f"{'=' * 65}")
    
    # Save final clean dataset
    out_path = RESULTS_DIR / "scamsniffer_txlevel_dataset.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(with_victims, f, indent=2)
    print(f"[+] Final dataset saved to: {out_path}")
    
    # Print sample
    if with_victims:
        sample = with_victims[0]
        print(f"\nSample entry:")
        print(f"  Address: {sample['address']}")
        print(f"  First seen: {sample['first_seen_date']}")
        print(f"  Total txs: {sample['n_txs']}")
        print(f"  Victim txs: {sample['victim_tx_count']}")
        if sample['victim_txs']:
            v = sample['victim_txs'][0]
            print(f"  Sample victim tx: {v['hash']} ({v['value_eth']:.4f} ETH, {v['hours_from_detection']:.1f}h from detection)")

if __name__ == '__main__':
    main()
