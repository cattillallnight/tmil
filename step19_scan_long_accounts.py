import json
import time
import urllib.request
import urllib.error
import os
import pandas as pd
import numpy as np

API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
BASE_URL = "https://api.etherscan.io/v2/api"

def fetch_transactions(address):
    url = f"{BASE_URL}?chainid=1&module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if data['status'] == '1' and data['message'] == 'OK':
                return data['result']
            else:
                return []
    except Exception as e:
        print(f"Error fetching txs for {address}: {e}")
        return []

def get_484_long_accounts():
    print("Extracting long-lived accounts from CSV...")
    in_csv = r'C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data\phisher_transaction_in.csv'
    out_csv = r'C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data\phisher_transaction_out.csv'
    
    df_in = pd.read_csv(in_csv, header=None)
    df_out = pd.read_csv(out_csv, header=None)

    all_addresses = np.concatenate([df_in[5].values, df_in[6].values, df_out[5].values, df_out[6].values])
    all_addresses = [str(a).lower() for a in all_addresses if str(a).startswith('0x') and len(str(a)) == 42]

    counts = pd.Series(all_addresses).value_counts()
    
    with open(r'C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data\phisher_account.txt') as f:
        phishers = set([line.strip().lower() for line in f])

    phisher_counts = counts[counts.index.isin(phishers)]
    long_phishers = phisher_counts[phisher_counts > 100].index.tolist()
    print(f"Found {len(long_phishers)} long-lived phishers (> 100 txs).")
    return long_phishers

def main():
    output_path = "graph_ground_truth_484_long.json"

    target_accounts = get_484_long_accounts()

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            graph_gt = json.load(f)
    else:
        graph_gt = []

    processed_addresses = {item['account_address'] for item in graph_gt}

    print("Starting Graph-based Taint Analysis on 484 long-lived accounts...")
    
    for count_idx, address in enumerate(target_accounts):
        if address in processed_addresses:
            continue
            
        print(f"\n[{count_idx+1}/484] Processing Phisher: {address}")
        
        # 1. Fetch Hop-1 (Phisher's Timeline)
        txs = fetch_transactions(address)
        time.sleep(0.3)

        if not txs:
            print(f"  -> No transactions found or API limit hit.")
            continue

        valid_txs = [tx for tx in txs if tx.get('isError') == '0']
        if not valid_txs:
            continue

        address_lower = address.lower()
        
        # 2. Find Taint Seed (First major incoming transaction)
        start_idx = 0
        first_in_tx = None
        for i, tx in enumerate(valid_txs):
            if tx['to'].lower() == address_lower and int(tx['value']) > 0:
                start_idx = i
                first_in_tx = tx
                break

        if not first_in_tx:
            print("  -> No incoming victim deposits found.")
            continue
            
        # 3. Find Cash-out point (Max outgoing)
        max_out_val = -1
        max_out_idx = start_idx
        hop2_address = None
        
        for i, tx in enumerate(valid_txs):
            if tx['from'].lower() == address_lower and int(tx['value']) > max_out_val:
                max_out_val = int(tx['value'])
                max_out_idx = i
                hop2_address = tx['to'].lower()

        if max_out_idx <= start_idx:
            max_out_idx = len(valid_txs) - 1
            
        print(f"  [+] Window: Tx {start_idx} -> Tx {max_out_idx}")
        
        # 4. Multi-hop Tracking
        if hop2_address:
            hop2_txs = fetch_transactions(hop2_address)
            time.sleep(0.3)
            
            if hop2_txs:
                hop2_valid = [tx for tx in hop2_txs if tx.get('isError') == '0']
                if len(hop2_valid) > 0:
                    print(f"  [Graph] Layer 2 node active with {len(hop2_valid)} txs.")

        if max_out_idx < start_idx:
            start_idx = 0
            
        # 5. Save the new Ground Truth Window
        new_gt_entry = {
            "account_address": address,
            "total_txs_fetched": len(valid_txs),
            "source_report": "Graph-based Taint Tracking (Victim Seed -> L2 Cashout)",
            "graph_gt_bursts": [
                {
                    "start_tx_idx": start_idx,
                    "end_tx_idx": max_out_idx
                }
            ]
        }
        
        graph_gt.append(new_gt_entry)
        
        # Checkpoint every iteration
        with open(output_path, "w") as f:
            json.dump(graph_gt, f, indent=2)
            
    print(f"\nProcessing complete! Check {output_path}")

if __name__ == "__main__":
    main()
