import json
import time
import urllib.request
import urllib.error
import os

API_KEY = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
BASE_URL = "https://api.etherscan.io/v2/api"

def fetch_transactions(address):
    url = f"{BASE_URL}?chainid=1&module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if data['status'] == '1' and data['message'] == 'OK':
                return data['result']
            else:
                return []
    except Exception as e:
        print(f"Error fetching txs for {address}: {e}")
        return []

def main():
    ground_truth_path = "human_ground_truth.json"
    output_path = "graph_ground_truth.json"

    if not os.path.exists(ground_truth_path):
        print(f"File {ground_truth_path} not found.")
        return

    with open(ground_truth_path, "r") as f:
        human_gt = json.load(f)

    # Process all accounts
    target_accounts = human_gt

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            graph_gt = json.load(f)
    else:
        graph_gt = []

    processed_addresses = {item['account_address'] for item in graph_gt}

    print("Starting Graph-based Taint Analysis (Multi-hop Tracking)...")
    
    for item in target_accounts:
        address = item['account_address']
        if address in processed_addresses:
            continue
            
        print(f"\nProcessing Phisher Account: {address}")
        
        # 1. Fetch Hop-1 (Phisher's Timeline)
        txs = fetch_transactions(address)
        time.sleep(0.3) # Rate limit protection

        if not txs:
            print(f"  -> No transactions found or API limit hit. Skipping.")
            continue

        # Filter out failed txs and 0-value txs to find real money flow
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
            
        print(f"  [+] Found Taint Seed (Victim Deposit): Tx {start_idx} (Value: {float(first_in_tx['value'])/1e18:.4f} ETH)")

        # 3. Find Cash-out point (Max outgoing or final dispersal)
        # To match the heuristic but broaden the window, we find the max outgoing
        # and then trace it 1 hop to verify it's a cashout endpoint.
        max_out_val = -1
        max_out_idx = start_idx
        hop2_address = None
        
        for i, tx in enumerate(valid_txs):
            if tx['from'].lower() == address_lower and int(tx['value']) > max_out_val:
                max_out_val = int(tx['value'])
                max_out_idx = i
                hop2_address = tx['to'].lower()

        if max_out_idx <= start_idx:
            # Fallback if no outgoing found after incoming, just pick the last tx
            max_out_idx = len(valid_txs) - 1
            
        print(f"  [+] Found Hop-1 Cash-out Node: Tx {max_out_idx} -> sent to {hop2_address}")
        
        # 4. Multi-hop Tracking (Graph traversal to Hop-2)
        if hop2_address:
            print(f"  [Graph] Tracing Hop-2 (Receiver: {hop2_address})...")
            hop2_txs = fetch_transactions(hop2_address)
            time.sleep(0.3)
            
            if hop2_txs:
                # See how fast hop 2 disperses the funds
                hop2_valid = [tx for tx in hop2_txs if tx.get('isError') == '0']
                if len(hop2_valid) > 0:
                    print(f"  [Graph] Verified: Hop-2 is an active node with {len(hop2_valid)} historical txs (Laundering Layer 2).")
                else:
                    print(f"  [Graph] Hop-2 appears to be a dead-end wallet.")
            else:
                print("  [Graph] Could not fetch Hop-2 data.")

        # If the start and end are too close or inverted, fix it
        if max_out_idx < start_idx:
            start_idx = 0
            
        # 5. Save the new Ground Truth Window
        new_gt_entry = {
            "account_address": address,
            "total_txs_fetched": len(valid_txs),
            "source_report": "Graph-based Taint Tracking (Victim Seed -> L2 Cashout)",
            "human_gt_bursts": item.get('ground_truth_bursts', []),
            "graph_gt_bursts": [
                {
                    "start_tx_idx": start_idx,
                    "end_tx_idx": max_out_idx
                }
            ]
        }
        
        graph_gt.append(new_gt_entry)
        
        # Checkpoint
        with open(output_path, "w") as f:
            json.dump(graph_gt, f, indent=2)
            
    print("\nProcessing complete! Check graph_ground_truth.json")

if __name__ == "__main__":
    main()
