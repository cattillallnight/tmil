import json

def main():
    input_file = "graph_ground_truth.json"
    output_file = "filtered_long_accounts.json"
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    filtered = []
    for item in data:
        # Filter condition: Account must have more than 100 transactions
        if item["total_txs_fetched"] > 100:
            filtered.append(item)
            
    # Sort by total_txs descending to see the whales first
    filtered.sort(key=lambda x: x["total_txs_fetched"], reverse=True)
    
    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)
        
    print(f"Filtered {len(filtered)} long-lived accounts ( > 100 txs ).")
    print(f"Saved to {output_file}")
    
    # Print top 5 to show the burst window vs total life
    print("\nTop 5 Longest Accounts Analysis:")
    for i, item in enumerate(filtered[:5]):
        total = item['total_txs_fetched']
        burst = item['graph_gt_bursts'][0]
        start = burst['start_tx_idx']
        end = burst['end_tx_idx']
        burst_len = end - start + 1
        percentage = (burst_len / total) * 100
        print(f"Account {i+1}: {item['account_address']}")
        print(f"  - Total Txs: {total}")
        print(f"  - Burst Window: Tx {start} -> Tx {end} ({burst_len} txs)")
        print(f"  - Burst Ratio: {percentage:.2f}% of account life\n")

if __name__ == "__main__":
    main()
