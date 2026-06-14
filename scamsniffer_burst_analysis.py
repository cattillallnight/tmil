import json
import numpy as np
from pathlib import Path

def analyze_scamsniffer_bursts():
    print("Loading ScamSniffer GT...")
    SS_FILE = Path("results/scamsniffer_txlevel_dataset_refined.json")
    if not SS_FILE.exists():
        SS_FILE = Path("results/scamsniffer_txlevel_dataset.json")
        
    with open(SS_FILE, "r") as f:
        ss_data = json.load(f)

    print(f"Loaded {len(ss_data)} SS accounts.")
    
    burst_lengths = []
    num_victim_txs = []
    zero_value_victim_txs = 0
    total_victim_txs = 0
    
    for rec in ss_data:
        # Full sequence is stored inside the JSON!
        hashes = [h.lower() for h in rec.get('hashes', [])]
        values = rec.get('values_eth', [])
        
        victim_hashes = set(tx['hash'].lower() for tx in rec.get('victim_txs', []))
        if not victim_hashes: continue
        
        # Check value=0 hypothesis
        for tx in rec.get('victim_txs', []):
            total_victim_txs += 1
            if float(tx.get('value', 0)) == 0.0:
                zero_value_victim_txs += 1
                
        # Find indices of victim txs in the sequence
        indices = [i for i, h in enumerate(hashes) if h in victim_hashes]
        if not indices: continue
        
        start_idx = min(indices)
        end_idx = max(indices)
        burst_len = end_idx - start_idx + 1
        
        burst_lengths.append(burst_len)
        num_victim_txs.append(len(indices))

    if not burst_lengths:
        print("No matches found.")
        return

    burst_lengths = np.array(burst_lengths)
    num_victim_txs = np.array(num_victim_txs)
    
    print("\n--- ScamSniffer Burst Analysis ---")
    print(f"Total valid accounts analyzed: {len(burst_lengths)}")
    print(f"Total Victim Txs: {total_victim_txs}")
    print(f"Zero-Value Victim Txs: {zero_value_victim_txs} ({(zero_value_victim_txs/total_victim_txs)*100:.2f}%)")
    
    print(f"\nBurst Length (end_idx - start_idx + 1):")
    print(f"  Mean   : {burst_lengths.mean():.2f}")
    print(f"  Median : {np.median(burst_lengths):.2f}")
    print(f"  Min    : {burst_lengths.min()}")
    print(f"  Max    : {burst_lengths.max()}")
    print(f"  P90    : {np.percentile(burst_lengths, 90):.2f}")
    
    print(f"\nNumber of Actual Victim Txs within Burst:")
    print(f"  Mean   : {num_victim_txs.mean():.2f}")
    print(f"  Median : {np.median(num_victim_txs):.2f}")
    print(f"  Max    : {num_victim_txs.max()}")

    # Histogram buckets
    bins = [1, 5, 20, 50, 100, 200, 500, 1000]
    hist, edges = np.histogram(burst_lengths, bins=bins)
    print("\nHistogram of Burst Lengths:")
    for i in range(len(hist)):
        print(f"  [{edges[i]:>4} - {edges[i+1]:>4}): {hist[i]} accounts")

if __name__ == "__main__":
    analyze_scamsniffer_bursts()
