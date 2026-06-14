import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
RESULTS_DIR = Path(r"C:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results")

def main():
    with open(RESULTS_DIR / 'step16_etherscan_tc_hits.json', 'r') as f:
        tc_hits = json.load(f)
        
    gt_hashes = set()
    for addr, txs in tc_hits.items():
        for tx in txs:
            gt_hashes.add(tx['hash'].lower())
            
    print(f"Total TC hits from Etherscan: {len(gt_hashes)}")
    
    # Check out.csv
    out_hashes = set()
    with open(DATA_DIR / 'phisher_transaction_out.csv', 'r') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) > 0:
                out_hashes.add(parts[0].lower())
                
    matches = gt_hashes.intersection(out_hashes)
    print(f"TC hits present in phisher_transaction_out.csv: {len(matches)}")

if __name__ == "__main__":
    main()
