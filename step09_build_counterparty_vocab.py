"""
TMIL-ETH — Build Counterparty Vocabulary
========================================
Scans transaction files to find the top N most frequent counterparty addresses.
Saves the vocabulary (Address -> ID) to results/counterparty_vocab.pkl
"""

import sys
import pickle
import pandas as pd
from collections import Counter
from pathlib import Path
from utils import RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT, NORMAL_TX_IN, NORMAL_TX_OUT

VOCAB_SIZE = 50000

def main():
    print("=" * 60)
    print("Building Counterparty Vocabulary")
    print("=" * 60)

    counter = Counter()

    def process_file(path, is_inbound, max_rows=None):
        if not path.exists(): return
        print(f"  Scanning {path.name}...")
        chunk_iter = pd.read_csv(path, chunksize=500000, header=None, low_memory=False)
        rows_processed = 0
        for chunk in chunk_iter:
            # 5 is from_address, 6 is to_address
            # If inbound to account, counterparty is from_address (5)
            # If outbound from account, counterparty is to_address (6)
            cp_col = 5 if is_inbound else 6
            
            chunk = chunk.dropna(subset=[cp_col])
            addrs = chunk[cp_col].astype(str).str.lower()
            
            # Count them
            counter.update(addrs)
            
            rows_processed += len(chunk)
            if max_rows and rows_processed >= max_rows:
                break

    process_file(PHISHER_TX_IN, is_inbound=True)
    process_file(PHISHER_TX_OUT, is_inbound=False)
    # Scan a sample of normal TXs to include major exchanges/contracts
    process_file(NORMAL_TX_IN, is_inbound=True, max_rows=5_000_000)
    process_file(NORMAL_TX_OUT, is_inbound=False, max_rows=5_000_000)

    print(f"\nTotal unique counterparties found: {len(counter):,}")
    
    # Keep top N
    top_n = counter.most_common(VOCAB_SIZE)
    print(f"Most frequent: {top_n[0][0]} ({top_n[0][1]:,} occurrences)")
    print(f"Least frequent in vocab: {top_n[-1][0]} ({top_n[-1][1]:,} occurrences)")

    # Build vocab dict: 0 is PAD/UNK
    vocab = {"[UNK]": 0}
    for i, (addr, count) in enumerate(top_n):
        vocab[addr] = i + 1  # 1-indexed (0 reserved)

    out_path = RESULTS_DIR / "counterparty_vocab.pkl"
    print(f"\nSaving vocabulary of size {len(vocab):,} to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(vocab, f)

    print("\n[OK] Vocab building complete.")

if __name__ == "__main__":
    main()
