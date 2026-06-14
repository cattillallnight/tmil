import json
from pathlib import Path
import sys
import os

sys.path.append(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth")
from step26_train_random_ablation import load_transactions, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT

TC_HITS_FILE = RESULTS_DIR / 'step16_etherscan_tc_hits.json'
with open(TC_HITS_FILE, 'r') as f:
    tc_hits = json.load(f)

tc_gt = {}
for addr, txs in tc_hits.items():
    tc_gt[addr.lower()] = set(tx['hash'].lower() for tx in txs)

target_accounts = set(tc_gt.keys())
print(f"Target accounts: {len(target_accounts)}")
tx_history = load_transactions(PHISHER_TX_IN, PHISHER_TX_OUT, target_accounts)
print(f"tx_history keys: {len(tx_history)}")

