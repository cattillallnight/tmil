import pandas as pd
import json

with open(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step16_etherscan_tc_hits.json", 'r') as f:
    tc_gt = json.load(f)
target_accounts = set([x.lower() for x in tc_gt.keys()])

found = 0
for chunk in pd.read_csv(r"c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data\phisher_transaction_in.csv", chunksize=100000, header=None):
    addrs = chunk[6].astype(str).str.lower()
    matches = addrs.isin(target_accounts)
    found += matches.sum()

print(f"Found {found} matching IN transactions.")
