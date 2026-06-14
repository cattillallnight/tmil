import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

tc_file = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step16_etherscan_tc_hits.json')
if tc_file.exists():
    with open(tc_file) as f:
        data = json.load(f)
    total_accts = len(data)
    total_txs = sum(len(v) for v in data.values())
    print(f'TC hits so far: {total_accts} accounts | {total_txs} cashout txs')
    for addr, txs in list(data.items())[:3]:
        val = int(txs[0]['value']) / 1e18
        to_addr = txs[0]['to']
        print(f'  {addr[:22]}... -> {len(txs)} TC txs, sample={val:.3f} ETH to {to_addr[:16]}...')
else:
    print('TC file not yet created (no hits found yet or crawler still scanning)')
