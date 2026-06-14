import requests, time, sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_KEY = 'QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV'
ENDPOINT = 'https://api.etherscan.io/v2/api'

FULL_TC = {
    '0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144',  # 0.1 ETH
    '0x12d66f87a04a9e220c9d45f6d8db75c93964f1f2',  # 1 ETH
    '0x47ce0c6ed5b0ce3d3a51fdb1d7921825a5dbbab4',  # 10 ETH
    '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf',  # 100 ETH
    '0xa160cdab225685da1d56aa342ad8841c3b53f291',  # 1000 ETH
    '0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3',  # cDAI
    '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',  # Router
}

df = pd.read_csv(r'c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data\phisher_transaction_out.csv',
                 header=None, usecols=[5], dtype=str, nrows=1000)
addrs = df[5].dropna().str.lower().unique()[:10]

found = False
for addr in addrs:
    resp = requests.get(ENDPOINT, params={
        'chainid': 1, 'module': 'account', 'action': 'txlist',
        'address': addr, 'startblock': 0, 'endblock': 99999999,
        'page': 1, 'offset': 10000, 'sort': 'asc', 'apikey': API_KEY
    }, timeout=10)
    data = resp.json()
    if data.get('status') == '1' and isinstance(data.get('result'), list):
        txs = data['result']
        tc_hits = [tx for tx in txs if tx.get('to', '').lower() in FULL_TC]
        print(f"{addr[:20]}... | Total txs: {len(txs)} | TC hits: {len(tc_hits)}")
        if tc_hits:
            h = tc_hits[0]['hash']
            v = int(tc_hits[0]['value']) / 1e18
            t = tc_hits[0]['to']
            print(f"  Sample hit: hash={h[:24]}... | value={v:.3f} ETH | to={t}")
            found = True
            break
    else:
        print(f"{addr[:20]}... | No txs or error: {data.get('message','')}")
    time.sleep(0.3)

if not found:
    print("No TC hits found in first 10 addresses (they may not use TC).")
print("\nSanity test complete.")
