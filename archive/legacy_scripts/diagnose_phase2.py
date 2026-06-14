import sys, requests, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

ETHERSCAN_API = 'QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV'

with open(r'C:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\scamsniffer_date_map.json') as f:
    date_map = json.load(f)

print(f'Loaded {len(date_map)} addresses from date map')
print('Testing first 5 addresses...\n')

test_addrs = list(date_map.items())[:5]
for addr, date in test_addrs:
    print(f'Address: {addr}')
    print(f'First seen: {date}')
    resp = requests.get('https://api.etherscan.io/api', params={
        'module': 'account', 'action': 'txlist',
        'address': addr, 'startblock': 0, 'endblock': 99999999,
        'sort': 'asc', 'apikey': ETHERSCAN_API
    }, timeout=10)
    data = resp.json()
    status = data.get('status', 'N/A')
    message = data.get('message', 'N/A')
    result = data.get('result', [])
    print(f'  API status={status}, message={message}')
    if isinstance(result, list):
        print(f'  Total txs: {len(result)}')
        in_txs = [t for t in result if t.get('to','').lower() == addr.lower() and int(t.get('value',0)) > 0]
        print(f'  IN txs with value: {len(in_txs)}')
    else:
        print(f'  Result (non-list): {str(result)[:100]}')
    print()
    time.sleep(0.3)
