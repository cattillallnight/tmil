import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

chk = Path(r'C:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\scamsniffer_dataset_checkpoint.json')
if chk.exists() and chk.stat().st_size > 10:
    with open(chk, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with_victims = [d for d in data if d.get('victim_tx_count', 0) > 0]
    total_victims = sum(d.get('victim_tx_count', 0) for d in data)
    print(f'Processed: {len(data)}/808 addresses')
    print(f'With victim txs: {len(with_victims)}')
    print(f'Total victim txs: {total_victims}')
    if with_victims:
        s = with_victims[0]
        addr = s['address']
        date = s['first_seen_date']
        n_txs = s['n_txs']
        n_vict = s['victim_tx_count']
        print(f'Sample: {addr} | date: {date} | {n_txs} txs | {n_vict} victims')
        if s['victim_txs']:
            v = s['victim_txs'][0]
            h = v['hash']
            eth = v['value_eth']
            hrs = v['hours_from_detection']
            print(f'  Victim tx: {h} | {eth:.4f} ETH | {hrs:.1f}h from detection')
else:
    print('Checkpoint not found yet or still empty...')
