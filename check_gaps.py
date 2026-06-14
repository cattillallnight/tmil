import json
from pathlib import Path
from utils import load_phisher_accounts, build_tx_sequences, PHISHER_TX_IN, PHISHER_TX_OUT

OFFICIAL_TC = {
    "0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144",
    "0x12d66f87a04a9e220c9d45f6d8db75c93964f1f2",
    "0x47ce0c6ed5b0ce3d3a51fdb1d7921825a5dbbab4",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3",
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
}

def main():
    phishers = load_phisher_accounts()
    phisher_set = set(a.lower() for a in phishers)
    txs_dict = build_tx_sequences(PHISHER_TX_IN, PHISHER_TX_OUT, phisher_set)

    with open(r"C:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step16_etherscan_tc_hits.json", 'r') as f:
        tc_hits = json.load(f)

    print(f"Total accounts with TC hits from Etherscan: {len(tc_hits)}")
    accounts_with_gap = 0
    accounts_with_no_inbound = 0

    for addr, hits in tc_hits.items():
        if addr not in txs_dict:
            continue
        tx_list = txs_dict[addr]
        
        inbounds = [ts for (ts, val, dir, cp) in tx_list if dir == "IN" and cp != addr and cp not in OFFICIAL_TC]
        outbounds_tc = [ts for (ts, val, dir, cp) in tx_list if dir == "OUT" and cp in OFFICIAL_TC]
        
        if not outbounds_tc:
            continue # TC tx not in CSV
            
        if not inbounds:
            accounts_with_no_inbound += 1
            continue
            
        # Find minimum gap
        min_gap = float('inf')
        for out_ts in outbounds_tc:
            for in_ts in inbounds:
                if out_ts >= in_ts:
                    gap = out_ts - in_ts
                    if gap < min_gap:
                        min_gap = gap
                        
        if min_gap > 7 * 24 * 3600:
            accounts_with_gap += 1
            print(f"Account {addr} has min gap {min_gap / (24*3600):.1f} days")

    print(f"Accounts with no valid inbound: {accounts_with_no_inbound}")
    print(f"Accounts with gap > 7 days: {accounts_with_gap}")

if __name__ == "__main__":
    main()
