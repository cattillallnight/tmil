"""
step16_expand_benchmark.py
Mở rộng forensic benchmark từ 100 lên 300-500 accounts.
Cần Etherscan API key. Chạy sau step15.

Usage:
  python step16_expand_benchmark.py --api_key YOUR_KEY --target 400
"""
import sys, json, time, argparse, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RESULTS_DIR = Path(__file__).parent / "results"

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"

def get_txlist(address, api_key):
    params = {
        "chainid": "1",
        "module": "account", "action": "txlist",
        "address": address, "startblock": 0,
        "endblock": 99999999, "sort": "asc",
        "apikey": api_key
    }
    r = requests.get(ETHERSCAN_BASE, params=params, timeout=10)
    data = r.json()
    if data["status"] == "1":
        return data["result"]
    return []

def find_laundering_burst(txlist, address):
    """Find the largest single outgoing transfer as ground truth."""
    outgoing = []
    address_lower = address.lower()
    for i, tx in enumerate(txlist):
        if tx["from"].lower() == address_lower:
            val_eth = int(tx["value"]) / 1e18
            if val_eth > 0.1:  # At least 0.1 ETH
                outgoing.append((i, val_eth, tx["hash"]))
    if not outgoing:
        return None
    # Take top-1 by value
    max_tx = max(outgoing, key=lambda x: x[1])
    idx = max_tx[0]
    return {
        "start_tx_idx": max(0, idx - 1),
        "end_tx_idx": min(idx + 1, len(txlist) - 1),
        "value_eth": max_tx[1],
        "tx_hash": max_tx[2]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--target", type=int, default=400)
    args = parser.parse_args()

    # Load existing GT
    gt_path = Path(__file__).parent / "human_ground_truth.json"
    with open(gt_path) as f:
        existing_gt = json.load(f)
    existing_addrs = {item["account_address"].lower() for item in existing_gt}
    print(f"Existing GT: {len(existing_gt)} accounts")

    # Load phishing addresses from dataset
    feat_path = RESULTS_DIR / "step2_features.pkl"
    import pickle
    with open(feat_path, "rb") as f:
        records = pickle.load(f)
    phish_addrs = [r["address"] for r in records if r["label"] == 1 and r["address"].lower() not in existing_addrs]
    print(f"Candidate phishing addresses: {len(phish_addrs)}")

    new_entries = []
    for addr in phish_addrs:
        if len(existing_gt) + len(new_entries) >= args.target:
            break
        try:
            txlist = get_txlist(addr, args.api_key)
            if len(txlist) < 5:
                continue
            burst = find_laundering_burst(txlist, addr)
            if burst is None:
                continue
            entry = {
                "account_address": addr.lower(),
                "source_report": f"Automated On-Chain Forensic (Max Transfer: {burst['value_eth']:.2f} ETH)",
                "total_txs": len(txlist),
                "ground_truth_bursts": [{
                    "start_tx_idx": burst["start_tx_idx"],
                    "end_tx_idx": burst["end_tx_idx"]
                }],
                "tx_hash": burst["tx_hash"]
            }
            new_entries.append(entry)
            print(f"  [{len(new_entries)}] {addr[:12]}... | {len(txlist)} txs | {burst['value_eth']:.2f} ETH | burst: {burst['start_tx_idx']}-{burst['end_tx_idx']}")
            time.sleep(0.2)  # Rate limit
        except Exception as e:
            print(f"  Error for {addr}: {e}")
            continue

    combined = existing_gt + new_entries
    expanded_path = Path(__file__).parent / "human_ground_truth_expanded.json"
    with open(expanded_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nExpanded GT: {len(combined)} accounts (added {len(new_entries)} new)")
    print(f"Saved to: {expanded_path}")
    print("\nNext steps:")
    print("  1. Replace human_ground_truth.json with human_ground_truth_expanded.json")
    print("  2. Re-run step12_human_eval.py with expanded GT")
    print("  3. Re-run step14_extended_eval.py for CI and Hit@k")

if __name__ == "__main__":
    main()
