import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_phisher_accounts, build_tx_sequences,
    DATA_DIR, RESULTS_DIR, PHISHER_TX_IN, PHISHER_TX_OUT
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Configuration ───────────────
DUST_THRESHOLD = 0.001       # 1e15 Wei
GAP_THRESHOLD  = 30 * 24 * 3600  # 30 days in seconds
WINDOW_72H     = 72 * 3600       # 72 hours
WINDOW_7D      = 7 * 24 * 3600   # 7 days
MIN_SENDERS    = 3
MIN_VALUE      = 0.1         # 0.1 ETH

# Dynamic threshold based on fixed denominations (will print distribution)
CONFIDENCE_THRESHOLD = 0.5   # Loosened to 50% to account for 0.1/1/10/100 ETH TC pools

OFFICIAL_TC = {
    "0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144",
    "0x12d66f87a04a9e220c9d45f6d8db75c93964f1f2",
    "0x47ce0c6ed5b0ce3d3a51fdb1d7921825a5dbbab4",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3",
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
}

def load_tc_endpoints():
    tc_endpoints = OFFICIAL_TC.copy()
    p = DATA_DIR / "tornado_trans_in_removed.csv"
    if p.exists():
        import pandas as pd
        df = pd.read_csv(p, header=None, usecols=[6], dtype=str)
        local_tc = set(df[6].dropna().str.lower().tolist())
        tc_endpoints.update(local_tc)
    return tc_endpoints

def merge_clusters(clusters_with_tc):
    """
    Merge clusters that share the same Tornado Cash indices.
    clusters_with_tc: list of dicts with 'c' (the cluster) and 'tc_indices' (set of indices)
    """
    merged = []
    for item in clusters_with_tc:
        cluster = item['c']
        tc_indices = item['tc_indices']
        
        # Check if overlaps with any existing merged cluster
        overlap_found = False
        for m in merged:
            if not tc_indices.isdisjoint(m['tc_indices']):
                # Merge them!
                m['c']['start_ts'] = min(m['c']['start_ts'], cluster['start_ts'])
                m['c']['end_ts'] = max(m['c']['end_ts'], cluster['end_ts'])
                m['c']['value'] += cluster['value']
                m['c']['inbound_indices'].update(cluster['inbound_indices'])
                m['tc_indices'].update(tc_indices)
                overlap_found = True
                break
                
        if not overlap_found:
            # Add as new
            # Deep copy to avoid mutating original
            merged.append({
                'c': {
                    'start_ts': cluster['start_ts'],
                    'end_ts': cluster['end_ts'],
                    'value': cluster['value'],
                    'inbound_indices': set(cluster['inbound_indices'])
                },
                'tc_indices': set(tc_indices)
            })
            
    return merged

def main():
    print("========== TMIL-ETH Step 15: Ground Truth Builder v2 ==========")
    phishers = load_phisher_accounts()
    phisher_set = set(a.lower() for a in phishers)
    print(f"Loaded {len(phisher_set)} Phisher Accounts.")
    
    tc_endpoints = load_tc_endpoints()
    print(f"Loaded {len(tc_endpoints)} TC Endpoints.")
    
    print("\nLoading Transaction Sequences...")
    txs_dict = build_tx_sequences(PHISHER_TX_IN, PHISHER_TX_OUT, phisher_set)
    
    gt_results = {}
    stats = {
        "total_phisher_accounts": len(phisher_set),
        "accounts_with_inbound_cluster": 0,
        "accounts_with_tornado_match_within_7d": 0,
        "high_confidence_windows": 0,
        "medium_confidence_windows": 0,
        "total_GT_windows": 0
    }
    
    all_diffs = [] # To log the distribution
    
    for addr, tx_list in tqdm(txs_dict.items(), desc="Processing Accounts"):
        clusters = []
        current_cluster = None
        
        for i, (ts, value, direction, cp) in enumerate(tx_list):
            if direction == "IN" and value > DUST_THRESHOLD and cp != addr and cp not in tc_endpoints:
                if current_cluster is None:
                    current_cluster = {"start_ts": ts, "end_ts": ts, "value": value, "senders": {cp}, "inbound_indices": {i}}
                else:
                    if ts - current_cluster["end_ts"] > GAP_THRESHOLD:
                        clusters.append(current_cluster)
                        current_cluster = {"start_ts": ts, "end_ts": ts, "value": value, "senders": {cp}, "inbound_indices": {i}}
                    else:
                        current_cluster["end_ts"] = ts
                        current_cluster["value"] += value
                        current_cluster["senders"].add(cp)
                        current_cluster["inbound_indices"].add(i)
                        
        if current_cluster is not None:
            clusters.append(current_cluster)
            
        valid_clusters = [c for c in clusters if len(c["senders"]) >= MIN_SENDERS or c["value"] > MIN_VALUE]
        
        if not valid_clusters:
            continue
            
        stats["accounts_with_inbound_cluster"] += 1
        
        # Find TC indices for each cluster
        clusters_with_tc = []
        for c in valid_clusters:
            cluster_start_ts = c["start_ts"]
            cluster_end_ts = c["end_ts"]
            tc_indices = set()
            for i, (ts, value, direction, cp) in enumerate(tx_list):
                if direction == "OUT" and cp in tc_endpoints:
                    if cluster_start_ts <= ts <= cluster_end_ts + WINDOW_7D:
                        tc_indices.add(i)
            if tc_indices:
                clusters_with_tc.append({'c': c, 'tc_indices': tc_indices})
                
        if not clusters_with_tc:
            continue
            
        # De-duplicate overlapping clusters
        merged_clusters = merge_clusters(clusters_with_tc)
        
        account_matches = []
        for m in merged_clusters:
            c = m['c']
            tc_indices = m['tc_indices']
            inbound_val = c["value"]
            cluster_end_ts = c["end_ts"]
            
            # Recalculate tc_val_72h and tc_val_7d from the deduplicated tc_indices
            tc_val_72h = 0.0
            tc_val_7d = 0.0
            
            for i in tc_indices:
                ts, value, _, _ = tx_list[i]
                tc_val_7d += value
                if ts <= cluster_end_ts + WINDOW_72H:
                    tc_val_72h += value
                    
            start_tx_idx = min(tc_indices)
            end_tx_idx = max(tc_indices)
            inbound_start_idx = min(c["inbound_indices"])
            inbound_end_idx = max(c["inbound_indices"])
            
            # Record distribution diff
            if tc_val_72h > 0:
                diff = abs(tc_val_72h - inbound_val) / inbound_val
                all_diffs.append(diff)
                if diff <= CONFIDENCE_THRESHOLD:
                    confidence = "high"
                    stats["high_confidence_windows"] += 1
                else:
                    confidence = "medium"
                    stats["medium_confidence_windows"] += 1
            else:
                confidence = "medium"
                stats["medium_confidence_windows"] += 1
                
            account_matches.append({
                "confidence": confidence,
                "start_tx_idx": start_tx_idx, # Only Tornado Cash
                "end_tx_idx": end_tx_idx,     # Only Tornado Cash
                "inbound_cluster_range": [inbound_start_idx, inbound_end_idx],
                "inbound_val": inbound_val,
                "tc_val_72h": tc_val_72h,
                "tc_val_7d": tc_val_7d
            })
            stats["total_GT_windows"] += 1
            
        if account_matches:
            stats["accounts_with_tornado_match_within_7d"] += 1
            gt_results[addr] = account_matches
            
    # Save Results
    out_json = RESULTS_DIR / "step15_ground_truth.json"
    with open(out_json, "w") as f:
        json.dump(gt_results, f, indent=2)
        
    out_txt = RESULTS_DIR / "step15_gt_funnel.txt"
    funnel_text = f"""Ground Truth Generation Funnel (v2.0 - Deduplicated)
=====================================
total_phisher_accounts: {stats['total_phisher_accounts']}
accounts_with_inbound_cluster: {stats['accounts_with_inbound_cluster']}
accounts_with_tornado_match_within_7d: {stats['accounts_with_tornado_match_within_7d']}
  high_confidence: {stats['high_confidence_windows']}
  medium_confidence: {stats['medium_confidence_windows']}
total_GT_windows (merged): {stats['total_GT_windows']}

Value Matching Differences (abs(tc_val - in_val) / in_val):
"""
    if all_diffs:
        funnel_text += f"  Min diff: {np.min(all_diffs):.2f}\n"
        funnel_text += f"  Median diff: {np.median(all_diffs):.2f}\n"
        funnel_text += f"  Mean diff: {np.mean(all_diffs):.2f}\n"
        funnel_text += f"  Max diff: {np.max(all_diffs):.2f}\n"
        
    with open(out_txt, "w") as f:
        f.write(funnel_text)
        
    print(funnel_text)
    print(f"\n[OK] Ground truth saved to {out_json}")

if __name__ == "__main__":
    main()
