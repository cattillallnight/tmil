"""
TMIL-ETH — Step 1: Dataset Preparation & Analysis
===================================================
Uses BERT4ETH public benchmark (35,340 accounts, 1:4 phishing:normal ratio).
Runs pilot study on 50 phisher + 50 normal accounts to verify burst signals.

Data sources (all in BERT4ETH/Data/):
  - phisher_account.txt          : 7,068 phisher addresses
  - phisher_transaction_in.csv   : inbound transactions for phishers
  - phisher_transaction_out.csv  : outbound transactions for phishers
  - normal_eoa_transaction_in_slice_1000K.csv
  - normal_eoa_transaction_out_slice_1000K.csv
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "BERT4ETH" / "Data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHISHER_ACCOUNTS_FILE = DATA_DIR / "phisher_account.txt"
PHISHER_TX_IN         = DATA_DIR / "phisher_transaction_in.csv"
PHISHER_TX_OUT        = DATA_DIR / "phisher_transaction_out.csv"
NORMAL_TX_IN          = DATA_DIR / "normal_eoa_transaction_in_slice_1000K.csv"
NORMAL_TX_OUT         = DATA_DIR / "normal_eoa_transaction_out_slice_1000K.csv"

# CSV column indices (0-based, matching gen_seq.py HEADER):
# hash,nonce,block_hash,block_number,tx_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,...
COL_FROM       = 5
COL_TO         = 6
COL_VALUE      = 7   # in Wei (int); convert to ETH / 1e18
COL_TIMESTAMP  = 11


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_phisher_accounts(path: Path):
    with open(path, "r") as f:
        accounts = [line.strip().lower() for line in f if line.strip()]
    return accounts


def load_transactions_for_accounts(tx_in_path: Path, tx_out_path: Path,
                                   account_set: set, sample_limit: int = None):
    """
    Returns dict:  address -> list of (timestamp, value_eth, direction)
    direction: 'IN' or 'OUT'
    Only keeps accounts present in account_set.
    """
    txs = defaultdict(list)
    total_in = total_out = 0

    def _read(path, direction):
        nonlocal total_in, total_out
        with open(path, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 12:
                    continue
                try:
                    from_addr = parts[COL_FROM].strip().lower()
                    to_addr   = parts[COL_TO].strip().lower()
                    value_wei = int(parts[COL_VALUE])
                    ts        = int(parts[COL_TIMESTAMP])
                except (ValueError, IndexError):
                    continue

                value_eth = value_wei / 1e18

                if direction == "OUT":
                    if from_addr in account_set:
                        txs[from_addr].append((ts, value_eth, "OUT", to_addr))
                        total_out += 1
                else:  # IN
                    if to_addr in account_set:
                        txs[to_addr].append((ts, value_eth, "IN", from_addr))
                        total_in += 1

                if sample_limit and (total_in + total_out) >= sample_limit * 500:
                    break

    print(f"  Reading OUT from {path.name}...")
    _read(tx_out_path, "OUT")
    print(f"  Reading IN  from {path_in.name}...")
    _read(tx_in_path, "IN")
    return txs


def load_txs_for_accounts(tx_in_path: Path, tx_out_path: Path, account_set: set,
                           max_rows: int = None):
    """Load in+out transactions for a set of accounts. Returns {addr: [(ts, value_eth, dir, counterpart)]}"""
    txs = defaultdict(list)

    def _read_file(path, direction):
        rows = 0
        with open(path, "r") as f:
            for line in f:
                if max_rows and rows > max_rows:
                    break
                parts = line.rstrip("\n").split(",")
                if len(parts) < 12:
                    continue
                try:
                    from_addr = parts[COL_FROM].strip().lower()
                    to_addr   = parts[COL_TO].strip().lower()
                    value_wei = int(parts[COL_VALUE])
                    ts        = int(parts[COL_TIMESTAMP])
                except (ValueError, IndexError):
                    continue
                value_eth = value_wei / 1e18
                if direction == "OUT" and from_addr in account_set:
                    txs[from_addr].append((ts, value_eth, "OUT", to_addr))
                    rows += 1
                elif direction == "IN" and to_addr in account_set:
                    txs[to_addr].append((ts, value_eth, "IN", from_addr))
                    rows += 1

    print(f"  Reading OUT: {tx_out_path.name}")
    _read_file(tx_out_path, "OUT")
    print(f"  Reading IN:  {tx_in_path.name}")
    _read_file(tx_in_path, "IN")
    return txs


def compute_burst_features(txs_sorted):
    """
    For a sorted list of (ts, value_eth, dir, counterpart):
    Compute per-transaction density and value_ratio.
    density_i  = number of transactions in the 1-hour window centred on tx i
    value_ratio_i = running ratio of cumulative IN / (cumulative OUT + 1e-9) at tx i
    counterparty_novelty_i = 1 if counterpart is seen for first time
    """
    n = len(txs_sorted)
    if n == 0:
        return [], [], [], []

    timestamps  = np.array([t[0] for t in txs_sorted], dtype=np.float64)
    values      = np.array([t[1] for t in txs_sorted], dtype=np.float64)
    directions  = [t[2] for t in txs_sorted]
    counterparts = [t[3] for t in txs_sorted]

    # density_i: number of txs within ±1800s (30-min each side = 1-hour window)
    density = []
    for i in range(n):
        lo = timestamps[i] - 1800
        hi = timestamps[i] + 1800
        cnt = int(np.sum((timestamps >= lo) & (timestamps <= hi)))
        density.append(cnt)

    # value_ratio_i = cumulative_in / (cumulative_out + eps) at position i
    cum_in = cum_out = 0.0
    value_ratio = []
    for i in range(n):
        if directions[i] == "IN":
            cum_in += values[i]
        else:
            cum_out += values[i]
        value_ratio.append(cum_in / (cum_out + 1e-9))

    # counterparty_novelty_i = 1 if counterpart first seen
    seen = set()
    novelty = []
    for cp in counterparts:
        novelty.append(0 if cp in seen else 1)
        seen.add(cp)

    return density, value_ratio, novelty, values.tolist()


def bag_purity_proxy(density, value_ratio):
    """
    bag_purity_proxy = |txs: density_i > p75 AND value_ratio_i > 1| / |total txs|
    """
    n = len(density)
    if n == 0:
        return 0.0
    p75 = np.percentile(density, 75)
    burst = sum(
        1 for d, vr in zip(density, value_ratio)
        if d > p75 and vr > 1.0
    )
    return burst / n


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("TMIL-ETH — Step 1: Dataset Preparation & Analysis")
    print("=" * 60)

    # ── 1. Load phisher accounts ──────────────────────────────────
    phisher_accounts = load_phisher_accounts(PHISHER_ACCOUNTS_FILE)
    n_phisher = len(phisher_accounts)
    print(f"\n[1] Phisher accounts loaded: {n_phisher:,}")
    # Paper: 7,068 phishing. We have 7,067 — likely one removed as invalid.
    # For 1:4 ratio: normal = 4 × 7,068 = 28,272 → total 35,340.

    # ── 2. Sample pilot study accounts ────────────────────────────
    pilot_phisher  = random.sample(phisher_accounts, min(50, n_phisher))
    pilot_phisher_set  = set(pilot_phisher)

    print(f"\n[2] Pilot study: sampled {len(pilot_phisher)} phisher accounts for burst verification")

    # ── 3. Load phisher transactions for pilot ────────────────────
    print("\n[3] Loading phisher transactions for pilot study...")
    pilot_txs_phisher = load_txs_for_accounts(
        PHISHER_TX_IN, PHISHER_TX_OUT,
        pilot_phisher_set
    )
    print(f"    Accounts with transactions: {len(pilot_txs_phisher)}")

    # ── 4. Load normal account names from normal TX OUT ───────────
    # We don't have a separate normal account list — derive from normal_tx_out FROM addresses
    print("\n[4] Sampling normal account addresses from normal TX OUT (scanning first 2M rows)...")
    normal_candidates = set()
    with open(NORMAL_TX_OUT, "r") as f:
        for i, line in enumerate(f):
            if i > 2_000_000:
                break
            parts = line.split(",")
            if len(parts) < 7:
                continue
            addr = parts[COL_FROM].strip().lower()
            if addr and len(addr) == 42 and addr.startswith("0x"):
                normal_candidates.add(addr)

    normal_account_list = list(normal_candidates - set(a.lower() for a in phisher_accounts))
    print(f"    Normal candidate accounts found: {len(normal_account_list):,}")

    pilot_normal = random.sample(normal_account_list, min(50, len(normal_account_list)))
    pilot_normal_set = set(pilot_normal)

    print(f"    Pilot normal accounts sampled: {len(pilot_normal)}")

    # ── 5. Load normal transactions for pilot ─────────────────────
    print("\n[5] Loading normal transactions for pilot (scanning up to 3M rows)...")
    pilot_txs_normal = load_txs_for_accounts(
        NORMAL_TX_IN, NORMAL_TX_OUT,
        pilot_normal_set,
        max_rows=3_000_000
    )
    print(f"    Normal accounts with transactions: {len(pilot_txs_normal)}")

    # ── 6. Compute burst features for pilot ───────────────────────
    print("\n[6] Computing burst features for pilot study...")

    def analyze_group(txs_dict, label):
        records = []
        for addr, txs in txs_dict.items():
            txs_sorted = sorted(txs, key=lambda x: x[0])  # sort by timestamp
            if len(txs_sorted) < 3:
                continue
            density, value_ratio, novelty, values = compute_burst_features(txs_sorted)
            proxy = bag_purity_proxy(density, value_ratio)
            records.append({
                "address": addr,
                "label": label,
                "n_transactions": len(txs_sorted),
                "bag_purity_proxy": proxy,
                "mean_density": float(np.mean(density)) if density else 0,
                "max_value_ratio": float(max(value_ratio)) if value_ratio else 0,
                "counterparty_novelty_rate": float(np.mean(novelty)) if novelty else 0,
                "total_in_eth": float(sum(v for t, v, d, _ in txs_sorted if d == "IN")),
                "total_out_eth": float(sum(v for t, v, d, _ in txs_sorted if d == "OUT")),
            })
        return records

    phisher_records = analyze_group(pilot_txs_phisher, "phishing")
    normal_records  = analyze_group(pilot_txs_normal, "normal")

    all_records = phisher_records + normal_records
    df_pilot = pd.DataFrame(all_records)

    # ── 7. Pilot study statistics ─────────────────────────────────
    print("\n[7] Pilot Study Results:")
    print("-" * 50)

    if len(phisher_records) > 0:
        proxy_phisher = [r["bag_purity_proxy"] for r in phisher_records]
        print(f"\nPhishing accounts (n={len(phisher_records)}):")
        print(f"  bag_purity_proxy min:    {min(proxy_phisher):.4f}")
        print(f"  bag_purity_proxy median: {np.median(proxy_phisher):.4f}  (target >= 0.05)")
        print(f"  bag_purity_proxy 90th:   {np.percentile(proxy_phisher, 90):.4f}")
        meets_target = np.median(proxy_phisher) >= 0.05
        print(f"  [OK] Meets target (median >= 0.05): {meets_target}")

    if len(normal_records) > 0:
        proxy_normal = [r["bag_purity_proxy"] for r in normal_records]
        print(f"\nNormal accounts (n={len(normal_records)}):")
        print(f"  bag_purity_proxy median: {np.median(proxy_normal):.4f}")

    # ── 8. Global dataset statistics ──────────────────────────────
    print("\n[8] Global Dataset Statistics (BERT4ETH benchmark):")
    print(f"  Phishing accounts (ground truth): {n_phisher:,}")
    # Paper uses 1:4 ratio → 7,068 phishing × 4 = 28,272 normal = 35,340 total
    n_normal_paper = 28_272
    n_total_paper  = 35_340
    print(f"  Normal accounts (paper 1:4 ratio): {n_normal_paper:,}")
    print(f"  Total accounts (paper):           {n_total_paper:,}")
    print(f"  Normal candidates found in TX:    {len(normal_candidates):,}")
    print(f"  (After removing phisher overlap): {len(normal_account_list):,}")

    # Sequence length stats from phisher transactions
    phisher_seq_lens = []
    with open(PHISHER_TX_OUT, "r") as f:
        addr_counts = defaultdict(int)
        for line in f:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            addr = parts[COL_FROM].strip().lower()
            if addr:
                addr_counts[addr] += 1
    phisher_seq_lens = list(addr_counts.values())

    if phisher_seq_lens:
        print(f"\nPhisher TX-OUT sequence lengths (per account):")
        print(f"  Median: {np.median(phisher_seq_lens):.0f}")
        print(f"  Mean:   {np.mean(phisher_seq_lens):.1f}")
        print(f"  Max:    {max(phisher_seq_lens)}")
        print(f"  Min:    {min(phisher_seq_lens)}")
        print(f"  Accounts with TX out: {len(phisher_seq_lens)}")

    # ── 9. Save results ───────────────────────────────────────────
    stats = {
        "dataset": "BERT4ETH public benchmark",
        "phishing_accounts_file": n_phisher,
        "phishing_accounts_paper": 7068,
        "normal_accounts_paper": n_normal_paper,
        "total_accounts_paper": n_total_paper,
        "class_ratio_paper": "1:4",
        "normal_candidates_from_tx": len(normal_candidates),
        "pilot_study": {
            "phisher_n": len(phisher_records),
            "normal_n": len(normal_records),
            "phisher_bag_purity_proxy_median": float(np.median(proxy_phisher)) if phisher_records else None,
            "phisher_bag_purity_proxy_min": float(min(proxy_phisher)) if phisher_records else None,
            "phisher_bag_purity_proxy_90th": float(np.percentile(proxy_phisher, 90)) if phisher_records else None,
            "meets_target_median_gte_005": bool(np.median(proxy_phisher) >= 0.05) if phisher_records else None,
        },
        "phisher_seq_len": {
            "median": float(np.median(phisher_seq_lens)) if phisher_seq_lens else None,
            "mean":   float(np.mean(phisher_seq_lens)) if phisher_seq_lens else None,
            "max":    int(max(phisher_seq_lens)) if phisher_seq_lens else None,
            "accounts_with_out_tx": len(phisher_seq_lens),
        }
    }

    stats_path  = RESULTS_DIR / "step01_dataset_stats.json"
    pilot_path  = RESULTS_DIR / "step1_pilot_study.csv"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if len(df_pilot) > 0:
        df_pilot.to_csv(pilot_path, index=False)
        print(f"\nPilot study CSV: {pilot_path}")

    print(f"\nDataset stats: {stats_path}")
    print("\n✓ Step 1 complete.\n")
    return stats


if __name__ == "__main__":
    # Fix unresolved name — path_in used inside _read_file closure
    path_in = PHISHER_TX_IN  # noqa (referenced in load_txs_for_accounts print)
    main()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CEX Address Source Documentation + Symmetric Bias Check
# (formerly step20_cex_source_doc_and_bias_check.py)
# ══════════════════════════════════════════════════════════════════════════════

import requests as _requests_s20
import time as _time_s20
from scipy import stats as _scipy_stats_s20

_RESULTS_DIR_S20 = Path(__file__).parent / "results"
_GT_FILE_S20     = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"
_SHEET_CSV_S20   = _RESULTS_DIR_S20 / "step17_annotation_sheet.csv"
_OUT_BIAS_S20    = _RESULTS_DIR_S20 / "step20_bias_check.json"

_API_KEY_S20  = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
_BASE_URL_S20 = "https://api.etherscan.io/api"

DOCUMENTED_CEX = [
    ("0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be", "Binance: Hot Wallet",       "Etherscan Label Cloud", "etherscan.io/address/0x3f5c..."),
    ("0xd551234ae421e3bcba99a0da6d736074f22192ff", "Binance: Hot Wallet 2",     "Etherscan Label Cloud", "etherscan.io/address/0xd551..."),
    ("0x564286362092d8e7936f0549571a803b203aaced", "Binance: Hot Wallet 3",     "Etherscan Label Cloud", "etherscan.io/address/0x5642..."),
    ("0xbe0eb53f46cd790cd13851d5eff43d12404d33e8", "Binance: Cold Wallet",      "Etherscan Label Cloud", "etherscan.io/address/0xbe0e..."),
    ("0xf977814e90da44bfa03b6295a0616a897441acec", "Binance: Cold Wallet 8",    "Etherscan Label Cloud", "etherscan.io/address/0xf977..."),
    ("0xab5c66752a9e8167967685f1450532fb96d5d24f", "Huobi: Hot Wallet",         "Etherscan Label Cloud", "etherscan.io/address/0xab5c..."),
    ("0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b", "Huobi: Hot Wallet 2",       "Etherscan Label Cloud", "etherscan.io/address/0x6748..."),
    ("0xfdb16996831753d5331ff813c29a93c76834a0ad", "Huobi: Hot Wallet 3",       "Etherscan Label Cloud", "etherscan.io/address/0xfdb1..."),
    ("0x6cc5f688a315f3dc28a7781717a9a798a59fda7b", "OKX: Hot Wallet",           "Etherscan Label Cloud", "etherscan.io/address/0x6cc5..."),
    ("0x236f9f97e0e62388479bf9e5ba4889e46b0273c3", "OKX: 2",                    "Etherscan Label Cloud", "etherscan.io/address/0x236f..."),
    ("0xa090e606e30bd747d4e6245a1517ebe430f0057e", "Coinbase: Hot Wallet",      "Etherscan Label Cloud", "etherscan.io/address/0xa090..."),
    ("0x71660c4005ba85c37ccec55d0c4493e66fe775d3", "Coinbase: Hot Wallet 2",    "Etherscan Label Cloud", "etherscan.io/address/0x7166..."),
    ("0x503828976d22510aad0201ac7ec88293211d23da", "Coinbase: Hot Wallet 3",    "Etherscan Label Cloud", "etherscan.io/address/0x5038..."),
    ("0x0d0707963952f2fba59dd06f2b425ace40b492fe", "Gate.io: Hot Wallet",       "Etherscan Label Cloud", "etherscan.io/address/0x0d07..."),
    ("0x46340b20830761efd32832a74d7169b29feb9758", "Crypto.com",                "Etherscan Label Cloud", "etherscan.io/address/0x4634..."),
    ("0xd24400ae8bfebb18ca49be86258a3c749cf46853", "Gemini: Hot Wallet",        "Etherscan Label Cloud", "etherscan.io/address/0xd244..."),
    ("0x742d35cc6634c0532925a3b844bc454e4438f44e", "Bitfinex: Hot Wallet",      "Etherscan Label Cloud", "etherscan.io/address/0x742d..."),
    ("0xe853c56864a2ebe4576a807d26fdc4a0ada51919", "Kraken: Hot Wallet",        "Etherscan Label Cloud", "etherscan.io/address/0xe853..."),
    ("0x7a250d5630b4cf539739df2c5dacb4c659f2488d", "Uniswap V2: Router",        "Etherscan Verified Contract", "etherscan.io/address/0x7a25..."),
    ("0xe592427a0aece92de3edee1f18e0157c05861564", "Uniswap V3: Router",        "Etherscan Verified Contract", "etherscan.io/address/0xe592..."),
    ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "WETH: Wrapped Ether",       "Etherscan Verified Contract + Uniswap Docs", "etherscan.io/address/0xc02a..."),
    ("0x881d40237659c251811cec9c364ef91dc08d300c", "MetaMask: Swap Router",     "Etherscan Label Cloud", "etherscan.io/address/0x881d..."),
    ("0x910cbd523d972eb0a6f4cae4618ad62622b39dbf", "Tornado Cash: 1 ETH Pool",  "OFAC SDN List 2022-08-08 + Etherscan", "home.treasury.gov/..."),
    ("0xa160cdab225685da1d56aa342ad8841c3b53f291", "Tornado Cash: 10 ETH Pool", "OFAC SDN List 2022-08-08 + Etherscan", "home.treasury.gov/..."),
    ("0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3", "Tornado Cash: 100 ETH Pool","OFAC SDN List 2022-08-08 + Etherscan", "home.treasury.gov/..."),
    ("0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144", "Tornado Cash: 0.1 ETH Pool","OFAC SDN List 2022-08-08 + Etherscan", "home.treasury.gov/..."),
]


def run_cex_source_doc_and_bias_check():
    """
    Part A: Document CEX address sources (generates cex_address_sources.csv).
    Part B: Symmetric bias check for SMALL/MICRO vs PRIMARY accounts.
    Saves: results/figures/step20_bias_check.json
    """
    import pandas as _pd_s20
    import numpy as _np_s20

    print("=" * 70)
    print("Step 1b: CEX Address Source Documentation + Bias Check")
    print("=" * 70)

    # Part A
    print("\n[A] Documenting CEX Address Sources...")
    out_cex_csv = _RESULTS_DIR_S20 / "cex_address_sources.csv"
    records = []
    for addr, label, source, ref in DOCUMENTED_CEX:
        # Quick API verify (optional, may timeout)
        try:
            r = _requests_s20.get(_BASE_URL_S20, params={
                "module": "contract", "action": "getsourcecode",
                "address": addr, "apikey": _API_KEY_S20}, timeout=5)
            data = r.json()
            cname = data["result"][0].get("ContractName", "") if data.get("status") == "1" else ""
            ev = f"Verified Contract: {cname}" if cname else "N/A"
        except Exception:
            ev = "N/A"
        records.append({"address": addr, "label": label, "documented_source": source,
                        "reference": ref, "etherscan_verify": ev,
                        "etherscan_link": f"https://etherscan.io/address/{addr}"})
        _time_s20.sleep(0.22)

    _pd_s20.DataFrame(records).to_csv(out_cex_csv, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_cex_csv} ({len(records)} addresses)")

    # Part B
    if not _SHEET_CSV_S20.exists() or not _GT_FILE_S20.exists():
        print("[SKIP] Annotation sheet or GT file not found for bias check.")
        return
    df_sheet = _pd_s20.read_csv(_SHEET_CSV_S20)
    with open(_GT_FILE_S20, "r") as f:
        import json as _js20; gt_map = {r["account_address"].lower(): r for r in _js20.load(f)}

    df_primary = df_sheet[df_sheet["cashout_value_eth"] >= 0.5].copy()
    df_small   = df_sheet[df_sheet["cashout_value_eth"] <  0.5].copy()

    def _get_features(df_sub):
        tx_c, vic_c = [], []
        for _, row in df_sub.iterrows():
            addr = str(row["account_address"]).lower()
            if addr in gt_map:
                tx_c.append(gt_map[addr].get("total_txs", 0))
                vic_c.append(gt_map[addr].get("active_victims_in_cluster", 0))
        return tx_c, vic_c

    prim_txs, prim_vic = _get_features(df_primary)
    small_txs, small_vic = _get_features(df_small)
    bias_report = {}
    for name, a, b in [("tx_count", prim_txs, small_txs), ("victim_count", prim_vic, small_vic)]:
        if a and b:
            u, p = _scipy_stats_s20.mannwhitneyu(a, b, alternative="two-sided")
            bias_report[f"{name}_pval"] = round(p, 4)
            bias_report[f"{name}_significant"] = bool(p < 0.05)
            sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
            print(f"  {name}: Mann-Whitney p={p:.4f} → {sig}")

    any_sig = any(bias_report.get(k, False) for k in ["tx_count_significant", "victim_count_significant"])
    verdict = "NON_SYMMETRIC_BIAS_LIKELY" if any_sig else "APPROXIMATELY_SYMMETRIC"
    result = {"verdict": verdict, "n_primary": len(df_primary), "n_small": len(df_small), "bias_report": bias_report}
    with open(_OUT_BIAS_S20, "w", encoding="utf-8") as f:
        import json as _j20; _j20.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Verdict: {verdict} | Saved: {_OUT_BIAS_S20}")
    print("[OK] CEX Source Doc + Bias Check complete.\n")
