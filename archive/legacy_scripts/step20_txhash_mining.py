"""
Step 20: Multi-Source Free Phishing TxHash Mining
==================================================
Sources:
1. Reddit (r/CryptoScams, r/ethereum, r/Metamask) - victim self-reports with TxHashes
2. GitHub Issues (eth-phishing-detect, web3-security repos) - structured reports
3. Cross-reference all found TxHashes with our 3921 phisher addresses via Etherscan

This is completely free - Reddit API is open, GitHub API has 60 req/hour unauthenticated.
"""

import sys, requests, re, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
DATA_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data')
RESULTS_DIR.mkdir(exist_ok=True)

ETHERSCAN_API = "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV"
REDDIT_HEADERS = {'User-Agent': 'phishing-research-academic-bot/1.0'}
TX_HASH_PATTERN = re.compile(r'0x[a-fA-F0-9]{64}')

import pandas as pd
def load_our_phishers():
    df_in = pd.read_csv(DATA_DIR / 'phisher_transaction_in.csv', header=None, usecols=[6], dtype=str)
    df_out = pd.read_csv(DATA_DIR / 'phisher_transaction_out.csv', header=None, usecols=[5], dtype=str)
    return set(df_in[6].dropna().str.lower()) | set(df_out[5].dropna().str.lower())

# =============================================================================
# SOURCE 1: Reddit scraping
# =============================================================================
def scrape_reddit():
    print("\n[*] SOURCE 1: Reddit API (free, no key required)")
    found_hashes = {}
    
    subreddits = ['CryptoScams', 'ethereum', 'Metamask', 'ethfinance', 'CryptoCurrency', 'BitcoinBeginners']
    queries = [
        'ethereum phishing scam transaction',
        'lost ETH scammed 0x',
        'ethereum wallet drained scam',
        'fake website ETH stolen transaction'
    ]
    
    for sub in subreddits:
        for query in queries:
            url = f'https://www.reddit.com/r/{sub}/search.json'
            params = {'q': query, 'restrict_sr': 'on', 'sort': 'new', 'limit': 25, 't': 'all'}
            try:
                resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=10)
                if resp.status_code == 200:
                    posts = resp.json().get('data', {}).get('children', [])
                    for p in posts:
                        d = p.get('data', {})
                        text = (d.get('selftext', '') or '') + ' ' + (d.get('title', '') or '')
                        hashes = TX_HASH_PATTERN.findall(text)
                        if hashes:
                            for h in hashes:
                                h_lower = h.lower()
                                if h_lower not in found_hashes:
                                    found_hashes[h_lower] = {
                                        'source': 'reddit',
                                        'subreddit': sub,
                                        'post_id': d.get('id'),
                                        'title': d.get('title', '')[:80],
                                        'url': 'https://reddit.com' + d.get('permalink', ''),
                                        'timestamp': d.get('created_utc')
                                    }
                    print(f"  r/{sub} query '{query[:30]}': {len(posts)} posts, hashes so far: {len(found_hashes)}")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(1.5)  # Be polite to Reddit
    
    print(f"  Total from Reddit: {len(found_hashes)} unique TxHashes")
    return found_hashes

# =============================================================================
# SOURCE 2: GitHub Issues
# =============================================================================
def scrape_github_issues():
    print("\n[*] SOURCE 2: GitHub Issues (free, 60 req/hr unauthenticated)")
    found_hashes = {}
    
    repos = [
        'MetaMask/eth-phishing-detect',
        'nicksavers/cryptofinance',
        'scamsniffer/scam-database'
    ]
    
    for repo in repos:
        url = f'https://api.github.com/repos/{repo}/issues'
        params = {'state': 'open', 'per_page': 100, 'labels': 'phishing'}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                issues = resp.json()
                for issue in issues:
                    text = (issue.get('title', '') or '') + ' ' + (issue.get('body', '') or '')
                    hashes = TX_HASH_PATTERN.findall(text)
                    for h in hashes:
                        h_lower = h.lower()
                        if h_lower not in found_hashes:
                            found_hashes[h_lower] = {
                                'source': 'github_issues',
                                'repo': repo,
                                'issue_number': issue.get('number'),
                                'title': issue.get('title', '')[:80],
                                'url': issue.get('html_url')
                            }
                print(f"  {repo}: {len(issues)} issues, hashes found: {len(found_hashes)}")
        except Exception as e:
            print(f"  Error {repo}: {e}")
        time.sleep(1)
    
    # Also search GitHub for phishing tx reports
    url = 'https://api.github.com/search/issues'
    params = {'q': 'ethereum phishing transaction 0x is:issue', 'per_page': 30}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            for item in items:
                text = (item.get('title', '') or '') + ' ' + (item.get('body', '') or '')
                hashes = TX_HASH_PATTERN.findall(text)
                for h in hashes:
                    h_lower = h.lower()
                    if h_lower not in found_hashes:
                        found_hashes[h_lower] = {
                            'source': 'github_search',
                            'repo': item.get('repository_url', '').split('/')[-1],
                            'url': item.get('html_url'),
                            'title': item.get('title', '')[:80]
                        }
            print(f"  GitHub search: {len(items)} issues, hashes: {len(found_hashes)}")
    except Exception as e:
        print(f"  GitHub search error: {e}")
    
    print(f"  Total from GitHub: {len(found_hashes)} unique TxHashes")
    return found_hashes

# =============================================================================
# CROSS-REFERENCE with our phisher set
# =============================================================================
def cross_reference(all_hashes, our_phishers):
    print(f"\n[*] Cross-referencing {len(all_hashes)} TxHashes with our {len(our_phishers)} phishers...")
    
    confirmed_gt = []
    batch_size = 20
    hashes_list = list(all_hashes.items())
    
    for i in range(0, len(hashes_list), batch_size):
        batch = hashes_list[i:i+batch_size]
        for tx_hash, meta in batch:
            # Look up the transaction
            resp = requests.get('https://api.etherscan.io/api', params={
                'module': 'proxy',
                'action': 'eth_getTransactionByHash',
                'txhash': tx_hash,
                'apikey': ETHERSCAN_API
            }, timeout=10)
            
            try:
                result = resp.json().get('result')
                if result and isinstance(result, dict):
                    to_addr = (result.get('to') or '').lower()
                    from_addr = (result.get('from') or '').lower()
                    value_hex = result.get('value', '0x0')
                    value_eth = int(value_hex, 16) / 1e18
                    
                    if to_addr in our_phishers and value_eth > 0:
                        entry = {
                            'tx_hash': tx_hash,
                            'phishing_address': to_addr,
                            'victim_address': from_addr,
                            'value_eth': value_eth,
                            'block_number': int(result.get('blockNumber', '0x0'), 16),
                            'report_source': meta.get('source'),
                            'report_url': meta.get('url'),
                            'report_title': meta.get('title')
                        }
                        confirmed_gt.append(entry)
                        print(f"  [HIT!] {tx_hash[:20]}... -> {to_addr[:20]}... ({value_eth:.4f} ETH)")
            except Exception as e:
                pass
            
            time.sleep(0.25)
        
        print(f"  Progress: {min(i+batch_size, len(hashes_list))}/{len(hashes_list)} | Confirmed hits: {len(confirmed_gt)}")
    
    return confirmed_gt

def main():
    print("=" * 65)
    print("Free Phishing Transaction-Level Ground Truth Mining")
    print("Sources: Reddit + GitHub Issues + Etherscan cross-reference")
    print("=" * 65)
    
    our_phishers = load_our_phishers()
    print(f"[*] Our dataset: {len(our_phishers)} phisher addresses")
    
    all_hashes = {}
    
    reddit_hashes = scrape_reddit()
    all_hashes.update(reddit_hashes)
    
    github_hashes = scrape_github_issues()
    all_hashes.update(github_hashes)
    
    # Save raw harvested hashes
    raw_path = RESULTS_DIR / 'harvested_txhashes_raw.json'
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(all_hashes, f, indent=2)
    print(f"\n[*] Total unique TxHashes harvested: {len(all_hashes)}")
    print(f"[*] Saved raw to: {raw_path}")
    
    if not all_hashes:
        print("[!] No hashes found. Check internet connection.")
        return
    
    confirmed = cross_reference(all_hashes, our_phishers)
    
    print(f"\n{'=' * 65}")
    print(f"FINAL RESULTS:")
    print(f"  TxHashes Harvested from Reports: {len(all_hashes)}")
    print(f"  Confirmed Ground Truth Txs (in our phisher set): {len(confirmed)}")
    print(f"{'=' * 65}")
    
    out_path = RESULTS_DIR / 'transaction_level_gt_mined.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(confirmed, f, indent=2)
    print(f"[+] Saved confirmed GT to: {out_path}")

if __name__ == '__main__':
    main()
