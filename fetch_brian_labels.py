import urllib.request
import csv
import ssl
from pathlib import Path

# Fix SSL context for python urllib
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

CEX_FILE = Path("results/cex_address_sources.csv")
ADDED_COUNT = 0

existing = set()
if CEX_FILE.exists():
    with open(CEX_FILE, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            next(r) # skip header
        except:
            pass
        for row in r:
            if row:
                existing.add(row[0].lower().strip())

print(f"Already have {len(existing)} addresses.")

out_file = open(CEX_FILE, "a", encoding="utf-8", newline="")
writer = csv.writer(out_file)

base_url = "https://raw.githubusercontent.com/brianleect/etherscan-labels/main/data/etherscan/accounts/"
targets = [
    "binance.csv", "huobi.csv", "okx.csv", "kraken.csv", "exchange.csv", 
    "tornado-cash.csv", "bitfinex.csv", "coinbase.csv", "kucoin.csv", 
    "ftx.csv", "gate-io.csv", "crypto-com.csv", "gemini.csv", "bybit.csv",
    "upbit.csv", "poloniex.csv", "bittrex.csv", "blockfi.csv", "celsius.csv",
    "hot-wallet.csv"
]

for target in targets:
    url = base_url + target
    print(f"Fetching {target}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            content = response.read().decode('utf-8').splitlines()
            r = csv.reader(content)
            # Find the address column. Usually: Address,Name Tag,Balance
            headers = next(r, [])
            if not headers: continue
            
            addr_idx = 0
            name_idx = 1
            for i, h in enumerate(headers):
                if h.lower() == "address": addr_idx = i
                if h.lower() == "name tag" or h.lower() == "name": name_idx = i
                
            for row in r:
                if len(row) > max(addr_idx, name_idx):
                    addr = row[addr_idx].lower().strip()
                    name = row[name_idx]
                    
                    if not addr.startswith("0x") or len(addr) != 42:
                        continue
                    if addr in existing:
                        continue
                        
                    existing.add(addr)
                    writer.writerow([addr, name, f"brianleect/etherscan-labels/{target}"])
                    ADDED_COUNT += 1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            pass # File doesn't exist, ignore
        else:
            print(f"  Failed {target}: {e}")
    except Exception as e:
        print(f"  Failed {target}: {e}")

out_file.close()
print(f"Done! Added {ADDED_COUNT} NEW exchange addresses.")
