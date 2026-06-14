import urllib.request
import json
import csv
import ssl
from pathlib import Path

# Fix SSL context for python urllib
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

CEX_FILE = Path("results/cex_address_sources.csv")
ADDED_COUNT = 0

def add_address(address, label, source):
    global ADDED_COUNT
    address = address.lower().strip()
    # verify format
    if not address.startswith("0x") or len(address) != 42:
        return
    if address in existing:
        return
    existing.add(address)
    writer.writerow([address, label, source])
    ADDED_COUNT += 1

print("Loading existing CEX addresses...")
existing = set()
with open(CEX_FILE, "r", encoding="utf-8") as f:
    r = csv.reader(f)
    next(r) # skip header
    for row in r:
        existing.add(row[0].lower().strip())

print(f"Already have {len(existing)} addresses.")

# Append mode
out_file = open(CEX_FILE, "a", encoding="utf-8", newline="")
writer = csv.writer(out_file)

# Source 1: merklescience/ethereum-exchange-addresses
url1 = "https://raw.githubusercontent.com/merklescience/ethereum-exchange-addresses/master/exchange_addresses.csv"
print(f"Fetching from {url1}...")
try:
    req = urllib.request.Request(url1, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response:
        content = response.read().decode('utf-8').splitlines()
        r = csv.reader(content)
        header = next(r)
        for row in r:
            if len(row) >= 2:
                # Typically format: address,name,type
                addr = row[0]
                name = row[1]
                add_address(addr, name, "MerkleScience GitHub")
except Exception as e:
    print(f"Failed Source 1: {e}")

# Source 2: dawsbot/eth-labels (etherscan labels)
url2 = "https://raw.githubusercontent.com/dawsbot/eth-labels/main/data/labels.json"
print(f"Fetching from {url2}...")
try:
    req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode('utf-8'))
        # format: list of dicts {"address": "0x...", "name": "Binance 14", "labels": ["Exchange"]}
        for item in data:
            addr = item.get("address", "")
            name = item.get("name", "")
            labels = item.get("labels", [])
            # Only add if it looks like an exchange or mixer
            lbl_str = " ".join(labels).lower()
            name_lower = name.lower()
            if "exchange" in lbl_str or "mixer" in lbl_str or "binance" in name_lower or "huobi" in name_lower or "okx" in name_lower or "coinbase" in name_lower or "kraken" in name_lower:
                add_address(addr, name, "dawsbot/eth-labels")
except Exception as e:
    print(f"Failed Source 2: {e}")

out_file.close()
print(f"Done! Added {ADDED_COUNT} NEW exchange addresses.")
