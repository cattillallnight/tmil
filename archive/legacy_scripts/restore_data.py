import zipfile
from pathlib import Path

DATA_DIR = Path(r"c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data")
DWN_DIR = DATA_DIR / "downloads"

zips = ["phishing.zip", "ens.zip", "tornado.zip", "normal.zip"]
for z in zips:
    z_path = DWN_DIR / z
    if z_path.exists():
        print(f"Extracting {z}...")
        with zipfile.ZipFile(z_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
print("Done!")
