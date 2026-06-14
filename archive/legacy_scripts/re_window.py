import sys
import os
import pickle
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RESULTS_DIR, sliding_windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--S", type=int, required=True)
    args = parser.parse_args()

    features_file = RESULTS_DIR / "step02_features.pkl"
    print(f"Loading features from {features_file}...")
    with open(features_file, "rb") as f:
        records = pickle.load(f)

    print(f"Recomputing windows with W={args.W}, S={args.S}...")
    for rec in records:
        n_tx = rec["n_tx"]
        wins = sliding_windows(n_tx, W=args.W, S=args.S)
        rec["windows"] = wins
        rec["n_windows"] = len(wins)

    out_path = RESULTS_DIR / f"step02_features_W{args.W}_S{args.S}.pkl"
    print(f"Saving to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print("Done!")

if __name__ == "__main__":
    main()
