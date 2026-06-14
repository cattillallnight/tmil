"""
Step 17: Case Study Validation (A* Ground Truth)
===================================================
1. Load features for specific high-profile hack addresses (e.g., Bancor Hack).
2. Run TMIL-ETH inference.
3. Map attention scores to transaction indices.
4. Plot the Attention curve over time, highlighting the exact cashout transaction.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\BERT4ETH\Data')

sys.path.append(str(BASE_DIR))
from step05_model_architecture import GatedTMILETH

# Top 4 Phishers (Bancor Hack & others with >10k ETH cashout)
CASE_STUDIES = [
    "0x33ed22f4b6b05f8a5faac4701550d52286bd735a", # Bancor Hacker
    "0xbceaa0040764009fdcff407e82ad1f06465fd2c4", # Bancor Laundry
    "0x69627dad496db160a81bed9e27ce2da67c3242bd", # Unknown Huge Hack
    "0x7df1bd58e8fd49803e43987787adfecb4a0a086c"  # Unknown Huge Hack
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_raw_transactions(address):
    df_in = pd.read_csv(DATA_DIR / "phisher_transaction_in.csv", header=None, dtype=str)
    df_out = pd.read_csv(DATA_DIR / "phisher_transaction_out.csv", header=None, dtype=str)
    
    txs = []
    # IN
    df_a_in = df_in[df_in[6].str.lower() == address]
    for _, row in df_a_in.iterrows():
        val = pd.to_numeric(row[7], errors='coerce') / 1e18
        ts = pd.to_numeric(row[11], errors='coerce')
        if pd.notna(val) and pd.notna(ts):
            txs.append({"type": "IN", "val": val, "ts": ts})
            
    # OUT
    df_a_out = df_out[df_out[5].str.lower() == address]
    for _, row in df_a_out.iterrows():
        val = pd.to_numeric(row[7], errors='coerce') / 1e18
        ts = pd.to_numeric(row[11], errors='coerce')
        if pd.notna(val) and pd.notna(ts):
            txs.append({"type": "OUT", "val": val, "ts": ts})
            
    txs.sort(key=lambda x: x["ts"])
    return txs

def main():
    print("========== Phase 3: TMIL-ETH Case Study Inference ==========")
    
    with open(RESULTS_DIR / "step02_features.pkl", "rb") as f:
        all_recs = pickle.load(f)
        
    model = GatedTMILETH(4, 64).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / "tmil_eth_final.pt", map_location=DEVICE))
    model.eval()
    
    os.makedirs(RESULTS_DIR / "plots", exist_ok=True)
    
    for address in CASE_STUDIES:
        print(f"\nEvaluating: {address}")
        rec = next((r for r in all_recs if r["address"].lower() == address), None)
        if not rec:
            print("  Not found in features.pkl!")
            continue
            
        raw_txs = load_raw_transactions(address)
        if not raw_txs:
            print("  No raw transactions found!")
            continue
            
        # Find Ground Truth Cashout Index
        gt_idx = -1
        max_val = 0
        for i, tx in enumerate(raw_txs):
            if tx["type"] == "OUT" and tx["val"] > max_val:
                max_val = tx["val"]
                gt_idx = i
                
        print(f"  Ground Truth Massive Cashout: {max_val:.2f} ETH at index {gt_idx}")
        
        hc = rec["hand_crafted"]
        bert = rec["bert_embedding"]
        wins = rec["windows"]
        
        total_len = hc.shape[0]
        attention_map = np.zeros(total_len)
        attention_counts = np.zeros(total_len)
        
        for start, end in wins:
            n = end - start
            hw = hc[start:end]
            if n < 200: 
                pad = np.zeros((200 - n, 4), dtype=np.float32)
                hw = np.vstack([hw, pad])
            else: 
                hw = hw[:200]
                
            hc_t = torch.tensor(hw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            be_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 200, -1).to(DEVICE)
            
            with torch.no_grad():
                p, attn = model(hc_t, be_t)
                
            # Aggregate attention
            attn = attn.squeeze(0).cpu().numpy()[:n]
            attention_map[start:end] += attn
            attention_counts[start:end] += 1
            
        # Average overlapping windows
        final_attn = attention_map / np.maximum(attention_counts, 1)
        
        ai_pred_idx = np.argmax(final_attn)
        print(f"  TMIL-ETH Max Attention Index: {ai_pred_idx}")
        print(f"  Distance to GT: {abs(ai_pred_idx - gt_idx)} transactions")
        
        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(range(total_len), final_attn, color='blue', label='TMIL-ETH Attention')
        plt.axvline(x=gt_idx, color='red', linestyle='--', linewidth=2, label=f'GT Hack Cashout ({max_val:.0f} ETH)')
        plt.axvline(x=ai_pred_idx, color='green', linestyle=':', linewidth=2, label=f'AI Peak')
        
        plt.title(f"TMIL-ETH Attention Localization\nCase Study: {address} (Bancor Hack)")
        plt.xlabel("Transaction Sequence Index")
        plt.ylabel("Attention Score")
        plt.legend()
        plt.tight_layout()
        
        plot_path = RESULTS_DIR / "plots" / f"case_study_{address[:8]}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved plot to {plot_path}")

if __name__ == "__main__":
    import os
    main()
