"""
Step 28a: ScamSniffer Pilot - BERT4ETH Extraction
=================================================
Extracts BERT4ETH embeddings for 5 OOD ScamSniffer accounts.
Uses the pre-trained BERT4ETH TF1.x environment.
"""

import sys
import os
import shutil
import json
import subprocess
import numpy as np
from pathlib import Path

# Force legacy keras to prevent Keras 3 AttributeError for tf.layers.dense
os.environ["TF_USE_LEGACY_KERAS"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
BERT_DIR = BASE_DIR / "BERT4ETH"
DATA_DIR = BERT_DIR / "Data"
MODEL_DIR = BERT_DIR / "Model"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PILOT_ACCOUNTS = [
    '0x9ce67dc9856c9f887ef5a80ae2178d5903864155', 
    '0x7e4384ad48860ae13107b8c8a2b877191edfe2a6', 
    '0x9307d0730bbe0e2df8f747e3f693772ad83debcb', 
    '0x40881dd5b6482854fc01d010ed99fd346f0608b1', 
    '0xe455395bd3468069e0f506e22e13f61666eba36a'
]

def write_dummy_csv(path):
    with open(path, "w") as f: pass

def run_bert_pipeline():
    print("--- ScamSniffer Pilot: Extracting BERT4ETH Embeddings ---")
    
    print("1. Loading dataset and writing temporary CSVs...")
    dataset_file = RESULTS_DIR / "scamsniffer_txlevel_dataset_refined.json"
    with open(dataset_file, "r") as f:
        data = json.load(f)
        
    pilot_data = [x for x in data if x['address'].lower() in PILOT_ACCOUNTS]
    
    # 15 dummy columns
    # hash(0),nonce,block_hash,block_number(3),tx_index,from(5),to(6),value(7),gas(8),gas_price(9),input,block_timestamp(11),...
    
    in_lines = []
    out_lines = []
    
    for item in pilot_data:
        addr = item['address'].lower()
        for t, dir_feat, v, h in zip(item['timestamps'], item['directions'], item['values_eth'], item['hashes']):
            v_wei = int(v * 1e18)
            # Create a 15-column dummy row
            row = [h, "0", "0x0", "0", "0", "from", "to", str(v_wei), "21000", "1000000000", "0x", str(int(t)), "0", "0", "0"]
            if dir_feat == 1:
                row[5] = "counterparty"
                row[6] = addr
                in_lines.append(",".join(row) + "\n")
            else:
                row[5] = addr
                row[6] = "counterparty"
                out_lines.append(",".join(row) + "\n")
                
    # Backup original files
    print("2. Backing up original BERT4ETH data files...")
    files_to_mock = [
        "normal_eoa_transaction_in_slice_1000K.csv",
        "normal_eoa_transaction_out_slice_1000K.csv",
        "phisher_transaction_in.csv",
        "phisher_transaction_out.csv",
        "dean_trans_in_new.csv",
        "dean_trans_out_new.csv",
        "tornado_trans_in_removed.csv",
        "tornado_trans_out_removed.csv"
    ]
    
    for fname in files_to_mock:
        src = DATA_DIR / fname
        if src.exists():
            shutil.move(src, DATA_DIR / (fname + ".bak"))
            write_dummy_csv(src)
            
    # Write our pilot data to phisher_transaction
    with open(DATA_DIR / "phisher_transaction_in.csv", "w") as f:
        f.writelines(in_lines)
    with open(DATA_DIR / "phisher_transaction_out.csv", "w") as f:
        f.writelines(out_lines)
        
    print(f"   Wrote {len(in_lines)} IN and {len(out_lines)} OUT transactions.")
    
    print("3. Executing BERT4ETH Inference Pipeline (TF1.x)...")
    BIZDATE = "scamsniffer_pilot"
    
    os.chdir(BERT_DIR)
    
    # 3.1 gen_seq
    print("   Running gen_seq.py...")
    subprocess.run(["python", "Model/gen_seq.py", f"--bizdate={BIZDATE}", "--phisher=True"], check=True)
    
    # 3.2 gen_pretrain_data
    print("   Running gen_pretrain_data.py...")
    subprocess.run([
        "python", "Model/gen_pretrain_data.py", 
        f"--bizdate={BIZDATE}",
        "--max_seq_length=100",
        "--dupe_factor=1",
        "--masked_lm_prob=0.0"
    ], check=True)
    
    # 3.3 output_embed
    print("   Running output_embed.py...")
    checkpoint_path = "bert4eth_exp/model_104000" 
    subprocess.run([
        "python", "Model/output_embed.py",
        f"--bizdate={BIZDATE}",
        f"--init_checkpoint={checkpoint_path}",
        "--test_input_file=./inter_data/embed.tfrecord",
        "--bert_config_file=Model/bert_config.json",
        "--neg_strategy=zip"
    ], check=True)
    
    print("4. Restoring original BERT4ETH data files...")
    for fname in files_to_mock:
        bak = DATA_DIR / (fname + ".bak")
        src = DATA_DIR / fname
        if bak.exists():
            shutil.move(bak, src)
            
    print("5. Aggregating Embeddings...")
    # The output is saved to inter_data/embedding_bert4eth_104000.npy and address_bert4eth_104000.npy
    embed_file = MODEL_DIR / "inter_data" / "embedding_bert4eth_exp_104000.npy"
    address_file = MODEL_DIR / "inter_data" / "address_bert4eth_exp_104000.npy"
    
    embeddings = np.load(embed_file)
    addresses = np.load(address_file, allow_pickle=True)
    
    print(f"   Loaded embeddings shape: {embeddings.shape}")
    addr2emb = {}
    for i, a in enumerate(addresses):
        a_str = str(a).lower()
        if a_str in PILOT_ACCOUNTS:
            addr2emb[a_str] = embeddings[i]
            
    print(f"   Successfully extracted embeddings for {len(addr2emb)} / 5 pilot accounts.")
    
    out_pkl = RESULTS_DIR / "step28_scamsniffer_bert_embeddings.pkl"
    import pickle
    with open(out_pkl, "wb") as f:
        pickle.dump(addr2emb, f)
    print(f"Saved to {out_pkl}")
    
    os.chdir(BASE_DIR / "tmil_eth")
    print("[OK] Step 28a Complete.")

if __name__ == "__main__":
    run_bert_pipeline()
