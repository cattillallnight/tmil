import sys, pickle, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# Add parent directory to path to import step05
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GatedTMILETH

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
print("Loading model...")
model = GatedTMILETH(4, 64).to(DEVICE)
ckpt = torch.load(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\checkpoints\tmil_eth_final.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.to(DEVICE)
model.eval()

print("Loading data...")
with open(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step02_features.pkl', 'rb') as f:
    records = pickle.load(f)

# Recreate exact same split as step07
labels_arr = [r["label"] for r in records]
_, val_recs = train_test_split(records, test_size=0.2, stratify=labels_arr, random_state=42)

print(f"Validation set: {len(val_recs)} accounts ({sum(r['label'] for r in val_recs)} phishers)")

@torch.no_grad()
def eval_mode(mode):
    # Evaluates the validation set and returns true labels and predicted probabilities
    y_true = []
    y_prob = []
    
    for r in val_recs:
        hc = np.array(r['hand_crafted'], dtype=np.float32)
        bert = np.array(r['bert_embedding'], dtype=np.float32)
        wins = r['windows']
        
        # Build windows
        win_hc = []
        for s, e in wins:
            hw = hc[s:e]
            if len(hw) < 200: hw = np.vstack([hw, np.zeros((200-len(hw), 4), dtype=np.float32)])
            else: hw = hw[:200]
            win_hc.append(hw)
            
        hc_t = torch.tensor(win_hc, dtype=torch.float32).to(DEVICE) # (N, W, 4)
        bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(len(wins), 200, -1).to(DEVICE) # (N, W, 64)
        
        if mode == 'bert_only':
            hc_t = torch.zeros_like(hc_t)
        elif mode == 'hc_only':
            bert_t = torch.zeros_like(bert_t)
            
        p, _ = model(hc_t, bert_t)
        
        # Account level prediction: max window prob
        prob = p.max().item()
        
        y_true.append(r['label'])
        y_prob.append(prob)
        
    return np.array(y_true), np.array(y_prob)

print("Running Evaluation (Full Model)...")
y_true, y_prob_full = eval_mode('full')

print("Running Evaluation (BERT Only)...")
_, y_prob_bert = eval_mode('bert_only')

print("Running Evaluation (HC Only)...")
_, y_prob_hc = eval_mode('hc_only')

def print_metrics(y_true, y_prob, name):
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Find precision at recall ~ 0.9
    idx = np.where(recall >= 0.9)[0][-1]
    p_at_90 = precision[idx]
    
    print(f"{name:<15} | AUC: {auc:.4f} | F1 (th=0.5): {f1:.4f} | Prec@Recall=0.9: {p_at_90:.4f}")

print("\n--- RESULTS ON HELD-OUT VAL SET ---")
print_metrics(y_true, y_prob_full, "Full (BERT+HC)")
print_metrics(y_true, y_prob_bert, "BERT Only")
print_metrics(y_true, y_prob_hc,   "HC Only")
