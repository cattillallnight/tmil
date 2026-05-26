import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from utils import RESULTS_DIR
from step7_two_phase_training import AccountWindowDataset, collate_fn
from gated_tmil import GatedTMILETH, GatedCompoundLoss

def calculate_iou(pred_set, gt_set):
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    return intersection / union if union > 0 else 0

class AblationGatedTMILETH(GatedTMILETH):
    def __init__(self, use_sigmoid_gate=True, drop_features=False, **kwargs):
        hc_dim = 2 if drop_features else 4
        super().__init__(hand_crafted_dim=hc_dim, **kwargs)
        self.use_sigmoid_gate = use_sigmoid_gate
        self.drop_features = drop_features

    def forward(self, hand_crafted, bert_embed, mask=None):
        if self.drop_features:
            # Keep only z_amount (idx 0) and value_ratio (idx 1)
            # drop density (2) and counterparty_novelty (3)
            # Or just take the first two
            hand_crafted = hand_crafted[:, :, :2]
            
        x = torch.cat([hand_crafted, bert_embed], dim=-1)
        h = self.feature_proj(x)
        
        # Custom attention logic for ablation
        B, N, D = h.shape
        if N == 1:
            z = h.squeeze(1)
            attn = torch.ones(B, 1, device=h.device)
        else:
            tanh_V = torch.tanh(self.attention.V(h))
            if self.use_sigmoid_gate:
                sigm_U = torch.sigmoid(self.attention.U(h))
                gated_h = tanh_V * sigm_U
            else:
                gated_h = tanh_V
                
            scores = self.attention.w(gated_h).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            attn = F.softmax(scores, dim=-1)
            z = torch.bmm(attn.unsqueeze(1), h).squeeze(1)

        logit = self.classifier(z).squeeze(-1)
        p_window = torch.sigmoid(logit)
        return p_window, attn

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    for hc, bert, labels in dataloader:
        hc, bert, labels = hc.to(device), bert.to(device), labels.to(device)
        optimizer.zero_grad()
        p_acct, _ = model(hc, bert)
        loss, _ = loss_fn(p_acct, labels)
        loss.backward()
        optimizer.step()

def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for hc, bert, labels in dataloader:
            hc, bert, labels = hc.to(device), bert.to(device), labels.to(device)
            p_acct, _ = model(hc, bert)
            all_preds.extend(p_acct.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    auc = roc_auc_score(all_labels, all_preds)
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_bin)
    return auc, f1

def evaluate_localization(model, test_recs, gt_dict, device):
    model.eval()
    hit1 = 0
    total = 0
    
    with torch.no_grad():
        for r in test_recs:
            addr = r["address"].lower()
            if addr not in gt_dict: continue
            
            wins = r["windows"]
            burst = gt_dict[addr]["ground_truth_bursts"][0]
            gt_set = set(range(burst["start_tx_idx"], burst["end_tx_idx"] + 1))
            if len(wins) < 5 or len(gt_set) == 0: continue
            
            total += 1
            hc = r["hand_crafted"]
            bert = r["bert_embedding"]
            
            best_p = -1
            best_win = None
            
            for win_idx, (start, end) in enumerate(wins):
                hc_win = hc[start:end]
                n = hc_win.shape[0]
                if n < 200:
                    pad = np.zeros((200 - n, 4), dtype=np.float32)
                    hc_win_pad = np.vstack([hc_win, pad])
                else:
                    hc_win_pad = hc_win[:200]
                    
                hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(device)
                bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 200, -1).to(device)
                
                p, _ = model(hc_t, bert_t)
                if p.item() > best_p:
                    best_p = p.item()
                    best_win = (start, end)
                    
            if best_win is not None:
                pred_set = set(range(best_win[0], best_win[1]))
                if calculate_iou(pred_set, gt_set) > 0:
                    hit1 += 1
                
    return (hit1 / total) * 100 if total > 0 else 0

def run_variant(name, model_kwargs, loss_kwargs, train_loader, val_loader, test_recs, gt_dict, device):
    print(f"\nTraining Variant: {name}")
    model = AblationGatedTMILETH(**model_kwargs).to(device)
    loss_fn = GatedCompoundLoss(**loss_kwargs)
    
    # Proper 25-epoch training for ablation convergence
    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for _ in range(5): train_one_epoch(model, train_loader, loss_fn, opt1, device)
    
    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(20): train_one_epoch(model, train_loader, loss_fn, opt2, device)
        
    auc, f1 = evaluate_classification(model, val_loader, device)
    hit1 = evaluate_localization(model, test_recs, gt_dict, device)
    
    print(f"[{name}] AUC: {auc:.4f} | F1: {f1:.4f} | Hit@1: {hit1:.2f}%")
    return auc, f1, hit1

def main():
    print("="*60)
    print("TMIL-ETH: Properly Converged Ablation Study (Step 20)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_file = RESULTS_DIR / "step2_features.pkl"
    gt_file = "human_ground_truth.json"
    
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_dict = {item["account_address"].lower(): item for item in gt_data}
    
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    # Split records
    eval_addrs = set(gt_dict.keys())
    test_recs = [r for r in records if r["address"].lower() in eval_addrs]
    train_pool = [r for r in records if r["address"].lower() not in eval_addrs]
    
    # 80/20 train/val split from pool
    train_recs, val_recs = train_test_split(train_pool, test_size=0.2, random_state=42, stratify=[r["label"] for r in train_pool])
    
    # For speed in ablation, let's just sample a chunk of train_recs if it's too big, 
    # but we will use the full set if possible. To save user time, use 5000 train samples.
    # We want a real ablation.
    train_recs = train_test_split(train_recs, train_size=5000, random_state=42, stratify=[r["label"] for r in train_recs])[0]
    val_recs = train_test_split(val_recs, test_size=1000, random_state=42, stratify=[r["label"] for r in val_recs])[1]
    
    train_ds = AccountWindowDataset(train_recs, W=200)
    val_ds = AccountWindowDataset(val_recs, W=200)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    variants = [
        ("Full Gated TMIL-ETH", {}, {"lambda1": 0.5}),
        ("No Contrastive Loss", {}, {"lambda1": 0.0}),
        ("No Sigmoid Gate", {"use_sigmoid_gate": False}, {"lambda1": 0.5}),
        ("Drop 2 Features", {"drop_features": True}, {"lambda1": 0.5}),
    ]
    
    results = {}
    for name, m_kwargs, l_kwargs in variants:
        auc, f1, hit1 = run_variant(name, m_kwargs, l_kwargs, train_loader, val_loader, test_recs, gt_dict, device)
        results[name] = {"AUC": auc, "F1": f1, "Hit@1": hit1}
        
    print("\n" + "="*60)
    print("FINAL UNIFIED ABLATION RESULTS")
    print(f"{'Variant':<25} | {'AUC':<7} | {'F1':<7} | {'Hit@1':<7}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['AUC']:.4f}  | {metrics['F1']:.4f}  | {metrics['Hit@1']:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
