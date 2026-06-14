import sys, pickle, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model architecture from step15
class SimpleAttentionHead(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.V = nn.Linear(d, h)
        self.w = nn.Linear(h, 1, bias=False)

    def forward(self, x):
        scores = self.w(torch.tanh(self.V(x))).squeeze(-1)
        attn   = F.softmax(scores, dim=-1)
        mean_p = x.mean(dim=1)
        max_p  = x.max(dim=1).values
        attn_p = (attn.unsqueeze(-1) * x).sum(dim=1)
        return torch.cat([mean_p, max_p, attn_p], dim=-1), attn

class MILHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SimpleAttentionHead()
        self.mlp = nn.Sequential(
            nn.Linear(192, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    def forward(self, h):
        pooled, attn = self.attention(h)
        return torch.sigmoid(self.mlp(pooled).squeeze(-1)), attn

class CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(68, 64),
            nn.LayerNorm(64),
        )
        self.mil_head = MILHead()

    def forward(self, hand_crafted, bert_embed):
        x = torch.cat([hand_crafted, bert_embed], dim=-1)
        h = self.feature_proj(x)
        p, attn = self.mil_head(h)
        return p, attn

print("Loading model...")
model = CheckpointModel()
ckpt = torch.load(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\checkpoints\tmil_eth_final.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.to(DEVICE)
model.eval()

print("Loading data...")
with open(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results\step02_features.pkl', 'rb') as f:
    records = pickle.load(f)

phishers = [r for r in records if r['label'] == 1 and r['n_windows'] >= 1]

# Split by size
size_bins = {
    '<1k': [],
    '1k-10k': [],
    '10k-50k': [],
    '>50k': []
}

for r in phishers:
    n = r['n_tx']
    if n < 1000: size_bins['<1k'].append(r)
    elif n < 10000: size_bins['1k-10k'].append(r)
    elif n < 50000: size_bins['10k-50k'].append(r)
    else: size_bins['>50k'].append(r)

@torch.no_grad()
def eval_acc(recs, mode):
    # mode: 'full', 'bert_only', 'hc_only'
    hits = 0
    total = 0
    for r in recs:
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
        
        # Account level prediction: if any window > 0.5
        if p.max().item() > 0.5:
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0

print("\nRunning Ablation...")
print(f"{'Account Size':<12} | {'Count':<6} | {'Full (BERT+HC)':<15} | {'BERT Only':<12} | {'HC Only':<12} | {'Gap (Full - BERT)':<18}")
print("-" * 80)

for bin_name in ['<1k', '1k-10k', '10k-50k', '>50k']:
    recs = size_bins[bin_name]
    if len(recs) == 0: continue
    
    acc_full = eval_acc(recs, 'full')
    acc_bert = eval_acc(recs, 'bert_only')
    acc_hc   = eval_acc(recs, 'hc_only')
    gap      = acc_full - acc_bert
    
    print(f"{bin_name:<12} | {len(recs):<6} | {acc_full*100:>13.1f}% | {acc_bert*100:>10.1f}% | {acc_hc*100:>10.1f}% | {gap*100:>16.1f}%")
