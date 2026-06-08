"""
step11_ablation_study.py
========================
Full ablation study over the entire 35,340 accounts (1:4 ratio) using 5-Fold Nested CV.
This script guarantees identical evaluation context to the main results (Context A).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

RESULTS_DIR   = Path(__file__).parent / "results"
FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"

SEED         = 42
OUTER_FOLDS  = 5
W            = 200
EPOCHS_DL    = 20
LR_DL        = 5e-4
BATCH_ACCUM  = 128

def get_bag_tensor(rec, apply_global_norm=False, g_mean=None, g_std=None):
    hc   = np.copy(rec["hand_crafted"])
    if apply_global_norm and g_mean is not None:
        hc = (hc - g_mean) / g_std
        
    bert = rec["bert_embedding"]
    wins = rec["windows"]
    feats = []
    
    for start, end in wins:
        hc_win = hc[start:end]
        n = hc_win.shape[0]
        if n < W:
            pad = np.zeros((W-n, 4), dtype=np.float32)
            hc_win = np.vstack([hc_win, pad])
        else:
            hc_win = hc_win[:W]
        mean_hc = np.mean(hc_win, axis=0)
        feats.append(np.concatenate([mean_hc, bert]))
        
    tensor = torch.tensor(np.array(feats, dtype=np.float32))
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

def get_no_window_tensor(rec, apply_global_norm=False, g_mean=None, g_std=None):
    """Nén tất cả vào 1 window duy nhất (Global mean)"""
    hc   = np.copy(rec["hand_crafted"])
    if apply_global_norm and g_mean is not None:
        hc = (hc - g_mean) / g_std
    bert = rec["bert_embedding"]
    
    if len(hc) > 0:
        mean_hc = np.mean(hc, axis=0)
    else:
        mean_hc = np.zeros(4, dtype=np.float32)
    feat = np.concatenate([mean_hc, bert])
    tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

class AblationGatedAttention(nn.Module):
    def __init__(self, input_dim=68, hidden_dim=64):
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        
    def forward(self, x, single_pooling=False):
        h = self.feature_proj(x)
        tanh_V = torch.tanh(self.V(h))
        sigm_U = torch.sigmoid(self.U(h))
        gated_h = tanh_V * sigm_U
        
        scores = self.w(gated_h).squeeze(-1)
        attn = F.softmax(scores, dim=0)
        
        if single_pooling:
            z = h.mean(0, keepdim=True)
        else:
            z = (attn.unsqueeze(-1) * h).sum(0, keepdim=True)
            
        prob = torch.sigmoid(self.classifier(z))
        p_window = torch.sigmoid(self.classifier(h))
        return prob, p_window

def compute_ablation_loss(p_acct, p_window, label, lambda_consist, lambda_contrast):
    y = torch.tensor([[float(label)]], device=p_acct.device)
    weight = 4.0 if label == 1 else 1.0
    l_bce = - weight * (y * torch.log(p_acct) + (1.0 - y) * torch.log(1.0 - p_acct)).mean()
    
    l_consist = torch.tensor(0.0, device=p_acct.device)
    if label == 1 and lambda_consist > 0:
        if p_window.shape[0] > 1:
            l_consist = torch.var(p_window)
            
    l_contrast = torch.tensor(0.0, device=p_acct.device)
    if label == 1 and lambda_contrast > 0:
        if p_window.shape[0] > 1:
            p_max = p_window.max()
            p_mean = p_window.mean()
            l_contrast = F.relu(0.3 - (p_max - p_mean))
            
    return l_bce + lambda_consist * l_consist + lambda_contrast * l_contrast

def train_eval_variant(variant_name, tr_recs, te_recs, device, g_mean, g_std):
    print(f"  Training {variant_name}...")
    lambda_consist = 0.3
    lambda_contrast = 0.2
    single_pooling = False
    no_sliding_window = False
    apply_global_norm = False
    
    if variant_name == "No L_consistency":
        lambda_consist = 0.0
    elif variant_name == "No L_contrast":
        lambda_contrast = 0.0
    elif variant_name == "BCE only":
        lambda_consist = 0.0
        lambda_contrast = 0.0
    elif variant_name == "Single pooling":
        single_pooling = True
    elif variant_name == "No sliding window":
        no_sliding_window = True
    elif variant_name == "Global normalization":
        apply_global_norm = True

    model = AblationGatedAttention().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR_DL, weight_decay=1e-4)
    
    model.train()
    for ep in range(EPOCHS_DL):
        opt.zero_grad()
        for i, rec in enumerate(tr_recs):
            if no_sliding_window:
                x = get_no_window_tensor(rec, apply_global_norm, g_mean, g_std).to(device)
            else:
                x = get_bag_tensor(rec, apply_global_norm, g_mean, g_std).to(device)
                
            p_acct, p_window = model(x, single_pooling)
            p_acct = torch.clamp(p_acct, 1e-7, 1.0 - 1e-7)
            
            loss = compute_ablation_loss(p_acct, p_window, rec["label"], lambda_consist, lambda_contrast)
            loss = loss / float(BATCH_ACCUM)
            loss.backward()
            
            if (i + 1) % BATCH_ACCUM == 0 or (i + 1) == len(tr_recs):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for rec in te_recs:
            if no_sliding_window:
                x = get_no_window_tensor(rec, apply_global_norm, g_mean, g_std).to(device)
            else:
                x = get_bag_tensor(rec, apply_global_norm, g_mean, g_std).to(device)
            p_acct, _ = model(x, single_pooling)
            if torch.isnan(p_acct).any():
                p_acct = torch.nan_to_num(p_acct, nan=0.0)
            y_score.append(p_acct.item())
            y_true.append(rec["label"])
            
    return np.array(y_true), np.array(y_score)

def metrics_from(y_true, y_score, tau=0.5):
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0
    b = (y_score >= tau).astype(int)
    return {
        "auc":  auc,
        "f1":   f1_score(y_true, b, zero_division=0),
        "prec": precision_score(y_true, b, zero_division=0),
        "rec":  recall_score(y_true, b, zero_division=0),
    }

def main():
    print("=" * 80)
    print("Step 11: FULL ABLATION STUDY (35,340 Accounts - 5-Fold CV)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)

    all_hc = [r["hand_crafted"] for r in records if len(r["hand_crafted"]) > 0]
    if len(all_hc) > 0:
        all_hc_stack = np.vstack(all_hc)
        g_mean = np.mean(all_hc_stack, axis=0)
        g_std  = np.std(all_hc_stack, axis=0) + 1e-8
    else:
        g_mean, g_std = None, None

    ph_all = [r for r in records if r["label"] == 1]
    nm_all = [r for r in records if r["label"] == 0]

    rng   = np.random.RandomState(SEED)
    n_ph  = min(4361, len(ph_all))
    n_nm  = min(4 * n_ph, len(nm_all))
    ph    = [ph_all[i] for i in rng.choice(len(ph_all), n_ph, replace=False)]
    nm    = [nm_all[i] for i in rng.choice(len(nm_all), n_nm, replace=False)]
    all_r = ph + nm
    all_l = np.array([r["label"] for r in all_r])

    print(f"Dataset: {n_ph} phishing + {n_nm} normal = {len(all_r)} total")

    variants = [
        "Full TMIL-ETH",
        "No L_consistency",
        "No L_contrast",
        "BCE only",
        "Single pooling",
        "No sliding window",
        "Global normalization"
    ]

    all_results = {v: {"auc": [], "f1": [], "prec": [], "rec": []} for v in variants}
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)

    for fi, (tri, tei) in enumerate(outer.split(np.zeros(len(all_r)), all_l)):
        print(f"\n[Fold {fi+1}/{OUTER_FOLDS}]")
        tr_recs = [all_r[i] for i in tri]
        te_recs = [all_r[i] for i in tei]

        for variant in variants:
            y_true, y_score = train_eval_variant(variant, tr_recs, te_recs, device, g_mean, g_std)
            m = metrics_from(y_true, y_score)
            for k in m:
                all_results[variant][k].append(m[k])
            print(f"    {variant:<25} | AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    print("\n" + "=" * 80)
    print("  FINAL UNIFIED ABLATION RESULTS (35k Accounts - 5-Fold CV)")
    print("=" * 80)
    print(f"  {'Configuration':<25} | {'AUC':>10} | {'F1':>10} | {'Precision':>10} | {'Recall':>10}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    final_out = {}
    for name in variants:
        rm = all_results[name]
        auc_m  = np.mean(rm["auc"])
        f1_m   = np.mean(rm["f1"])
        prec_m = np.mean(rm["prec"])
        rec_m  = np.mean(rm["rec"])
        print(f"  {name:<25} | {auc_m:.4f}     | {f1_m:.4f}     | {prec_m:.4f}     | {rec_m:.4f}")
        final_out[name] = {
            "AUC": round(float(auc_m), 4),
            "F1":  round(float(f1_m),  4),
            "Precision": round(float(prec_m), 4),
            "Recall":    round(float(rec_m), 4),
        }

    out_path = RESULTS_DIR / "step11_full_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
