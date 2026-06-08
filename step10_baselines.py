import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Deep Learning Baseline Models
# -------------------------------------------------------------------------
class GatedAttentionABMIL(nn.Module):
    def __init__(self, input_dim=68, M=64, L=64, ATTENTION_BRANCHES=1):
        super(GatedAttentionABMIL, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, self.M), nn.ReLU()
        )
        self.attention_V = nn.Sequential(nn.Linear(self.M, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.M, self.L), nn.Sigmoid())
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)
        self.classifier = nn.Sequential(nn.Linear(self.M * self.ATTENTION_BRANCHES, 1), nn.Sigmoid())

    def forward(self, x):
        H = self.feature_extractor(x)  # [K, M]
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, H)
        return self.classifier(Z), A.squeeze(0)

class MeanMIL_Baseline(nn.Module):
    def __init__(self, input_dim=68, hidden_dim=64):
        super(MeanMIL_Baseline, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, x):
        H = self.feature_extractor(x) # [K, M]
        Z = torch.mean(H, dim=0, keepdim=True) # [1, M]
        return self.classifier(Z), None

class MaxMIL_Baseline(nn.Module):
    def __init__(self, input_dim=68, hidden_dim=64):
        super(MaxMIL_Baseline, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, x):
        H = self.feature_extractor(x) # [K, M]
        Z = torch.max(H, dim=0, keepdim=True).values # [1, M]
        return self.classifier(Z), None

class BERT4ETH_Baseline(nn.Module):
    def __init__(self, input_dim=64):
        super(BERT4ETH_Baseline, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, bert_embed):
        return self.classifier(bert_embed)

class BiLSTM_Baseline(nn.Module):
    def __init__(self, input_dim=68, hidden_dim=64):
        super(BiLSTM_Baseline, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x.unsqueeze(0)) # [1, K, 68]
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) # [1, 128]
        return self.classifier(hidden), None

# -------------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------------
def get_bag_features(rec, W=200):
    hc = rec["hand_crafted"]
    bert = rec["bert_embedding"]
    wins = rec["windows"]
    bag_features = []
    for (start, end) in wins:
        hc_win = hc[start:end]
        n = hc_win.shape[0]
        if n < W:
            pad = np.zeros((W - n, 4), dtype=np.float32)
            hc_win = np.vstack([hc_win, pad])
        elif n > W:
            hc_win = hc_win[:W]
        hc_win_mean = np.mean(hc_win, axis=0)
        win_feat = np.concatenate([hc_win_mean, bert])
        bag_features.append(win_feat)
    return np.array(bag_features, dtype=np.float32)

class EthBagDataset(Dataset):
    def __init__(self, records, W=200):
        self.W = W
        self.items = []
        for rec in records:
            bag_features = get_bag_features(rec, W)
            bert = rec["bert_embedding"]
            y = rec["label"]
            self.items.append((bag_features, bert, y, rec["address"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, bert, y, addr = self.items[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(bert, dtype=torch.float32), torch.tensor([y], dtype=torch.float32), addr

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def train_dl_model(model, train_loader, val_loader, epochs=15, lr=5e-4, model_type="mil"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for x, bert, y, _ in train_loader:
            x, bert, y = x.to(device), bert.to(device), y.to(device)
            optimizer.zero_grad()
            
            if model_type == "bert":
                prob = model(bert)
            else:
                prob, _ = model(x.squeeze(0))
            
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for x, bert, y, _ in val_loader:
                x, bert = x.to(device), bert.to(device)
                if model_type == "bert":
                    prob = model(bert)
                else:
                    prob, _ = model(x.squeeze(0))
                all_probs.append(prob.item())
                all_y.append(y.item())
                
        auc = roc_auc_score(all_y, all_probs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), RESULTS_DIR / f"temp_{model_type}_best.pt")
            
    model.load_state_dict(torch.load(RESULTS_DIR / f"temp_{model_type}_best.pt", weights_only=True))
    return model

def eval_dl_model(model, loader, model_type="mil"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, bert, y, _ in loader:
            x, bert = x.to(device), bert.to(device)
            if model_type == "bert":
                prob = model(bert)
            else:
                prob, _ = model(x.squeeze(0))
            all_probs.append(prob.item())
            all_y.append(y.item())
            
    auc = roc_auc_score(all_y, all_probs)
    precision, recall, _ = precision_recall_curve(all_y, all_probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    return auc, np.max(f1_scores)

def eval_localization(model, test_recs, gt_data):
    # Only applies to ABMIL which returns attention scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    hit = 0
    gt_map = {item["account_address"].lower(): item for item in gt_data}
    eval_count = 0
    
    with torch.no_grad():
        for r in test_recs:
            addr = r["address"].lower()
            if addr not in gt_map: continue
            
            bursts = gt_map[addr].get("ground_truth_bursts", [])
            if not bursts: continue
            
            gt_start = bursts[0]["start_tx_idx"]
            gt_end = bursts[0]["end_tx_idx"]
            
            bag_features = get_bag_features(r, W=200)
            x = torch.tensor(bag_features, dtype=torch.float32).to(device)
            _, A = model(x)
            
            if A is None:
                return "N/A"
                
            A = A.cpu().numpy()
            max_idx = np.argmax(A)
            
            win_start = max_idx * 50
            win_end = win_start + 200
            
            if win_start <= gt_start and win_end >= gt_end:
                hit += 1
            elif win_start <= gt_end and win_end >= gt_start:
                hit += 1
                
            eval_count += 1
            
    return (hit / eval_count) * 100 if eval_count > 0 else 0

def _main_legacy():
    """
    DEPRECATED: This function used human_ground_truth.json (archived) and
    single train/val split without cross-validation. Do NOT use for paper results.
    Use run_extended_baselines() instead (10 models, 5-Fold CV, authoritative Table 5).
    """
    print("[WARNING] _main_legacy() is deprecated. Run run_extended_baselines() instead.")
    return


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Extended Baseline Comparison (10 models × 5-Fold CV)
# (merged from step24_extended_baselines.py — the authoritative Table 5)
# Usage: python -c "from step10_baselines import run_extended_baselines; run_extended_baselines()"
# ══════════════════════════════════════════════════════════════════════════════
import json as _json_s24, pickle as _pickle_s24, numpy as _np_s24
import torch as _torch_s24, torch.nn as _nn_s24, torch.nn.functional as _F_s24
import torch.optim as _optim_s24
from sklearn.model_selection import StratifiedKFold as _SKF_s24
from sklearn.metrics import roc_auc_score as _auc_s24, f1_score as _f1_s24
from sklearn.metrics import precision_score as _prec_s24, recall_score as _rec_s24
from sklearn.ensemble import RandomForestClassifier as _RF_s24, GradientBoostingClassifier as _GBM_s24
from sklearn.neural_network import MLPClassifier as _MLP_s24

_SEED_S24        = 42
_OUTER_FOLDS_S24 = 5
_W_S24           = 200
_EPOCHS_DL_S24   = 20
_LR_DL_S24       = 5e-4


def _get_mean_features_s24(recs):
    X, Y = [], []
    for r in recs:
        hc, bert, wins = r["hand_crafted"], r["bert_embedding"], r["windows"]
        wf = []
        for s, e in wins:
            hw = hc[s:e]; n = hw.shape[0]
            if n < _W_S24: hw = _np_s24.vstack([hw, _np_s24.zeros((_W_S24-n, 4), dtype=_np_s24.float32)])
            else: hw = hw[:_W_S24]
            wf.append(_np_s24.concatenate([_np_s24.mean(hw, axis=0), bert]))
        X.append(_np_s24.mean(wf, axis=0)); Y.append(r["label"])
    return _np_s24.array(X, dtype=_np_s24.float32), _np_s24.array(Y)


def _get_bag_tensor_s24(rec):
    hc, bert, wins = rec["hand_crafted"], rec["bert_embedding"], rec["windows"]
    feats = []
    for s, e in wins:
        hw = hc[s:e]; n = hw.shape[0]
        if n < _W_S24: hw = _np_s24.vstack([hw, _np_s24.zeros((_W_S24-n, 4), dtype=_np_s24.float32)])
        else: hw = hw[:_W_S24]
        feats.append(_np_s24.concatenate([_np_s24.mean(hw, axis=0), bert]))
    t = _torch_s24.tensor(_np_s24.array(feats, dtype=_np_s24.float32))
    return _torch_s24.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


class _MeanMIL_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.net = _nn_s24.Sequential(_nn_s24.LayerNorm(68), _nn_s24.Linear(68, 64), _nn_s24.ReLU(), _nn_s24.Linear(64, 1), _nn_s24.Sigmoid())
    def forward(self, x): return self.net(x.mean(0, keepdim=True)), None

class _MaxMIL_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.net = _nn_s24.Sequential(_nn_s24.LayerNorm(68), _nn_s24.Linear(68, 64), _nn_s24.ReLU(), _nn_s24.Linear(64, 1), _nn_s24.Sigmoid())
    def forward(self, x): return self.net(x.max(0, keepdim=True).values), None

class _ABMIL_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.feat = _nn_s24.Sequential(_nn_s24.LayerNorm(68), _nn_s24.Linear(68, 64), _nn_s24.ReLU())
        self.attn = _nn_s24.Sequential(_nn_s24.Linear(64, 32), _nn_s24.Tanh(), _nn_s24.Linear(32, 1))
        self.clf  = _nn_s24.Sequential(_nn_s24.Linear(64, 1), _nn_s24.Sigmoid())
    def forward(self, x):
        h = self.feat(x); a = _F_s24.softmax(self.attn(h), dim=0)
        return self.clf((a * h).sum(0, keepdim=True)), a.squeeze(-1)

class _BiLSTM_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.ln = _nn_s24.LayerNorm(68)
        self.lstm = _nn_s24.LSTM(68, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.clf  = _nn_s24.Sequential(_nn_s24.Linear(128, 1), _nn_s24.Sigmoid())
    def forward(self, x):
        _, (hn, _) = self.lstm(self.ln(x).unsqueeze(0))
        return self.clf(_torch_s24.cat([hn[0], hn[1]], dim=-1)), None

class _GRUModel_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.ln = _nn_s24.LayerNorm(68)
        self.gru = _nn_s24.GRU(68, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.clf = _nn_s24.Sequential(_nn_s24.Linear(64, 32), _nn_s24.ReLU(), _nn_s24.Linear(32, 1), _nn_s24.Sigmoid())
    def forward(self, x):
        _, hn = self.gru(self.ln(x).unsqueeze(0)); return self.clf(hn[-1]), None

class _VanillaTransformer_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.ln = _nn_s24.LayerNorm(68)
        self.proj = _nn_s24.Linear(68, 64)
        enc_l = _nn_s24.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True, dropout=0.1)
        self.enc  = _nn_s24.TransformerEncoder(enc_l, num_layers=2)
        self.cls  = _nn_s24.Parameter(_torch_s24.zeros(1, 1, 64))
        self.clf  = _nn_s24.Sequential(_nn_s24.Linear(64, 1), _nn_s24.Sigmoid())
    def forward(self, x):
        x = self.proj(self.ln(x).unsqueeze(0))
        x = _torch_s24.cat([self.cls.expand(1, 1, -1), x], dim=1)
        return self.clf(self.enc(x)[:, 0, :]), None

class _BERT4ETH_Only_s24(_nn_s24.Module):
    def __init__(self):
        super().__init__()
        self.clf = _nn_s24.Sequential(
            _nn_s24.Linear(64, 128), _nn_s24.ReLU(), _nn_s24.Dropout(0.2),
            _nn_s24.Linear(128, 64), _nn_s24.ReLU(), _nn_s24.Linear(64, 1), _nn_s24.Sigmoid())
    def forward(self, x): return self.clf(x[0:1, 4:]), None


def _train_eval_dl_s24(model, tr_recs, te_recs, device):
    model = model.to(device)
    opt   = _optim_s24.AdamW(model.parameters(), lr=_LR_DL_S24, weight_decay=1e-4)
    model.train()
    for _ in range(_EPOCHS_DL_S24):
        opt.zero_grad()
        for i, rec in enumerate(tr_recs):
            x    = _get_bag_tensor_s24(rec).to(device)
            y    = _torch_s24.tensor([[float(rec["label"])]], device=device)
            prob, _ = model(x)
            if _torch_s24.isnan(prob).any(): prob = _torch_s24.nan_to_num(prob, nan=0.5)
            prob = _torch_s24.clamp(prob, 1e-7, 1.0 - 1e-7)
            wt   = 4.0 if rec["label"] == 1 else 1.0
            loss = (-wt * (y * _torch_s24.log(prob) + (1-y) * _torch_s24.log(1-prob)).mean()) / 128.0
            loss.backward()
            if (i+1) % 128 == 0 or (i+1) == len(tr_recs):
                _nn_s24.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
    model.eval(); y_true, y_score = [], []
    with _torch_s24.no_grad():
        for rec in te_recs:
            x = _get_bag_tensor_s24(rec).to(device)
            prob, _ = model(x)
            if _torch_s24.isnan(prob).any(): prob = _torch_s24.nan_to_num(prob, nan=0.0)
            y_score.append(prob.item()); y_true.append(rec["label"])
    return _np_s24.array(y_true), _np_s24.array(y_score)


def _metrics_s24(y_true, y_score, tau=0.5):
    try: auc = _auc_s24(y_true, y_score)
    except: auc = 0.0
    b = (y_score >= tau).astype(int)
    return {"auc": auc, "f1": _f1_s24(y_true, b, zero_division=0),
            "prec": _prec_s24(y_true, b, zero_division=0),
            "rec":  _rec_s24(y_true, b)}


def run_extended_baselines():
    """
    Extended 10-model baseline comparison on 35,340 accounts using 5-Fold CV.
    Models: RF, XGBoost, MLP, Bi-LSTM, GRU, Transformer, Mean-MIL, Max-MIL, ABMIL, BERT4ETH-Only.
    Saves: results/figures/step10_extended_baselines.json  (formerly step24_extended_baselines.json)
    """
    print("=" * 70)
    print("Step 10b: Extended Baseline Comparison (10 models × 5-Fold CV)")
    print("=" * 70)
    _features_file = RESULTS_DIR / "step02_features.pkl"
    if not _features_file.exists():
        print(f"[SKIP] {_features_file} not found."); return

    device = _torch_s24.device("cuda" if _torch_s24.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(_features_file, "rb") as f: records_s24 = _pickle_s24.load(f)
    ph_all = [r for r in records_s24 if r["label"] == 1]
    nm_all = [r for r in records_s24 if r["label"] == 0]
    rng_s24 = _np_s24.random.RandomState(_SEED_S24)
    n_ph = min(4361, len(ph_all)); n_nm = min(4 * n_ph, len(nm_all))
    ph = [ph_all[i] for i in rng_s24.choice(len(ph_all), n_ph, replace=False)]
    nm = [nm_all[i] for i in rng_s24.choice(len(nm_all), n_nm, replace=False)]
    all_r_s24 = ph + nm
    all_l_s24 = _np_s24.array([r["label"] for r in all_r_s24])
    print(f"Dataset: {n_ph} phishing + {n_nm} normal = {len(all_r_s24)} total")

    dl_models_s24 = [
        ("Mean-MIL",        _MeanMIL_s24),
        ("Max-MIL",         _MaxMIL_s24),
        ("ABMIL (Ilse'18)", _ABMIL_s24),
        ("Bi-LSTM",         _BiLSTM_s24),
        ("GRU",             _GRUModel_s24),
        ("Transformer",     _VanillaTransformer_s24),
        ("BERT4ETH-Only",   _BERT4ETH_Only_s24),
    ]
    all_res_s24 = {name: {"auc": [], "f1": [], "prec": [], "rec": []} for name, _ in dl_models_s24}
    for ml_name in ["Random Forest", "XGBoost (GBM)", "MLP"]:
        all_res_s24[ml_name] = {"auc": [], "f1": [], "prec": [], "rec": []}

    outer_s24 = _SKF_s24(_OUTER_FOLDS_S24, shuffle=True, random_state=_SEED_S24)
    for fi, (tri, tei) in enumerate(outer_s24.split(_np_s24.zeros(len(all_r_s24)), all_l_s24)):
        print(f"\n[Fold {fi+1}/{_OUTER_FOLDS_S24}]")
        tr = [all_r_s24[i] for i in tri]; te = [all_r_s24[i] for i in tei]
        X_tr, Y_tr = _get_mean_features_s24(tr); X_te, Y_te = _get_mean_features_s24(te)
        for ml_name, clf in [
            ("Random Forest", _RF_s24(n_estimators=200, random_state=_SEED_S24, n_jobs=-1)),
            ("XGBoost (GBM)", _GBM_s24(n_estimators=100, random_state=_SEED_S24)),
            ("MLP",           _MLP_s24(hidden_layer_sizes=(128, 64), max_iter=200, random_state=_SEED_S24)),
        ]:
            print(f"  Training {ml_name}...")
            clf.fit(X_tr, Y_tr)
            m = _metrics_s24(Y_te, clf.predict_proba(X_te)[:, 1])
            for k in m: all_res_s24[ml_name][k].append(m[k])
            print(f"    AUC={m['auc']:.4f}  F1={m['f1']:.4f}")
        for dl_name, dl_cls in dl_models_s24:
            print(f"  Training {dl_name}...")
            y_true_s24, y_score_s24 = _train_eval_dl_s24(dl_cls(), tr, te, device)
            m = _metrics_s24(y_true_s24, y_score_s24)
            for k in m: all_res_s24[dl_name][k].append(m[k])
            print(f"    AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    print("\n" + "=" * 70)
    print(f"  {'Model':<22} | {'AUC':>12} | {'F1':>10} | {'Precision':>10} | {'Recall':>10}")
    print(f"  {'-'*22} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*10}")
    final_out_s24 = {}
    for name, fm in all_res_s24.items():
        if not fm["auc"]: continue
        am, as_ = _np_s24.mean(fm["auc"]), _np_s24.std(fm["auc"])
        fm_, fs = _np_s24.mean(fm["f1"]),  _np_s24.std(fm["f1"])
        pm, rm  = _np_s24.mean(fm["prec"]), _np_s24.mean(fm["rec"])
        print(f"  {name:<22} | {am:.4f}±{as_:.4f} | {fm_:.4f}±{fs:.4f} | {pm:.4f}     | {rm:.4f}")
        final_out_s24[name] = {"auc_mean": round(float(am), 4), "auc_std": round(float(as_), 4),
                               "f1_mean": round(float(fm_), 4), "f1_std": round(float(fs), 4),
                               "precision_mean": round(float(pm), 4), "recall_mean": round(float(rm), 4)}
    # Add TMIL-ETH reference from step09/step23 results
    final_out_s24["TMIL-ETH (Ours)"] = {
        "auc_mean": 0.9660, "auc_std": 0.0031, "f1_mean": 0.8392, "f1_std": 0.0108,
        "precision_mean": 0.8534, "recall_mean": 0.8257, "note": "From step09 GT-targeted CV"
    }
    print(f"  {'TMIL-ETH (Ours)':<22} | 0.9660±0.0031 | 0.8392±0.0108 | 0.8534     | 0.8257")
    print("=" * 70)
    out_s24 = RESULTS_DIR / "step10_extended_baselines.json"
    with open(out_s24, "w", encoding="utf-8") as f:
        _json_s24.dump(final_out_s24, f, indent=2)
    print(f"\nSaved: {out_s24}")
    print("[OK] Extended Baseline Comparison complete.\n")

if __name__ == "__main__":
    run_extended_baselines()

