"""
step24_extended_baselines.py
============================
Comprehensive comparison table with 10 models on the same protocol as step23.
All models evaluated with 5-fold CV → mean ± std.

Models:
  Traditional ML:
    1. Random Forest
    2. XGBoost (GBM)
    3. MLP (2-hidden-layer, on mean-pooled features)
  Sequence Models:
    4. Bi-LSTM
    5. GRU
    6. Vanilla Transformer (no MIL)
  MIL Baselines:
    7. Mean-Pooling MIL
    8. Max-Pooling MIL
    9. ABMIL (Ilse et al. 2018) - attention MIL WITHOUT gating
  Ablation of TMIL-ETH:
    10. BERT4ETH-only (no hand-crafted features, no MIL)
  Our model:
    11. TMIL-ETH (from step23 results)

All use same data: 4361 phishing + normals, 5-fold outer CV, BERT4ETH 68-dim features
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from torch.utils.data import Dataset, DataLoader

RESULTS_DIR   = Path(__file__).parent / "results"
FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"
GT_FILE       = Path(__file__).parent / "ground_truth" / "time_aware_ground_truth.json"

SEED         = 42
OUTER_FOLDS  = 5
W            = 200
EPOCHS_DL    = 20
LR_DL        = 5e-4

# ─── Feature Helpers ─────────────────────────────────────────────────────────
def get_mean_features(recs):
    """Mean-pool all windows → one 68-dim vector per account."""
    X, Y = [], []
    for r in recs:
        hc   = r["hand_crafted"]
        bert = r["bert_embedding"]
        wins = r["windows"]
        win_feats = []
        for start, end in wins:
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < W:
                pad = np.zeros((W-n, 4), dtype=np.float32)
                hc_win = np.vstack([hc_win, pad])
            else:
                hc_win = hc_win[:W]
            mean_hc = np.mean(hc_win, axis=0)
            win_feats.append(np.concatenate([mean_hc, bert]))
        X.append(np.mean(win_feats, axis=0))
        Y.append(r["label"])
    return np.array(X, dtype=np.float32), np.array(Y)

def get_bag_tensor(rec):
    """Return (N_windows, 68) tensor for one account."""
    hc   = rec["hand_crafted"]
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
    return torch.tensor(np.array(feats, dtype=np.float32))

# ─── PyTorch Models ───────────────────────────────────────────────────────────
class MeanMIL(nn.Module):
    def __init__(self, in_dim=68, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())
    def forward(self, x):          # x: (N, 68)
        return self.net(x.mean(0, keepdim=True)), None

class MaxMIL(nn.Module):
    def __init__(self, in_dim=68, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x.max(0, keepdim=True).values), None

class ABMIL(nn.Module):
    """Attention MIL without gating (Ilse 2018)."""
    def __init__(self, in_dim=68, M=64, L=32):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_dim, M), nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(M, L), nn.Tanh(), nn.Linear(L, 1))
        self.clf  = nn.Sequential(nn.Linear(M, 1), nn.Sigmoid())
    def forward(self, x):
        h = self.feat(x)
        a = F.softmax(self.attn(h), dim=0)
        z = (a * h).sum(0, keepdim=True)
        return self.clf(z), a.squeeze(-1)

class BiLSTM(nn.Module):
    def __init__(self, in_dim=68, h=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, h, num_layers=1, batch_first=True, bidirectional=True)
        self.clf  = nn.Sequential(nn.Linear(h*2, 1), nn.Sigmoid())
    def forward(self, x):
        _, (hn, _) = self.lstm(x.unsqueeze(0))
        out = torch.cat([hn[0], hn[1]], dim=-1)
        return self.clf(out), None

class GRUModel(nn.Module):
    def __init__(self, in_dim=68, h=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, h, num_layers=2, batch_first=True, dropout=0.1)
        self.clf = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        _, hn = self.gru(x.unsqueeze(0))
        return self.clf(hn[-1]), None

class VanillaTransformer(nn.Module):
    """Transformer encoder → CLS token → classify."""
    def __init__(self, in_dim=68, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj   = nn.Linear(in_dim, d_model)
        enc_layer   = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.enc    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls    = nn.Parameter(torch.zeros(1, 1, d_model))
        self.clf    = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.proj(x.unsqueeze(0))               # (1, N, d)
        cls = self.cls.expand(1, 1, -1)
        x   = torch.cat([cls, x], dim=1)            # (1, N+1, d)
        out = self.enc(x)
        return self.clf(out[:, 0, :]), None

class BERT4ETH_Only(nn.Module):
    """Classify using ONLY BERT4ETH 64-dim embedding (no hand-crafted features)."""
    def __init__(self):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        # x is (N, 68) → use only last 64 dims (BERT4ETH embedding)
        return self.clf(x[0:1, 4:]), None   # Just use BERT part of first window

# ─── Train/Eval helpers ──────────────────────────────────────────────────────
def train_eval_dl(model, tr_recs, te_recs, device, epochs=EPOCHS_DL, lr=LR_DL):
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        for rec in tr_recs:
            x = get_bag_tensor(rec).to(device)
            y = torch.tensor([[float(rec["label"])]], device=device)
            opt.zero_grad()
            prob, _ = model(x)
            if torch.isnan(prob).any():
                prob = torch.nan_to_num(prob, nan=0.5)
            prob = torch.clamp(prob, 1e-7, 1.0 - 1e-7)
            loss = crit(prob, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for rec in te_recs:
            x = get_bag_tensor(rec).to(device)
            prob, _ = model(x)
            if torch.isnan(prob).any():
                prob = torch.nan_to_num(prob, nan=0.0)
            y_score.append(prob.item())
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

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Step 24: Extended Baseline Comparison (10 models × 5-fold CV)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)

    # Globally normalize the first 4 hand-crafted features to prevent Deep Learning gradients from exploding
    all_hc = []
    for r in records:
        all_hc.append(r["hand_crafted"])
    all_hc = np.vstack(all_hc)
    g_mean = np.mean(all_hc, axis=0)
    g_std  = np.std(all_hc, axis=0) + 1e-8
    for r in records:
        r["hand_crafted"] = (r["hand_crafted"] - g_mean) / g_std

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

    # Model registry: (name, constructor_fn, type)
    # type: "ml" for sklearn, "dl" for pytorch
    dl_models = [
        ("Mean-MIL",        lambda: MeanMIL()),
        ("Max-MIL",         lambda: MaxMIL()),
        ("ABMIL (Ilse'18)", lambda: ABMIL()),
        ("Bi-LSTM",         lambda: BiLSTM()),
        ("GRU",             lambda: GRUModel()),
        ("Transformer",     lambda: VanillaTransformer()),
        ("BERT4ETH-Only",   lambda: BERT4ETH_Only()),
    ]

    # Store fold results
    all_results = {name: {"auc": [], "f1": [], "prec": [], "rec": []} for name, _ in dl_models}
    all_results["Random Forest"]    = {"auc": [], "f1": [], "prec": [], "rec": []}
    all_results["XGBoost (GBM)"]    = {"auc": [], "f1": [], "prec": [], "rec": []}
    all_results["MLP"]              = {"auc": [], "f1": [], "prec": [], "rec": []}

    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)

    for fi, (tri, tei) in enumerate(outer.split(np.zeros(len(all_r)), all_l)):
        print(f"\n[Fold {fi+1}/{OUTER_FOLDS}]")
        tr_recs = [all_r[i] for i in tri]
        te_recs = [all_r[i] for i in tei]

        # ── Traditional ML (needs mean features) ───────────────────────────
        print("  Extracting ML features...")
        X_tr, Y_tr = get_mean_features(tr_recs)
        X_te, Y_te = get_mean_features(te_recs)

        for ml_name, clf in [
            ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
            ("XGBoost (GBM)", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
            ("MLP",           MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED)),
        ]:
            print(f"  Training {ml_name}...")
            clf.fit(X_tr, Y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            m = metrics_from(Y_te, prob)
            for k in m:
                all_results[ml_name][k].append(m[k])
            print(f"    AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

        # ── Deep Learning (bag-level, window-level) ─────────────────────────
        for dl_name, dl_fn in dl_models:
            print(f"  Training {dl_name}...")
            model = dl_fn()
            y_true, y_score = train_eval_dl(model, tr_recs, te_recs, device)
            m = metrics_from(y_true, y_score)
            for k in m:
                all_results[dl_name][k].append(m[k])
            print(f"    AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    # ── Aggregate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE — 10 Baselines + TMIL-ETH (step23)")
    print("=" * 70)
    print(f"  {'Model':<22} | {'AUC':>12} | {'F1':>10} | {'Precision':>10} | {'Recall':>10}")
    print(f"  {'-'*22} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*10}")

    final_out = {}
    for name, fold_metrics in all_results.items():
        if not fold_metrics["auc"]:
            continue
        auc_m  = np.mean(fold_metrics["auc"])
        auc_s  = np.std(fold_metrics["auc"])
        f1_m   = np.mean(fold_metrics["f1"])
        f1_s   = np.std(fold_metrics["f1"])
        prec_m = np.mean(fold_metrics["prec"])
        rec_m  = np.mean(fold_metrics["rec"])
        print(f"  {name:<22} | {auc_m:.4f}±{auc_s:.4f} | {f1_m:.4f}±{f1_s:.4f} | {prec_m:.4f}     | {rec_m:.4f}")
        final_out[name] = {
            "auc_mean": round(auc_m, 4), "auc_std": round(auc_s, 4),
            "f1_mean":  round(f1_m,  4), "f1_std":  round(f1_s, 4),
            "precision_mean": round(prec_m, 4),
            "recall_mean":    round(rec_m, 4),
        }

    # Add TMIL-ETH from step23
    final_out["TMIL-ETH (Ours)"] = {
        "auc_mean": 0.9660, "auc_std": 0.0031,
        "f1_mean":  0.8392, "f1_std":  0.0108,
        "precision_mean": 0.8534,
        "recall_mean":    0.8257,
        "note": "From step23_gt_targeted_cv.json"
    }
    print(f"  {'TMIL-ETH (Ours)':<22} | 0.9660±0.0031 | 0.8392±0.0108 | 0.8534     | 0.8257")
    print("=" * 70)

    out_path = RESULTS_DIR / "step24_extended_baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("[OK] Step 24 complete.\n")

if __name__ == "__main__":
    main()
