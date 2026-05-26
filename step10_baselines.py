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

def main():
    print("=" * 70)
    print("TMIL-ETH: Comprehensive Baseline Comparison (7 Models)")
    print("=" * 70)
    
    feat_path = RESULTS_DIR / "step2_features.pkl"
    if not feat_path.exists():
        print("Features not found. Please run step2_feature_extraction.py first.")
        return
        
    print("Loading features...")
    with open(feat_path, "rb") as f:
        records = pickle.load(f)
        
    gt_path = Path(__file__).parent / "human_ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            gt_data = json.load(f)
    else:
        gt_data = []
        
    eval_addrs = {item["account_address"].lower() for item in gt_data}
    
    # Split Data
    test_recs = [r for r in records if r["address"].lower() in eval_addrs]
    train_pool = [r for r in records if r["address"].lower() not in eval_addrs]
    
    phish_pool = [r for r in train_pool if r["label"] == 1]
    norm_pool = [r for r in train_pool if r["label"] == 0]
    
    np.random.seed(42)
    norm_sample = np.random.choice(norm_pool, size=len(phish_pool)*4, replace=False).tolist()
    train_val_recs = phish_pool + norm_sample
    
    train_recs, val_recs = train_test_split(train_val_recs, test_size=0.2, random_state=42, stratify=[r["label"] for r in train_val_recs])
    
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test (Forensic): {len(test_recs)}\n")
    
    results = {}
    
    # ---------------------------------------------------------
    # A. Traditional ML Baselines
    # ---------------------------------------------------------
    def get_ml_features(recs):
        X, Y = [], []
        dataset = EthBagDataset(recs)
        for x, bert, y, _ in dataset:
            mean_feat = torch.mean(x, dim=0).numpy()
            X.append(mean_feat)
            Y.append(y.item())
        return np.array(X), np.array(Y)
        
    print("--- [1/7] Training Random Forest ---")
    X_train, Y_train = get_ml_features(train_recs)
    X_val, Y_val = get_ml_features(val_recs)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(Y_val, rf_probs)
    rf_f1 = np.max(2 * precision_recall_curve(Y_val, rf_probs)[1] * precision_recall_curve(Y_val, rf_probs)[0] / (precision_recall_curve(Y_val, rf_probs)[1] + precision_recall_curve(Y_val, rf_probs)[0] + 1e-8))
    results["Random Forest"] = {"AUC": rf_auc, "F1": rf_f1, "Hit@1": "N/A"}
    
    print("--- [2/7] Training Gradient Boosting (HistGBM) ---")
    gbm = HistGradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, Y_train)
    gbm_probs = gbm.predict_proba(X_val)[:, 1]
    gbm_auc = roc_auc_score(Y_val, gbm_probs)
    gbm_f1 = np.max(2 * precision_recall_curve(Y_val, gbm_probs)[1] * precision_recall_curve(Y_val, gbm_probs)[0] / (precision_recall_curve(Y_val, gbm_probs)[1] + precision_recall_curve(Y_val, gbm_probs)[0] + 1e-8))
    results["Gradient Boosting"] = {"AUC": gbm_auc, "F1": gbm_f1, "Hit@1": "N/A"}
    
    # ---------------------------------------------------------
    # PyTorch DataLoader
    # ---------------------------------------------------------
    train_loader = DataLoader(EthBagDataset(train_recs), batch_size=1, shuffle=True)
    val_loader = DataLoader(EthBagDataset(val_recs), batch_size=1, shuffle=False)
    
    # ---------------------------------------------------------
    # B. Sequence / Transformer Baselines
    # ---------------------------------------------------------
    print("--- [3/7] Training Bi-LSTM ---")
    lstm_model = train_dl_model(BiLSTM_Baseline(), train_loader, val_loader, epochs=15, model_type="mil")
    lstm_auc, lstm_f1 = eval_dl_model(lstm_model, val_loader, model_type="mil")
    results["Bi-LSTM"] = {"AUC": lstm_auc, "F1": lstm_f1, "Hit@1": "N/A"}

    print("--- [4/7] Training BERT4ETH Base ---")
    bert_model = train_dl_model(BERT4ETH_Baseline(), train_loader, val_loader, epochs=15, model_type="bert")
    bert_auc, bert_f1 = eval_dl_model(bert_model, val_loader, model_type="bert")
    results["BERT4ETH Base"] = {"AUC": bert_auc, "F1": bert_f1, "Hit@1": "N/A"}

    # ---------------------------------------------------------
    # C. MIL Baselines
    # ---------------------------------------------------------
    print("--- [5/7] Training Mean-Pooling MIL ---")
    mean_model = train_dl_model(MeanMIL_Baseline(), train_loader, val_loader, epochs=15, model_type="mil")
    mean_auc, mean_f1 = eval_dl_model(mean_model, val_loader, model_type="mil")
    results["Mean-MIL"] = {"AUC": mean_auc, "F1": mean_f1, "Hit@1": "N/A"}

    print("--- [6/7] Training Max-Pooling MIL ---")
    max_model = train_dl_model(MaxMIL_Baseline(), train_loader, val_loader, epochs=15, model_type="mil")
    max_auc, max_f1 = eval_dl_model(max_model, val_loader, model_type="mil")
    results["Max-MIL"] = {"AUC": max_auc, "F1": max_f1, "Hit@1": "N/A"}
    
    print("--- [7/7] Training ABMIL (Ilse et al. 2018) ---")
    abmil_model = train_dl_model(GatedAttentionABMIL(), train_loader, val_loader, epochs=15, model_type="mil")
    abmil_auc, abmil_f1 = eval_dl_model(abmil_model, val_loader, model_type="mil")
    abmil_hit1 = eval_localization(abmil_model, test_recs, gt_data)
    results["ABMIL (Ilse 2018)"] = {"AUC": abmil_auc, "F1": abmil_f1, "Hit@1": abmil_hit1}
    
    # ---------------------------------------------------------
    # Output Summary
    # ---------------------------------------------------------
    with open(RESULTS_DIR / "step13_baselines_7models.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print(f"{'Model':<22} | {'AUC':<10} | {'F1':<10} | {'Hit@1 (%)':<10}")
    print("-" * 65)
    for model_name, metrics in results.items():
        hit = metrics["Hit@1"]
        hit_str = f"{hit:.2f}" if isinstance(hit, (int, float)) else hit
        print(f"{model_name:<22} | {metrics['AUC']:<10.4f} | {metrics['F1']:<10.4f} | {hit_str:<10}")
    print("="*70)
    print("[OK] Comprehensive 7-Baseline comparison ready!")

if __name__ == "__main__":
    main()
