import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. ABMIL Model (Adapted from Ilse et al., 2018)
# -------------------------------------------------------------------------
class GatedAttentionABMIL(nn.Module):
    def __init__(self, input_dim=68, M=64, L=64, ATTENTION_BRANCHES=1):
        super(GatedAttentionABMIL, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        # Replace Conv2d with Linear for our 68-dim vectors
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.M),
            nn.ReLU()
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [K, 68]
        H = self.feature_extractor(x)  # [K, M]

        A_V = self.attention_V(H)  # [K, L]
        A_U = self.attention_U(H)  # [K, L]
        A = self.attention_w(A_V * A_U)  # [K, 1]
        A = torch.transpose(A, 1, 0)  # [1, K]
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # [1, M]
        Y_prob = self.classifier(Z)
        
        return Y_prob, A.squeeze(0)

# -------------------------------------------------------------------------
# 2. Bi-LSTM Model
# -------------------------------------------------------------------------
class BiLSTM_Baseline(nn.Module):
    def __init__(self, input_dim=68, hidden_dim=64):
        super(BiLSTM_Baseline, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [1, K, 68]
        output, (hn, cn) = self.lstm(x)
        # hn is [2, 1, 64] -> concat to [1, 128]
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        prob = self.classifier(hidden)
        return prob

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
        hc_win_mean = np.mean(hc_win, axis=0) # (4,)
        win_feat = np.concatenate([hc_win_mean, bert]) # (68,)
        bag_features.append(win_feat)
    return np.array(bag_features, dtype=np.float32)

class EthBagDataset(Dataset):
    def __init__(self, records, W=200):
        self.W = W
        self.items = []
        for rec in records:
            bag_features = get_bag_features(rec, W)
            y = rec["label"]
            self.items.append((bag_features, y, rec["address"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, addr = self.items[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32), addr

# -------------------------------------------------------------------------
# Utility / Metric Functions
# -------------------------------------------------------------------------
def train_dl_model(model, train_loader, val_loader, epochs=20, lr=1e-3, is_lstm=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if is_lstm:
                # LSTM expects batch_first: [1, K, 68]
                prob = model(x)
            else:
                # ABMIL expects [K, 68]
                prob, _ = model(x.squeeze(0))
            
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                if is_lstm:
                    prob = model(x)
                else:
                    prob, _ = model(x.squeeze(0))
                all_probs.append(prob.item())
                all_y.append(y.item())
                
        auc = roc_auc_score(all_y, all_probs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), RESULTS_DIR / "temp_best_model.pt")
            
    model.load_state_dict(torch.load(RESULTS_DIR / "temp_best_model.pt", weights_only=True))
    return model

def eval_dl_model(model, loader, is_lstm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            if is_lstm:
                prob = model(x)
            else:
                prob, _ = model(x.squeeze(0))
            all_probs.append(prob.item())
            all_y.append(y.item())
            
    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    
    auc = roc_auc_score(all_y, all_probs)
    precision, recall, thresholds = precision_recall_curve(all_y, all_probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    best_f1 = np.max(f1_scores)
    
    return auc, best_f1

def eval_abmil_localization(model, test_recs, gt_data):
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
            A = A.cpu().numpy()
            
            max_idx = np.argmax(A)
            
            win_start = max_idx * 50
            win_end = win_start + 200
            
            # Hit@1 Logic
            if win_start <= gt_start and win_end >= gt_end:
                hit += 1
            elif win_start <= gt_end and win_end >= gt_start:
                hit += 1
                
            eval_count += 1
            
    return (hit / eval_count) * 100 if eval_count > 0 else 0

def main():
    print("=" * 60)
    print("TMIL-ETH: Baseline Comparison (RF, Bi-LSTM, ABMIL)")
    print("=" * 60)
    
    # Load features
    feat_path = RESULTS_DIR / "step2_features.pkl"
    if not feat_path.exists():
        print("Features not found. Please run step2_feature_extraction.py first.")
        return
        
    print("Loading features...")
    with open(feat_path, "rb") as f:
        records = pickle.load(f)
        
    # Load Ground Truth
    gt_path = Path(__file__).parent / "human_ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            gt_data = json.load(f)
    else:
        gt_data = []
        
    eval_addrs = {item["account_address"].lower() for item in gt_data}
    
    # ---------------------------------------------------------
    # Split Data
    # ---------------------------------------------------------
    test_recs = [r for r in records if r["address"].lower() in eval_addrs]
    train_pool = [r for r in records if r["address"].lower() not in eval_addrs]
    
    phish_pool = [r for r in train_pool if r["label"] == 1]
    norm_pool = [r for r in train_pool if r["label"] == 0]
    
    # Subsample normal to 1:4 ratio for training (Full Scale Dataset)
    np.random.seed(42)
    
    norm_sample = np.random.choice(norm_pool, size=len(phish_pool)*4, replace=False).tolist()
    train_val_recs = phish_pool + norm_sample
    
    train_recs, val_recs = train_test_split(train_val_recs, test_size=0.2, random_state=42, stratify=[r["label"] for r in train_val_recs])
    
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test (Forensic): {len(test_recs)}")
    
    # ---------------------------------------------------------
    # 1. Random Forest (Mean Pooling)
    # ---------------------------------------------------------
    print("\n--- Training Random Forest (Mean Pooling) ---")
    def get_rf_features(recs):
        X, Y = [], []
        dataset = EthBagDataset(recs)
        for x, y, _ in dataset:
            mean_feat = torch.mean(x, dim=0).numpy()
            X.append(mean_feat)
            Y.append(y.item())
        return np.array(X), np.array(Y)
        
    X_train_rf, Y_train_rf = get_rf_features(train_recs)
    X_val_rf, Y_val_rf = get_rf_features(val_recs)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_rf, Y_train_rf)
    rf_probs = rf.predict_proba(X_val_rf)[:, 1]
    
    rf_auc = roc_auc_score(Y_val_rf, rf_probs)
    precision, recall, _ = precision_recall_curve(Y_val_rf, rf_probs)
    rf_f1 = np.max(2 * recall * precision / (recall + precision + 1e-8))
    
    print(f"Random Forest -> AUC: {rf_auc:.4f}, F1: {rf_f1:.4f}, Hit@1: N/A")
    
    # ---------------------------------------------------------
    # PyTorch DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(EthBagDataset(train_recs), batch_size=1, shuffle=True)
    val_loader = DataLoader(EthBagDataset(val_recs), batch_size=1, shuffle=False)
    
    # ---------------------------------------------------------
    # 2. Bi-LSTM
    # ---------------------------------------------------------
    print("\n--- Training Bi-LSTM ---")
    lstm_model = BiLSTM_Baseline(input_dim=68, hidden_dim=64)
    lstm_model = train_dl_model(lstm_model, train_loader, val_loader, epochs=15, lr=5e-4, is_lstm=True)
    lstm_auc, lstm_f1 = eval_dl_model(lstm_model, val_loader, is_lstm=True)
    print(f"Bi-LSTM -> AUC: {lstm_auc:.4f}, F1: {lstm_f1:.4f}, Hit@1: N/A")
    
    # ---------------------------------------------------------
    # 3. ABMIL (Ilse et al., 2018)
    # ---------------------------------------------------------
    print("\n--- Training ABMIL (Ilse et al. 2018) ---")
    abmil_model = GatedAttentionABMIL(input_dim=68)
    abmil_model = train_dl_model(abmil_model, train_loader, val_loader, epochs=15, lr=5e-4, is_lstm=False)
    abmil_auc, abmil_f1 = eval_dl_model(abmil_model, val_loader, is_lstm=False)
    
    abmil_hit1 = eval_abmil_localization(abmil_model, test_recs, gt_data)
    print(f"ABMIL -> AUC: {abmil_auc:.4f}, F1: {abmil_f1:.4f}, Hit@1: {abmil_hit1:.2f}%")
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    results = {
        "Random_Forest": {"AUC": rf_auc, "F1": rf_f1, "Hit_at_1": "N/A"},
        "Bi_LSTM": {"AUC": lstm_auc, "F1": lstm_f1, "Hit_at_1": "N/A"},
        "ABMIL": {"AUC": abmil_auc, "F1": abmil_f1, "Hit_at_1": abmil_hit1}
    }
    
    with open(RESULTS_DIR / "step13_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- Baseline Results Summary ---")
    print(f"{'Model':<20} | {'AUC':<10} | {'F1':<10} | {'Hit@1 (%)':<10}")
    print("-" * 55)
    print(f"{'Random Forest':<20} | {rf_auc:<10.4f} | {rf_f1:<10.4f} | {'N/A':<10}")
    print(f"{'Bi-LSTM':<20} | {lstm_auc:<10.4f} | {lstm_f1:<10.4f} | {'N/A':<10}")
    print(f"{'ABMIL (Ilse 2018)':<20} | {abmil_auc:<10.4f} | {abmil_f1:<10.4f} | {abmil_hit1:<10.2f}")
    
    print("\n[OK] Baseline comparison completed successfully.")

if __name__ == "__main__":
    main()
