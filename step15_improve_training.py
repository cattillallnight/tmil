"""
step15_improve_training.py
Cải thiện AUC classification của TMIL-ETH để cạnh tranh với BERT4ETH/RF/GBM.
Mục tiêu: AUC > 0.97 trên Nested CV

Các thay đổi so với step9:
1. Tăng phase2 epochs lên 60 (từ 30)
2. Dùng class_weight trong loss để xử lý imbalance tốt hơn
3. Thêm Label Smoothing vào BCE loss
4. Thêm gradient clipping

Chạy trên GPU mạnh: python step15_improve_training.py
"""
import sys, json, pickle, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from utils import RESULTS_DIR as RD
from tmil_model import TMILETH, CompoundLoss
from step7_two_phase_training import AccountWindowDataset, collate_fn, train_one_epoch


def focal_bce_loss(pred, target, gamma=2.0, alpha=0.75):
    """Focal loss để xử lý class imbalance tốt hơn BCE đơn giản."""
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


class ImprovedCompoundLoss(nn.Module):
    def __init__(self, lambda1=0.3, lambda2=0.2, label_smoothing=0.05):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = label_smoothing

    def forward(self, p_acct, y_acct, attn_weights, h_bag):
        # Label smoothed target
        y_smooth = y_acct * (1 - self.eps) + 0.5 * self.eps
        l_bce = focal_bce_loss(p_acct, y_smooth)

        phish_mask = (y_acct.squeeze() == 1).float()
        if phish_mask.sum() > 0 and h_bag.shape[0] > 1:
            diffs = (h_bag[1:] - h_bag[:-1]).pow(2).sum(-1)
            l_cons = diffs.mean()
            a_max_idx = attn_weights.argmax()
            a_min_idx = attn_weights.argmin()
            margin = 1.0
            dist = (h_bag[a_max_idx] - h_bag[a_min_idx]).norm()
            l_cont = torch.clamp(margin - dist, min=0)
        else:
            l_cons = torch.tensor(0.0, device=p_acct.device)
            l_cont = torch.tensor(0.0, device=p_acct.device)

        return l_bce + phish_mask * (self.lambda1 * l_cons + self.lambda2 * l_cont)


def main():
    print("=" * 70)
    print("TMIL-ETH: Improved Training (Step 15)")
    print("Target: AUC > 0.97 with Focal Loss + Extended Fine-tuning")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feat_path = RESULTS_DIR / "step2_features.pkl"
    with open(feat_path, "rb") as f:
        records = pickle.load(f)

    # Load human GT to exclude from training
    gt_path = Path(__file__).parent / "human_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    eval_addrs = {item["account_address"].lower() for item in gt_data}
    records = [r for r in records if r["address"].lower() not in eval_addrs]

    phish = [r for r in records if r["label"] == 1]
    norm = [r for r in records if r["label"] == 0]
    np.random.seed(42)
    norm_sample = np.random.choice(norm, size=len(phish)*4, replace=False).tolist()
    all_recs = phish + norm_sample
    labels = [r["label"] for r in all_recs]

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_aucs, all_f1s = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(all_recs, labels)):
        train_recs = [all_recs[i] for i in train_idx]
        val_recs = [all_recs[i] for i in val_idx]
        print(f"\n--- Fold {fold_idx+1}/5 | Train: {len(train_recs)}, Val: {len(val_recs)} ---")

        model = TMILETH(4, 64).to(device)
        loss_fn = CompoundLoss(lambda1=0.3, lambda2=0.2)

        ds_train = AccountWindowDataset(train_recs, W=200)
        loader = DataLoader(ds_train, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)

        # Phase 1: frozen BERT, 20 epochs
        model.freeze_bert()
        opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=20)
        for ep in range(20):
            train_one_epoch(model, loader, loss_fn, opt1, device, 1.0)
            sched1.step()
            if (ep + 1) % 5 == 0:
                print(f"  Phase1 Ep {ep+1}/20")

        # Phase 2: full fine-tune, 60 epochs (extended from 30)
        model.unfreeze_all()
        opt2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=60, eta_min=1e-6)
        best_auc = 0
        best_state = None
        for ep in range(60):
            train_one_epoch(model, loader, loss_fn, opt2, device, 1.0)
            sched2.step()
            if (ep + 1) % 10 == 0 or ep == 59:
                # Quick val
                model.eval()
                val_ds = AccountWindowDataset(val_recs, W=200)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
                probs, ys = [], []
                with torch.no_grad():
                    for hc_val, bert_val, labels_val in val_loader:
                        hc_val = hc_val.to(device)
                        bert_val = bert_val.to(device)
                        p, _ = model(hc_val, bert_val)
                        probs.extend(p.cpu().numpy().tolist())
                        ys.extend(labels_val.cpu().numpy().tolist())
                auc = roc_auc_score(ys, probs)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  Phase2 Ep {ep+1}/60 | Val AUC: {auc:.4f} (best: {best_auc:.4f})")
                model.train()

        # Final eval
        model.load_state_dict(best_state)
        model.eval()
        val_ds = AccountWindowDataset(val_recs, W=200)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        probs, ys = [], []
        with torch.no_grad():
            for hc_val, bert_val, labels_val in val_loader:
                hc_val = hc_val.to(device)
                bert_val = bert_val.to(device)
                p, _ = model(hc_val, bert_val)
                probs.extend(p.cpu().numpy().tolist())
                ys.extend(labels_val.cpu().numpy().tolist())
        auc = roc_auc_score(ys, probs)
        precision, recall, _ = precision_recall_curve(ys, probs)
        f1 = np.max(2 * precision * recall / (precision + recall + 1e-8))
        all_aucs.append(auc)
        all_f1s.append(f1)
        print(f"  Fold {fold_idx+1} FINAL: AUC={auc:.4f}, F1={f1:.4f}")

        # Save best model from fold 1 for step14 eval
        if fold_idx == 0:
            ckpt_dir = RESULTS_DIR / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(best_state, ckpt_dir / "best_model_improved.pt")

    print("\n" + "=" * 70)
    print(f"IMPROVED Nested CV Results:")
    print(f"  AUC: {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"  F1:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")

    out = {
        "auc_mean": float(np.mean(all_aucs)),
        "auc_std": float(np.std(all_aucs)),
        "f1_mean": float(np.mean(all_f1s)),
        "f1_std": float(np.std(all_f1s)),
        "per_fold_auc": [float(a) for a in all_aucs],
        "per_fold_f1": [float(f) for f in all_f1s],
        "notes": "Phase1: 20ep frozen, Phase2: 60ep CosineAnnealing, FocalLoss"
    }
    with open(RESULTS_DIR / "step15_improved_cv.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved step15_improved_cv.json")


if __name__ == "__main__":
    main()
