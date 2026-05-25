"""
TMIL-ETH — Step 10: Ablation Study & Interpretability
=======================================================
7 ablation configurations (§7.1):
  1. Full TMIL-ETH (baseline)
  2. No L_consistency (lambda1=0)
  3. No L_contrast (lambda2=0)
  4. BCE only (lambda1=lambda2=0)
  5. Single pooling (attention only)
  6. No sliding window (last 200 txs only, single window)
  7. Global normalization (not per-account)

Interpretability (§8.1):
  gamma = |Top-10% attention txs ∩ Burst proxy| / |Burst proxy|
  Burst proxy: density_i > p75 AND value_ratio_i > 1

Generates:
  results/step10_ablation_table.csv
  results/step10_saliency_maps/  (per-account attention CSV)
  results/step10_interpretability.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from utils import RESULTS_DIR, sidak_threshold
from tmil_model import TMILETH, CompoundLoss, TriplePoolingMIL, AttentionMIL
from step7_two_phase_training import (AccountWindowDataset, collate_fn,
                                       train_one_epoch, evaluate_epoch,
                                       compute_metrics)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SALIENCY_DIR = RESULTS_DIR / "step10_saliency_maps"
SALIENCY_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"

SEED = 42
W, S = 200, 50
HAND_DIM, BERT_DIM = 4, 64
PHASE1_EP, PHASE2_EP = 10, 15
BATCH_SIZE = 32
LR1, LR2, LR2_MIN = 1e-3, 5e-5, 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


# ─── Ablation model variants ─────────────────────────────────────────────────

class SinglePoolingMIL(nn.Module):
    """Ablation 5: attention-only pooling (no z_mean, z_max)."""
    def __init__(self, input_dim: int, attn_hidden: int = 128, mlp_hidden: int = 256):
        super().__init__()
        self.attention = AttentionMIL(input_dim, attn_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        z_attn, attn = self.attention(x)
        logit = self.mlp(z_attn).squeeze(-1)
        return torch.sigmoid(logit), attn


class SinglePoolingTMIL(nn.Module):
    """Wraps SinglePoolingMIL with same interface as TMILETH."""
    def __init__(self, hand_crafted_dim=4, bert_dim=64, proj_dim=64):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(hand_crafted_dim + bert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.mil = SinglePoolingMIL(proj_dim)

    def forward(self, hand_crafted, bert_embed):
        x = torch.cat([hand_crafted, bert_embed], dim=-1)
        x = self.feature_proj(x)
        return self.mil(x)

    def freeze_bert(self):
        for p in self.feature_proj.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ─── Ablation configs ─────────────────────────────────────────────────────────

def get_ablation_configs():
    return [
        {
            "name": "Full TMIL-ETH",
            "lambda1": 0.3,
            "lambda2": 0.2,
            "single_pool": False,
            "single_window": False,
            "global_norm": False,
            "description": "Complete model with triple pooling + compound loss",
        },
        {
            "name": "No L_consistency",
            "lambda1": 0.0,
            "lambda2": 0.2,
            "single_pool": False,
            "single_window": False,
            "global_norm": False,
            "description": "Remove L_consistency (lambda1=0)",
        },
        {
            "name": "No L_contrast",
            "lambda1": 0.3,
            "lambda2": 0.0,
            "single_pool": False,
            "single_window": False,
            "global_norm": False,
            "description": "Remove L_contrast (lambda2=0)",
        },
        {
            "name": "BCE only",
            "lambda1": 0.0,
            "lambda2": 0.0,
            "single_pool": False,
            "single_window": False,
            "global_norm": False,
            "description": "Binary cross-entropy only (no compound loss)",
        },
        {
            "name": "Single pooling",
            "lambda1": 0.3,
            "lambda2": 0.2,
            "single_pool": True,
            "single_window": False,
            "global_norm": False,
            "description": "Attention-only pooling (no z_mean, z_max)",
        },
        {
            "name": "No sliding window",
            "lambda1": 0.3,
            "lambda2": 0.2,
            "single_pool": False,
            "single_window": True,
            "global_norm": False,
            "description": "Single window (last W transactions, no full sweep)",
        },
        {
            "name": "Global normalization",
            "lambda1": 0.3,
            "lambda2": 0.2,
            "single_pool": False,
            "single_window": False,
            "global_norm": True,
            "description": "Global z-score normalization instead of per-account",
        },
    ]


def prepare_records_with_global_norm(records: list) -> list:
    """Apply global (not per-account) normalization to z_amount."""
    all_amounts = []
    for rec in records:
        all_amounts.extend(rec["hand_crafted"][:, 0].tolist())
    global_mean = np.mean(all_amounts)
    global_std  = np.std(all_amounts) + 1e-9

    import copy
    modified = []
    for rec in records:
        r = copy.deepcopy(rec)
        amounts = r["hand_crafted"][:, 0]
        r["hand_crafted"][:, 0] = np.clip((amounts - global_mean) / global_std, -3, 3) / 3
        modified.append(r)
    return modified


def train_eval_ablation(config: dict, train_recs: list, test_recs: list,
                        device) -> dict:
    """Train and evaluate a single ablation configuration."""
    loss_fn = CompoundLoss(lambda1=config["lambda1"], lambda2=config["lambda2"])

    # Global norm variant
    if config["global_norm"]:
        all_recs = train_recs + test_recs
        all_recs_normed = prepare_records_with_global_norm(all_recs)
        split = len(train_recs)
        train_recs = all_recs_normed[:split]
        test_recs  = all_recs_normed[split:]

    # Single window variant
    if config["single_window"]:
        import copy
        def to_single_window(recs):
            modified = []
            for rec in recs:
                r = copy.deepcopy(rec)
                r["windows"] = [(max(0, rec["n_tx"] - W), rec["n_tx"])]
                r["n_windows"] = 1
                modified.append(r)
            return modified
        train_recs = to_single_window(train_recs)
        test_recs  = to_single_window(test_recs)

    # Model
    if config["single_pool"]:
        model = SinglePoolingTMIL(HAND_DIM, BERT_DIM).to(device)
    else:
        model = TMILETH(HAND_DIM, BERT_DIM).to(device)

    train_ds = AccountWindowDataset(train_recs, W=W)
    test_ds  = AccountWindowDataset(test_recs,  W=W)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR1, weight_decay=WEIGHT_DECAY)
    for _ in range(PHASE1_EP):
        train_one_epoch(model, train_loader, loss_fn, opt1, device, GRAD_CLIP)

    model.unfreeze_all()
    opt2 = optim.AdamW(model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=LR2_MIN)
    for _ in range(PHASE2_EP):
        train_one_epoch(model, train_loader, loss_fn, opt2, device, GRAD_CLIP)
        sched.step()

    _, preds, labels = evaluate_epoch(model, test_loader, loss_fn, device)
    metrics = compute_metrics(preds, labels)

    # FPR@95%TPR
    try:
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, _ = roc_curve(labels, preds)
        idx = np.argmin(np.abs(tpr_arr - 0.95))
        metrics["fpr_at_95tpr"] = float(fpr_arr[idx])
    except Exception:
        metrics["fpr_at_95tpr"] = 1.0

    return metrics, model, preds, labels


# ─── Interpretability: attention alignment gamma ───────────────────────────

def compute_gamma(model: nn.Module, test_recs: list, device,
                  n_sample: int = 100) -> dict:
    """
    gamma = |Top-10% attention txs intersection Burst proxy| / |Burst proxy|
    Burst proxy: density_i > p75 AND value_ratio_i > 1

    Computed on sample of phishing accounts.
    """
    model.eval()
    phish_recs = [r for r in test_recs if r["label"] == 1][:n_sample]

    gamma_values = []
    for rec in phish_recs:
        hc = rec["hand_crafted"]      # (n_tx, 4)
        bert = rec["bert_embedding"]  # (64,)
        wins = rec["windows"]
        n_tx = rec["n_tx"]

        if n_tx < 10:
            continue

        # Get window with highest score (the forensic burst window)
        best_attn = None
        best_hc_win = None

        for (start, end) in wins:
            hc_win = hc[start:end]
            n = hc_win.shape[0]
            if n < W:
                pad = np.zeros((W - n, 4), dtype=np.float32)
                hc_win = np.vstack([hc_win, pad])
            elif n > W:
                hc_win = hc_win[:W]

            hc_t = torch.tensor(hc_win, dtype=torch.float32).unsqueeze(0).to(device)
            bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0)
            bert_t = bert_t.unsqueeze(0).expand(-1, W, -1).to(device)

            with torch.no_grad():
                p, attn = model(hc_t, bert_t)

            attn_np = attn.squeeze(0).cpu().numpy()
            if best_attn is None or p.item() > 0.5:
                best_attn = attn_np
                best_hc_win = hc_win

        if best_attn is None or best_hc_win is None:
            continue

        # Top-10% attention transactions
        n_top = max(1, int(0.10 * W))
        top_idx = np.argsort(best_attn)[-n_top:]  # highest attention scores

        # Burst proxy on window
        density  = best_hc_win[:, 1]   # density feature
        v_ratio  = best_hc_win[:, 3]   # value_ratio feature
        p75_density = np.percentile(density, 75)
        burst_mask = (density > p75_density) & (v_ratio > 1.0)

        n_burst = np.sum(burst_mask)
        if n_burst == 0:
            continue

        # Intersection: top attention AND burst proxy
        top_mask = np.zeros(W, dtype=bool)
        top_mask[top_idx] = True
        intersection = np.sum(top_mask & burst_mask)

        gamma = intersection / n_burst
        gamma_values.append(gamma)

    if not gamma_values:
        return {"gamma_mean": 0.0, "gamma_median": 0.0, "gamma_std": 0.0, "n_accounts": 0}

    return {
        "gamma_mean":   float(np.mean(gamma_values)),
        "gamma_median": float(np.median(gamma_values)),
        "gamma_std":    float(np.std(gamma_values)),
        "n_accounts":   len(gamma_values),
    }


def generate_saliency_map(model: nn.Module, rec: dict, device) -> pd.DataFrame:
    """Generate transaction-level attention saliency for one account."""
    model.eval()
    hc   = rec["hand_crafted"]
    bert = rec["bert_embedding"]
    wins = rec["windows"]

    all_rows = []
    for win_idx, (start, end) in enumerate(wins):
        hc_win = hc[start:end]
        n = hc_win.shape[0]
        if n < W:
            pad = np.zeros((W - n, 4), dtype=np.float32)
            hc_win_pad = np.vstack([hc_win, pad])
        elif n > W:
            hc_win_pad = hc_win[:W]
        else:
            hc_win_pad = hc_win

        hc_t = torch.tensor(hc_win_pad, dtype=torch.float32).unsqueeze(0).to(device)
        bert_t = torch.tensor(bert, dtype=torch.float32).unsqueeze(0)
        bert_t = bert_t.unsqueeze(0).expand(-1, W, -1).to(device)

        with torch.no_grad():
            p, attn = model(hc_t, bert_t)

        attn_np = attn.squeeze(0).cpu().numpy()

        for tx_i, (global_idx, attn_score) in enumerate(
                zip(range(start, min(end, start + W)), attn_np[:n])):
            all_rows.append({
                "window_idx": win_idx,
                "tx_global_idx": global_idx,
                "tx_in_window_idx": tx_i,
                "attention_score": float(attn_score),
                "window_score": float(p.item()),
                "z_amount": float(hc[global_idx, 0]) if global_idx < len(hc) else 0,
                "density": float(hc[global_idx, 1]) if global_idx < len(hc) else 0,
                "counterparty_novelty": float(hc[global_idx, 2]) if global_idx < len(hc) else 0,
                "value_ratio": float(hc[global_idx, 3]) if global_idx < len(hc) else 0,
            })

    return pd.DataFrame(all_rows)


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 10: Ablation Study & Interpretability")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run Step 2 first.")
        return

    print(f"\nLoading features...")
    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)

    # Use subset for speed: all phishing + 4x normal (1:4 ratio)
    phish_recs  = [r for r in records if r["label"] == 1]
    normal_recs = [r for r in records if r["label"] == 0]

    rng = np.random.RandomState(SEED)
    n_norm = min(4 * len(phish_recs), len(normal_recs))
    norm_idx = rng.choice(len(normal_recs), n_norm, replace=False)
    normal_subset = [normal_recs[i] for i in norm_idx]
    all_recs = phish_recs + normal_subset

    print(f"  Dataset: {len(all_recs):,} accounts "
          f"(phish={len(phish_recs):,}, normal={len(normal_subset):,})")

    labels_arr = [r["label"] for r in all_recs]
    train_recs, test_recs = train_test_split(
        all_recs, test_size=0.2, stratify=labels_arr, random_state=SEED
    )

    # ── Ablation study ────────────────────────────────────────────
    ablation_configs = get_ablation_configs()
    ablation_results = []

    print(f"\n[1] Running {len(ablation_configs)} ablation configurations...")
    for i, config in enumerate(ablation_configs):
        print(f"\n  Config {i+1}/{len(ablation_configs)}: {config['name']}")
        try:
            metrics, model, preds, labels = train_eval_ablation(
                config, train_recs, test_recs, device
            )
            # Compute gamma for interpretability
            if not config["single_pool"]:
                gamma = compute_gamma(model, test_recs, device, n_sample=50)
            else:
                gamma = {"gamma_mean": None, "gamma_median": None,
                         "n_accounts": 0, "note": "N/A for single pool"}

            result = {
                "config": config["name"],
                "description": config["description"],
                "metrics": metrics,
                "gamma": gamma,
            }
            print(f"    AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                  f"FPR@95%TPR={metrics.get('fpr_at_95tpr', 0):.4f}")
            if gamma.get("gamma_mean") is not None:
                print(f"    gamma={gamma['gamma_mean']:.4f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            result = {"config": config["name"], "description": config["description"],
                      "error": str(e)}

        ablation_results.append(result)

    # ── Saliency maps for top phishing accounts ─────────────────
    print(f"\n[2] Generating saliency maps for 10 sample phishing accounts...")
    full_model = TMILETH(HAND_DIM, BERT_DIM).to(device)

    # Re-train full model
    loss_fn = CompoundLoss(lambda1=0.3, lambda2=0.2)
    train_ds = AccountWindowDataset(train_recs, W=W)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    full_model.freeze_bert()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, full_model.parameters()),
                       lr=LR1, weight_decay=WEIGHT_DECAY)
    for _ in range(PHASE1_EP):
        train_one_epoch(full_model, train_loader, loss_fn, opt1, device, GRAD_CLIP)
    full_model.unfreeze_all()
    opt2 = optim.AdamW(full_model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=LR2_MIN)
    for _ in range(PHASE2_EP):
        train_one_epoch(full_model, train_loader, loss_fn, opt2, device, GRAD_CLIP)
        sched.step()

    sample_phish = [r for r in test_recs if r["label"] == 1][:10]
    saliency_paths = []
    for rec in sample_phish:
        try:
            df_sal = generate_saliency_map(full_model, rec, device)
            addr_short = rec["address"][:10]
            sal_path = SALIENCY_DIR / f"saliency_{addr_short}.csv"
            df_sal.to_csv(sal_path, index=False)
            saliency_paths.append(str(sal_path))
        except Exception as e:
            print(f"  Saliency failed for {rec['address'][:10]}: {e}")

    print(f"  Generated {len(saliency_paths)} saliency maps")

    # ── Save ablation table ────────────────────────────────────────
    rows = []
    for res in ablation_results:
        if "error" in res:
            rows.append({"Configuration": res["config"], "Error": res["error"]})
            continue
        m = res["metrics"]
        g = res.get("gamma", {})
        rows.append({
            "Configuration": res["config"],
            "AUC": round(m.get("auc", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "FPR@95%TPR": round(m.get("fpr_at_95tpr", 0), 4),
            "Gamma (attn alignment)": round(g.get("gamma_mean", 0) or 0, 4),
            "Description": res["description"],
        })

    df_ablation = pd.DataFrame(rows)
    ablation_csv = RESULTS_DIR / "step10_ablation_table.csv"
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"\nAblation table saved: {ablation_csv}")

    print("\nAblation Results:")
    print(df_ablation[["Configuration", "AUC", "F1", "FPR@95%TPR",
                        "Gamma (attn alignment)"]].to_string(index=False))

    # ── Save interpretability analysis ─────────────────────────────
    output = {
        "ablation_configurations": ablation_configs,
        "ablation_results": ablation_results,
        "saliency_maps_generated": saliency_paths,
        "interpretability_notes": {
            "gamma_formula": "gamma = |Top-10% attn txs ∩ Burst proxy| / |Burst proxy|",
            "burst_proxy": "density_i > p75 AND value_ratio_i > 1",
            "scope_limitation": (
                "gamma uses same features (density, value_ratio) as model inputs — "
                "may reflect feature learning rather than independent forensic discovery."
            ),
            "usage": "Reported as relative diagnostic only, not for model selection.",
        },
    }

    interp_path = RESULTS_DIR / "step10_interpretability.json"
    with open(interp_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Interpretability analysis: {interp_path}")

    print("\n[OK] Step 10 complete.\n")
    return output


if __name__ == "__main__":
    main()
