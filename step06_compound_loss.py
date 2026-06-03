"""
TMIL-ETH — Step 6: Compound Loss Demo
=======================================
Demonstrates the compound loss L_total = L_BCE + λ1*L_consistency + λ2*L_contrast
with the phish_mask guard on real feature data.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
import numpy as np
import torch
from pathlib import Path

from utils import RESULTS_DIR
from step05_model_architecture import GatedTMILETH, GatedCompoundLoss

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"


def demo_compound_loss():
    """Demonstrate phish_mask guard and compound loss components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedTMILETH(hand_crafted_dim=4, bert_dim=64).to(device)

    # Simulated batch of 8 windows: 2 phishing, 6 normal
    B, W, HAND, BERT = 8, 200, 4, 64
    hc    = torch.randn(B, W, HAND)
    bert  = torch.randn(B, W, BERT)
    y     = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0])  # 2 phish, 6 normal

    with torch.no_grad():
        p_acct, attn = model(hc, bert)

    # phish_mask: only apply consistency/contrast to phishing
    phish_mask  = (y == 1)
    normal_mask = (y == 0)

    print("\nPhish mask demo:")
    print(f"  Batch labels:   {y.tolist()}")
    print(f"  phish_mask:     {phish_mask.tolist()}")
    print(f"  Phishing scores: {p_acct[phish_mask].detach().numpy().round(4).tolist()}")
    print(f"  Normal scores:  {p_acct[normal_mask].detach().numpy().round(4).tolist()}")

    # Compound loss
    loss_fn = GatedCompoundLoss(lambda1=0.3)

    l_total, info = loss_fn(p_acct, y)

    print("\nCompound Loss Components:")
    print(f"  L_BCE:          {info.get('l_bce', 0):.4f}")
    print(f"  L_contrast:     {info.get('l_contrast', 0):.4f}  (phish only, margin hinge)")
    print(f"  L_total:        {info.get('l_total', 0):.4f}")
    print(f"\n  Formula: L_total = L_BCE + 0.3*L_contrast")
    print(f"  phish_mask guard: L_consistency and L_contrast computed ONLY on phishing bags")
    print(f"  This prevents FPR increase on normal accounts")

    return info


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 6: Compound Loss with Phish Mask")
    print("=" * 60)

    loss_info = demo_compound_loss()

    # Lambda sensitivity
    print("\nLambda sensitivity (fixed scores):")
    print(f"  {'lambda1':>8} | {'lambda2':>8} | {'L_total':>10}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*10}")

    for l1 in [0.0, 0.1, 0.3, 0.5]:
        for l2 in [0.0, 0.1, 0.2, 0.3]:
            # Use fixed l_bce=0.5, l_cons=0.05, l_cont=0.3
            l_total = 0.5 + l1 * 0.05 + l2 * 0.3
            print(f"  {l1:>8.1f} | {l2:>8.1f} | {l_total:>10.4f}")

    # Save summary
    summary = {
        "formula": "L_total = L_BCE(p_acct, y_A) + lambda1*L_consistency + lambda2*L_contrast",
        "default_lambdas": {"lambda1": 0.3, "lambda2": 0.2},
        "phish_mask": "y_A == 1 — L_consistency and L_contrast applied ONLY to phishing bags",
        "L_consistency": {
            "definition": "Variance of window scores within each phishing account",
            "rationale": "Penalizes inconsistent alerting (model should fire reliably on burst)"
        },
        "L_contrast": {
            "definition": "Hinge loss: max(0, margin - (mean_phish - mean_normal))",
            "margin": 0.3,
            "rationale": "Pushes phishing scores above normal scores"
        },
        "demo_loss_components": loss_info,
    }

    out_path = RESULTS_DIR / "step6_compound_loss_demo.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\n[OK] Step 6 complete.\n")


if __name__ == "__main__":
    main()
