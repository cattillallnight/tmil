import sys
import pickle
import torch
from pathlib import Path

sys.path.append(r"c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth")
from step26_train_random_ablation import evaluate_on_tornado_cash, RESULTS_DIR, DEVICE
from src.model import GatedTMILETH

def eval_seed(seed):
    ckpt = RESULTS_DIR / "checkpoints" / f"tmil_random_final_seed{seed}.pt"
    if not ckpt.exists():
        ckpt = RESULTS_DIR / "checkpoints" / "tmil_random_final.pt" # Seed 42 was saved as this probably?
        if not ckpt.exists():
            print(f"Skipping {seed}, no checkpoint.")
            return

    print(f"\n--- EVALUATING SEED {seed} ---")
    features_file = RESULTS_DIR / "step02d_features_hybrid_norm.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)
        
    model = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    evaluate_on_tornado_cash(model, records, seed)

if __name__ == "__main__":
    eval_seed(42)
    eval_seed(43)
