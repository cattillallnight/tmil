import sys
import os
import pickle
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RESULTS_DIR
from src.model import GatedTMILETH
from step26_train_random_ablation import evaluate_on_tornado_cash

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("Loading normalized features...")
    features_file = RESULTS_DIR / "step02d_features_hybrid_norm.pkl"
    with open(features_file, "rb") as f:
        records = pickle.load(f)

    seed = 44
    print(f"\nEvaluating Seed {seed} from saved checkpoint...")
    
    model = GatedTMILETH(hand_crafted_dim=5, bert_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / "checkpoints" / f"tmil_random_final_seed{seed}.pt", weights_only=True))
    model.eval()
    
    evaluate_on_tornado_cash(model, records, seed)

if __name__ == "__main__":
    main()
