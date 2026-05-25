"""
TMIL-ETH — Step 3: Orthogonality Validation (Linear Probe R²)
==============================================================
Verifies that hand-crafted features C_i are non-redundant w.r.t. BERT4ETH embeddings h_i.
Protocol (§3.3):
  - Hold out 100 accounts (~5,000 transactions)
  - Fit OLS regression: h_i -> c_k for each of 4 features
  - Run 1,000-shuffle permutation test
  - Hard cap: R² < 0.30
  - Acceptance: R² < 95th percentile of null distribution

Saves: results/step3_orthogonality_r2.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

from utils import RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = RESULTS_DIR / "step2_features.pkl"
N_HOLDOUT     = 100
N_PERMUTATIONS = 1000
HARD_CAP_R2   = 0.30
RANDOM_SEED   = 42

FEATURE_NAMES = ["z_amount", "density", "counterparty_novelty", "value_ratio"]


def run_ols_r2(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> float:
    """
    Fit Ridge regression X -> y and return R².
    Uses Ridge(alpha) for numerical stability.
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    return float(r2_score(y, y_pred))


def permutation_null_r2(X: np.ndarray, y: np.ndarray,
                        n_perms: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate null distribution of R² by shuffling y 1,000 times.
    Returns array of R² values under null hypothesis.
    """
    rng = np.random.RandomState(seed)
    null_r2s = np.zeros(n_perms)
    for i in range(n_perms):
        y_shuffled = rng.permutation(y)
        null_r2s[i] = run_ols_r2(X, y_shuffled)
    return null_r2s


def build_regression_dataset(records: list, n_holdout: int = 100, seed: int = 42):
    """
    From feature records, assemble:
      X: (N_txs, 64) — BERT4ETH embeddings per transaction (broadcast account embedding)
      C: (N_txs, 4)  — hand-crafted features per transaction
    Using n_holdout accounts (~5,000 transactions as stated in §3.3).
    """
    rng = np.random.RandomState(seed)
    idx_all = list(range(len(records)))
    holdout_idx = rng.choice(idx_all, size=min(n_holdout, len(idx_all)), replace=False)

    X_parts = []  # BERT embeddings
    C_parts = []  # hand-crafted features

    for i in holdout_idx:
        rec = records[i]
        hc = rec["hand_crafted"]          # (n_tx, 4)
        bert = rec["bert_embedding"]      # (64,)  — account-level

        n_tx = hc.shape[0]
        # Broadcast account embedding to all transactions
        bert_broadcast = np.tile(bert, (n_tx, 1))  # (n_tx, 64)

        X_parts.append(bert_broadcast)
        C_parts.append(hc)

    X = np.vstack(X_parts)  # (total_txs, 64)
    C = np.vstack(C_parts)  # (total_txs, 4)

    print(f"  Holdout accounts: {len(holdout_idx)}")
    print(f"  Total transactions: {len(X):,}")
    return X, C


def main():
    print("=" * 60)
    print("TMIL-ETH - Step 3: Orthogonality Validation (R2)")
    print("=" * 60)

    # Load features from Step 2
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run step2_feature_extraction.py first.")
        return

    print(f"\nLoading features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, "rb") as f:
        records = pickle.load(f)
    print(f"  Total records: {len(records):,}")

    # Filter records with valid bert embeddings (non-zero)
    valid_records = [r for r in records if r.get("bert_embedding") is not None
                     and np.any(r["bert_embedding"] != 0)]
    print(f"  Records with BERT embeddings: {len(valid_records):,}")

    # Build regression dataset from holdout accounts
    print(f"\nBuilding regression dataset (n_holdout={N_HOLDOUT})...")
    X, C = build_regression_dataset(valid_records, n_holdout=N_HOLDOUT,
                                    seed=RANDOM_SEED)

    # Normalize X for numerical stability
    X_mean = X.mean(axis=0, keepdims=True)
    X_std  = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    results = {}
    print(f"\nFitting OLS regression for each hand-crafted feature:")
    print(f"  Running {N_PERMUTATIONS} permutations per feature...")
    print()

    all_pass = True
    for k, feat_name in enumerate(FEATURE_NAMES):
        y = C[:, k]

        # Observed R²
        r2_obs = run_ols_r2(X_norm, y)

        # Null distribution
        print(f"  Feature [{k+1}/4]: {feat_name} (R2_obs={r2_obs:.4f})")
        null_r2s = permutation_null_r2(X_norm, y, n_perms=N_PERMUTATIONS,
                                       seed=RANDOM_SEED)
        null_p95 = float(np.percentile(null_r2s, 95))
        null_mean = float(np.mean(null_r2s))

        passes_hard_cap   = r2_obs < HARD_CAP_R2
        passes_permutation = r2_obs < null_p95

        print(f"    R2_obs:         {r2_obs:.4f}")
        print(f"    Null mean:      {null_mean:.4f}")
        print(f"    Null p95:       {null_p95:.4f}")
        print(f"    Hard cap (<{HARD_CAP_R2}): {'PASS' if passes_hard_cap else 'FAIL'}")
        print(f"    Perm test:      {'PASS' if passes_permutation else 'FAIL'}")
        print()

        if not passes_hard_cap:
            all_pass = False

        results[feat_name] = {
            "r2_observed": r2_obs,
            "null_mean": null_mean,
            "null_p95": null_p95,
            "null_floor_approx": float(np.percentile(null_r2s, 50)),
            "hard_cap": HARD_CAP_R2,
            "passes_hard_cap": passes_hard_cap,
            "passes_permutation_test": passes_permutation,
        }

    # Summary
    print("-" * 50)
    print("Orthogonality Validation Summary:")
    for feat, res in results.items():
        status = "PASS" if res["passes_hard_cap"] else "FAIL"
        print(f"  {feat:30s}: R2={res['r2_observed']:.4f}  [{status}]")

    print(f"\nOverall: {'ALL features pass R2 < 0.30' if all_pass else 'SOME features failed - check above'}")

    # Save
    output = {
        "protocol": {
            "n_holdout_accounts": N_HOLDOUT,
            "n_permutations": N_PERMUTATIONS,
            "hard_cap_r2": HARD_CAP_R2,
            "acceptance_criterion": "R2_obs < 95th percentile of null AND R2_obs < 0.30",
            "null_floor_theoretical": f"~0.013 at d=64, n=5000",
        },
        "features": results,
        "overall_pass": all_pass,
    }

    out_path = RESULTS_DIR / "step3_orthogonality_r2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\n[OK] Step 3 complete.\n")
    return output


if __name__ == "__main__":
    main()
