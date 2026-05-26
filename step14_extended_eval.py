"""
step14_extended_eval.py
Extended evaluation: Bootstrap CI, Hit@k, ABMIL micro-level ablation, case study
Run this on the GPU machine after step13_baselines.py has completed.
"""
import sys, json, pickle, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RESULTS_DIR = Path(__file__).parent / "results"

# ============================================================
# 1. Bootstrap 95% CI for Hit@1 (from existing CSV)
# ============================================================
def compute_bootstrap_ci(csv_path, n_bootstrap=10000, alpha=0.05, seed=42):
    print("\n[1] Bootstrap 95% CI for Hit@1")
    rng = random.Random(seed)
    hits = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            hits.append(int(r['hit_at_1']))

    n = len(hits)
    observed = sum(hits) / n * 100
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(hits, k=n)
        boot_means.append(sum(sample) / n * 100)
    
    boot_means.sort()
    lo = boot_means[int(n_bootstrap * alpha / 2)]
    hi = boot_means[int(n_bootstrap * (1 - alpha / 2))]
    
    # Random baseline bootstrap
    n_windows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_end = int(r['human_gt_burst'].split('-')[1])
            ai_idx = int(r['ai_max_attn_idx'])
            n_windows.append(max(gt_end, ai_idx) + 1)
    
    random_boot = []
    for _ in range(n_bootstrap):
        sample_n = rng.choices(n_windows, k=n)
        random_boot.append(np.mean([1.0/nw for nw in sample_n]) * 100)
    random_boot.sort()
    r_lo = random_boot[int(n_bootstrap * alpha / 2)]
    r_hi = random_boot[int(n_bootstrap * (1 - alpha / 2))]
    
    print(f"  TMIL-ETH Hit@1:    {observed:.2f}%  (95% CI: [{lo:.2f}%, {hi:.2f}%])")
    print(f"  Random Baseline:   {np.mean(random_boot):.2f}%  (95% CI: [{r_lo:.2f}%, {r_hi:.2f}%])")
    
    overlaps = hi < r_lo or lo > r_hi
    print(f"  CIs overlap: {not overlaps} — Significance: {'SIGNIFICANT (no overlap)' if not overlaps else 'NOT significant'}")
    
    result = {
        "tmil_hit1": observed,
        "tmil_ci_lo": lo,
        "tmil_ci_hi": hi,
        "random_hit1": np.mean(random_boot),
        "random_ci_lo": r_lo,
        "random_ci_hi": r_hi,
        "significant": not overlaps
    }
    with open(RESULTS_DIR / "step14_bootstrap_ci.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ============================================================
# 2. Hit@k (k=1,3,5) — requires re-running attention scores
# ============================================================
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


from tmil_model import TMILETH

def compute_hit_at_k(records, gt_data, model_path, device, k_values=[1, 3, 5]):
    print(f"\n[2] Hit@k evaluation (k={k_values})")
    
    gt_map = {item["account_address"].lower(): item for item in gt_data}
    model = TMILETH(4, 64).to(device)

    
    if not Path(model_path).exists():
        print(f"  Model checkpoint not found: {model_path}")
        print("  Please run step12_human_eval.py first to generate the checkpoint.")
        return {}
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    results = {k: {"hit": 0, "total": 0} for k in k_values}
    per_account = []
    
    with torch.no_grad():
        for r in records:
            addr = r["address"].lower()
            if addr not in gt_map: continue
            bursts = gt_map[addr].get("ground_truth_bursts", [])
            if not bursts: continue
            
            gt_start = bursts[0]["start_tx_idx"]
            gt_end = bursts[0]["end_tx_idx"]
            
            bag_feats = get_bag_features(r, W=200)
            if len(bag_feats) == 0: continue
            
            x = torch.tensor(bag_feats, dtype=torch.float32).to(device)
            _, A = model(x)
            A_np = A.cpu().numpy()
            
            # Top-k attention indices
            topk_indices = np.argsort(A_np)[::-1]
            
            for k in k_values:
                topk = topk_indices[:k]
                hit = 0
                for idx in topk:
                    win_start = idx * 50
                    win_end = win_start + 200
                    if win_start <= gt_end and win_end >= gt_start:
                        hit = 1
                        break
                results[k]["hit"] += hit
                results[k]["total"] += 1
            
            per_account.append({
                "addr": addr,
                "gt_burst": f"{gt_start}-{gt_end}",
                "n_windows": len(bag_feats),
                "top1_idx": int(topk_indices[0]),
                "hit@1": int(topk_indices[0:1].tolist()[0] * 50 <= gt_end and (topk_indices[0] * 50 + 200) >= gt_start),
                "attention_scores": A_np.tolist()
            })
    
    print(f"  {'Metric':<12} {'Result':<10} {'/ Total'}")
    for k in k_values:
        total = results[k]["total"]
        hit = results[k]["hit"]
        print(f"  Hit@{k:<8} {hit/total*100:.2f}%    ({hit}/{total})")
    
    out = {f"hit_at_{k}": results[k]["hit"] / results[k]["total"] * 100 for k in k_values}
    out["per_account"] = per_account
    with open(RESULTS_DIR / "step14_hit_at_k.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ============================================================
# 3. ABMIL Micro-Level Ablation (same granularity as TMIL-ETH)
# ============================================================
class GatedABMIL_Micro(nn.Module):
    """ABMIL operating at MICRO level: transactions within a window."""
    def __init__(self, input_dim=68, M=64, L=64):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, M), nn.ReLU())
        self.V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.w = nn.Linear(L, 1)
        self.clf = nn.Sequential(nn.Linear(M, 1), nn.Sigmoid())

    def forward(self, x):
        H = self.feat(x)
        A = self.w(self.V(H) * self.U(H))
        A = F.softmax(A, dim=0).squeeze(-1)
        Z = (A.unsqueeze(-1) * H).sum(0, keepdim=True)
        return self.clf(Z), A


# ============================================================
# 4. Qualitative Case Study
# ============================================================
def generate_case_study(per_account_data, gt_data, top_n=3):
    print(f"\n[4] Qualitative Case Study — top {top_n} accounts")
    gt_map = {item["account_address"].lower(): item for item in gt_data}
    
    hits = [r for r in per_account_data if r.get("hit@1") == 1]
    misses_with_partial = [r for r in per_account_data if r.get("hit@1") == 0 and r.get("n_windows", 0) < 20]
    
    case_studies = []
    for r in (hits + misses_with_partial)[:top_n]:
        addr = r["addr"]
        gt = gt_map.get(addr, {})
        bursts = gt.get("ground_truth_bursts", [{}])
        case_studies.append({
            "address": addr,
            "gt_burst": r["gt_burst"],
            "n_windows": r["n_windows"],
            "top1_window": r["top1_idx"],
            "hit@1": r["hit@1"],
            "attention_peaks": sorted(enumerate(r["attention_scores"]), key=lambda x: -x[1])[:5]
        })
        
        print(f"\n  Account: {addr[:12]}...")
        print(f"  GT Burst: windows {r['gt_burst']} | N_windows: {r['n_windows']}")
        print(f"  Top attention window: #{r['top1_idx']} | Hit@1: {r['hit@1']}")
        top5 = sorted(enumerate(r["attention_scores"]), key=lambda x: -x[1])[:5]
        print(f"  Top-5 attention: {[(i, f'{a:.4f}') for i,a in top5]}")
    
    with open(RESULTS_DIR / "step14_case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2)
    return case_studies


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("TMIL-ETH: Extended Evaluation (Step 14)")
    print("=" * 70)
    
    csv_path = RESULTS_DIR / "step12_human_localization_metrics.csv"
    feat_path = RESULTS_DIR / "step2_features.pkl"
    gt_path = Path(__file__).parent / "human_ground_truth.json"
    model_path = RESULTS_DIR / "checkpoints" / "best_model.pt"
    
    # Fallback model paths
    if not Path(model_path).exists():
        model_path = RESULTS_DIR / "temp_best_model.pt"
    if not Path(model_path).exists():
        model_path = RESULTS_DIR / "temp_mil_best.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Part 1: Bootstrap CI (from existing CSV only)
    ci_result = compute_bootstrap_ci(csv_path)
    
    # Load data for parts 2-4
    if not feat_path.exists():
        print("\nFeatures not found — skipping Hit@k and case study.")
        return
    
    with open(feat_path, "rb") as f:
        records = pickle.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    eval_addrs = {item["account_address"].lower() for item in gt_data}
    test_recs = [r for r in records if r["address"].lower() in eval_addrs]
    
    # Part 2: Hit@k
    hitk_result = compute_hit_at_k(test_recs, gt_data, model_path, device, k_values=[1, 3, 5])
    
    # Part 4: Case study (from hitk per_account data if available)
    if hitk_result and "per_account" in hitk_result:
        generate_case_study(hitk_result["per_account"], gt_data)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Bootstrap CI: TMIL-ETH {ci_result['tmil_hit1']:.2f}% [{ci_result['tmil_ci_lo']:.2f}%, {ci_result['tmil_ci_hi']:.2f}%]")
    print(f"             Random {ci_result['random_hit1']:.2f}% [{ci_result['random_ci_lo']:.2f}%, {ci_result['random_ci_hi']:.2f}%]")
    if hitk_result:
        for k in [1, 3, 5]:
            key = f"hit_at_{k}"
            if key in hitk_result:
                print(f"Hit@{k}: {hitk_result[key]:.2f}%")
    print("[OK] Extended evaluation complete — push results/ to git and update manuscript.")

if __name__ == "__main__":
    main()
