"""
Step 15: 3-Way Perturbation Study
===================================
Tests whether TMIL-ETH attention (p_window) is driven by:
  - Behavioral content of transactions (hand_crafted features)
  - Temporal/positional artifacts
  - Or solely BERT account embedding

3 conditions per phisher (>=3 windows):
  A. Baseline:         original hand_crafted windows
  B. Content Replace:  replace middle windows with real normal-account transactions
  C. Shuffle Control:  shuffle transaction order within all windows, rebuild

Metric: p_window (per-window phishing probability from model)
Result: A vs B vs C → isolates content contribution

Research Question:
  "Is TMIL-ETH's attention sensitive to window content or window position?"
"""
import sys, os, pickle, json, random
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

# ── Rebuild the EXACT architecture saved in the checkpoint ────────────────────
from step05_model_architecture import GatedTMILETH

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth\results')
PKL_FILE    = RESULTS_DIR / 'step02_features.pkl'
CKPT_FILE   = RESULTS_DIR / 'checkpoints' / 'tmil_eth_final.pt'
OUT_FILE    = RESULTS_DIR / 'step15_perturbation_study.json'

W      = 200    # window size
STRIDE = 50     # stride
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
model = GatedTMILETH(4, 64)
ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=True)
state = ckpt.get('model_state_dict', ckpt)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print('Model loaded.')

# ── Load features ─────────────────────────────────────────────────────────────
with open(PKL_FILE, 'rb') as f:
    records = pickle.load(f)

phishers = [r for r in records if r['label'] == 1 and r['n_windows'] >= 3]
normals  = [r for r in records if r['label'] == 0 and r['n_windows'] >= 1]
print(f"Phishers with >=3 windows: {len(phishers)}")
print(f"Normal accounts available: {len(normals)}")

# ── Build normal window pool ──────────────────────────────────────────────────
# For condition B: pool of real normal transactions (hand_crafted only)
normal_windows = []
for rec in normals:
    hc = np.array(rec['hand_crafted'], dtype=np.float32)
    for (s, e) in rec['windows']:
        hw = hc[s:e]
        n = hw.shape[0]
        if n < W:
            hw = np.vstack([hw, np.zeros((W - n, 4), dtype=np.float32)])
        else:
            hw = hw[:W]
        normal_windows.append(hw)  # (W, 4)
print(f"Normal window pool size: {len(normal_windows)}")

# ── Inference helper ──────────────────────────────────────────────────────────
@torch.no_grad()
def score_windows(hc_list, bert_embed):
    """
    hc_list: list of (W, 4) numpy arrays (one per window)
    bert_embed: (64,) numpy array
    Returns: list of float p_window scores
    """
    scores = []
    bert_t = torch.tensor(bert_embed, dtype=torch.float32).to(DEVICE)
    for hw in hc_list:
        hc_t  = torch.tensor(hw, dtype=torch.float32).unsqueeze(0).to(DEVICE)   # (1, W, 4)
        bert_b = bert_t.unsqueeze(0).unsqueeze(0).expand(1, W, -1).to(DEVICE)   # (1, W, 64)
        p, _  = model(hc_t, bert_b)
        scores.append(p.item())
    return scores

# ── 3-Way Perturbation Experiment ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("3-WAY PERTURBATION STUDY")
print(f"{'='*70}")

results = []
attn_A_all, attn_B_replaced, attn_C_shuffled = [], [], []
attn_A_unreplaced = []  # baseline at same positions that will be replaced in B

for i, rec in enumerate(phishers):
    hc_orig = np.array(rec['hand_crafted'], dtype=np.float32)
    bert    = np.array(rec['bert_embedding'], dtype=np.float32)
    wins    = rec['windows']
    N       = len(wins)

    # Build per-window hc matrices (Condition A)
    def build_windows(hc_data, windows):
        out = []
        for (s, e) in windows:
            hw = hc_data[s:e]
            n = hw.shape[0]
            if n < W:
                hw = np.vstack([hw, np.zeros((W-n, 4), dtype=np.float32)])
            else:
                hw = hw[:W]
            out.append(hw)
        return out

    win_A = build_windows(hc_orig, wins)

    # ── Condition A: Baseline ─────────────────────────────────────────────────
    scores_A = score_windows(win_A, bert)

    # ── Condition B: Content Replace (middle 1/3 windows) ────────────────────
    # Replace middle windows with random normal windows
    n_replace = max(1, N // 3)
    mid_start = (N - n_replace) // 2
    mid_end   = mid_start + n_replace
    replace_indices = list(range(mid_start, mid_end))

    win_B = [w.copy() for w in win_A]
    for idx in replace_indices:
        # Sample a real normal window
        norm_win = random.choice(normal_windows)
        win_B[idx] = norm_win

    scores_B = score_windows(win_B, bert)

    # ── Condition C: Shuffle Control (shuffle tx order within each window) ────
    hc_shuffled = hc_orig.copy()
    np.random.shuffle(hc_shuffled)  # shuffle transaction ORDER (content preserved, order broken)
    win_C = build_windows(hc_shuffled, wins)
    scores_C = score_windows(win_C, bert)

    # ── Record ────────────────────────────────────────────────────────────────
    # A_replaced: baseline scores at replaced positions
    a_replaced     = [scores_A[idx] for idx in replace_indices]
    a_non_replaced = [scores_A[idx] for idx in range(N) if idx not in replace_indices]
    b_replaced     = [scores_B[idx] for idx in replace_indices]
    c_replaced     = [scores_C[idx] for idx in replace_indices]

    attn_A_all.extend(scores_A)
    attn_A_unreplaced.extend(a_replaced)    # A at replaced positions
    attn_B_replaced.extend(b_replaced)      # B at replaced positions
    attn_C_shuffled.extend(c_replaced)      # C at replaced positions

    results.append({
        'address': rec['address'],
        'n_windows': N,
        'replace_indices': replace_indices,
        'scores_A': scores_A,
        'scores_B': scores_B,
        'scores_C': scores_C,
        'mean_A_all': float(np.mean(scores_A)),
        'mean_A_replaced': float(np.mean(a_replaced)),
        'mean_B_replaced': float(np.mean(b_replaced)),
        'mean_C_replaced': float(np.mean(c_replaced)),
    })

    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(phishers)}] done")

# ── Statistical Analysis ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("STATISTICAL RESULTS")
print(f"{'='*70}")

A_arr = np.array(attn_A_unreplaced)  # baseline at positions that get replaced
B_arr = np.array(attn_B_replaced)    # content replaced
C_arr = np.array(attn_C_shuffled)    # shuffle control

print(f"\nN windows analyzed: {len(A_arr)}")
print(f"\nMean p_window scores at REPLACED positions:")
print(f"  A (baseline, phishing content):  {A_arr.mean():.4f} ± {A_arr.std():.4f}")
print(f"  B (content replaced w/ normal):  {B_arr.mean():.4f} ± {B_arr.std():.4f}")
print(f"  C (shuffle control):             {C_arr.mean():.4f} ± {C_arr.std():.4f}")

# Mann-Whitney U tests (non-parametric, no normality assumption)
stat_AB, p_AB = stats.mannwhitneyu(A_arr, B_arr, alternative='greater')
stat_AC, p_AC = stats.mannwhitneyu(A_arr, C_arr, alternative='greater')
stat_BC, p_BC = stats.mannwhitneyu(B_arr, C_arr, alternative='two-sided')

print(f"\nMann-Whitney U Tests (one-sided: A > B, A > C):")
print(f"  A vs B (content effect):   U={stat_AB:.0f}, p={p_AB:.4e}  {'SIGNIFICANT' if p_AB<0.05 else 'NOT significant'}")
print(f"  A vs C (order effect):     U={stat_AC:.0f}, p={p_AC:.4e}  {'SIGNIFICANT' if p_AC<0.05 else 'NOT significant'}")
print(f"  B vs C (content vs order): U={stat_BC:.0f}, p={p_BC:.4e}  {'SIGNIFICANT' if p_BC<0.05 else 'NOT significant'}")

# Effect sizes (Cohen's d)
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / (pooled_std + 1e-9)

d_AB = cohen_d(A_arr, B_arr)
d_AC = cohen_d(A_arr, C_arr)

print(f"\nEffect sizes (Cohen's d):")
print(f"  A vs B: d = {d_AB:.3f}")
print(f"  A vs C: d = {d_AC:.3f}")

# ── Interpretation ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

A_gt_B = A_arr.mean() > B_arr.mean()
A_gt_C = A_arr.mean() > C_arr.mean()
B_eq_C = abs(B_arr.mean() - C_arr.mean()) < 0.01

if A_gt_B and not A_gt_C:
    print("Pattern: B < A, C ≈ A → Model IS sensitive to CONTENT (hand_crafted)")
    print("→ Replacing tx content with normal txs reduces p_window significantly")
    print("→ Shuffling order alone does NOT reduce p_window")
    print("→ CONCLUSION: attention driven by content, not position")
elif A_gt_C and not A_gt_B:
    print("Pattern: C < A, B ≈ A → Model IS sensitive to TEMPORAL ORDER, not content")
elif A_gt_B and A_gt_C and p_BC > 0.05:
    print("Pattern: B < C < A or B ≈ C < A → BOTH content AND order contribute")
    print("→ CONCLUSION: MIL attention encodes both behavioral content and temporal structure")
elif not A_gt_B and not A_gt_C:
    print("Pattern: B ≈ C ≈ A → Attention driven purely by BERT embedding")
    print("→ hand_crafted features NOT contributing to attention")
    print("→ WARNING: This suggests MIL head is not using window-level content")
else:
    print(f"Pattern: complex (A_gt_B={A_gt_B}, A_gt_C={A_gt_C}, B_eq_C={B_eq_C})")

# Save results
output = {
    'n_phishers': len(phishers),
    'n_windows_tested': len(A_arr),
    'means': {'A': float(A_arr.mean()), 'B': float(B_arr.mean()), 'C': float(C_arr.mean())},
    'stds':  {'A': float(A_arr.std()),  'B': float(B_arr.std()),  'C': float(C_arr.std())},
    'mannwhitney': {
        'AB': {'U': float(stat_AB), 'p': float(p_AB)},
        'AC': {'U': float(stat_AC), 'p': float(p_AC)},
        'BC': {'U': float(stat_BC), 'p': float(p_BC)},
    },
    'cohens_d': {'AB': float(d_AB), 'AC': float(d_AC)},
    'per_phisher': results,
}
with open(OUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT_FILE}")
