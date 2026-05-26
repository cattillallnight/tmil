"""Quick bootstrap CI computation from existing CSV"""
import csv, random, json, numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
csv_path = RESULTS_DIR / "step12_human_localization_metrics.csv"

rng = random.Random(42)
hits = []
n_windows = []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        hits.append(int(r['hit_at_1']))
        gt_end = int(r['human_gt_burst'].split('-')[1])
        ai_idx = int(r['ai_max_attn_idx'])
        n_windows.append(max(gt_end, ai_idx) + 1)

n = len(hits)
observed = sum(hits) / n * 100

# Bootstrap TMIL-ETH CI
boot_means = sorted([sum(rng.choices(hits, k=n))/n*100 for _ in range(10000)])
lo = boot_means[250]
hi = boot_means[9750]

# Bootstrap random CI
random_boot = sorted([np.mean([1.0/nw for nw in rng.choices(n_windows, k=n)])*100 for _ in range(10000)])
r_lo = random_boot[250]
r_hi = random_boot[9750]
r_mean = np.mean(random_boot)

print(f"TMIL-ETH Hit@1: {observed:.2f}%  95% CI: [{lo:.2f}%, {hi:.2f}%]")
print(f"Random baseline: {r_mean:.2f}%  95% CI: [{r_lo:.2f}%, {r_hi:.2f}%]")
print(f"CIs overlap: {not (hi < r_lo or lo > r_hi)}")

result = {"tmil": observed, "tmil_lo": lo, "tmil_hi": hi,
          "random": r_mean, "random_lo": r_lo, "random_hi": r_hi}
with open(RESULTS_DIR / "step14_bootstrap_ci.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved step14_bootstrap_ci.json")
