"""
TMIL-ETH — Master Pipeline Runner
===================================
Runs all 12 steps in sequence.
Skips steps that have already produced output files.

Usage:
  python run_all.py              # run all steps
  python run_all.py --from 5    # start from step 5
  python run_all.py --step 9    # run only step 9
  python run_all.py --force     # re-run even if output exists
"""

import sys
import argparse
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent / "results"

# ── Step → output file (used to skip if already done) ─────────────────────
STEP_OUTPUTS = {
    1:  RESULTS_DIR / "step1_dataset_stats.json",
    2:  RESULTS_DIR / "step2_features.pkl",
    3:  RESULTS_DIR / "step3_orthogonality_r2.json",
    4:  RESULTS_DIR / "step4_windows_stats.json",
    5:  None,   # model definition, no output file
    6:  None,   # loss definition, no output file
    7:  RESULTS_DIR / "step7_training_curves.json",
    8:  RESULTS_DIR / "step8_sidak_thresholds.json",
    9:  RESULTS_DIR / "step9_nested_cv_results.json",
    10: RESULTS_DIR / "step10_baselines.json",
    11: RESULTS_DIR / "step11_localization_metrics.json",
    12: RESULTS_DIR / "step12_ablation_results.json",
}

# ── Step → script name ─────────────────────────────────────────────────────
STEP_SCRIPTS = {
    1:  "step01_dataset_analysis.py",
    2:  "step02_feature_extraction.py",
    3:  "step03_orthogonality_validation.py",
    4:  "step04_sliding_window.py",
    5:  "step05_model_architecture.py",
    6:  "step06_compound_loss.py",
    7:  "step07_training.py",
    8:  "step08_sidak_correction.py",
    9:  "step09_nested_cv.py",
    10: "step10_baselines.py",
    11: "step11_forensic_localization.py",
    12: "step12_ablation_study.py",
}

STEP_DESCRIPTIONS = {
    1:  "Dataset Preparation & Analysis",
    2:  "Feature Extraction (BERT + Heuristics)",
    3:  "Orthogonality Validation (Linear Probing)",
    4:  "Sliding Window Formulation",
    5:  "Model Architecture (Gated Attention MIL)",
    6:  "Phish-Masked Contrastive Loss Demo",
    7:  "Two-Phase Training",
    8:  "Sidak FPR Correction",
    9:  "Nested Stratified CV (Main Results)",
    10: "Baseline Comparison (RF, GBM, Bi-LSTM, BERT4ETH)",
    11: "Forensic Localization (Hit@1 + Heuristics)",
    12: "Ablation Study (AUC + Hit@1, 25-epoch)",
}

# Critical dependencies: if these fail, stop the pipeline
CRITICAL_STEPS = {2, 4, 7, 9}


def run_step(step_num: int, force: bool = False) -> bool:
    script  = STEP_SCRIPTS.get(step_num)
    out_file = STEP_OUTPUTS.get(step_num)
    desc    = STEP_DESCRIPTIONS.get(step_num, "")

    if script is None:
        print(f"\n  Step {step_num:02d}: No script — skipping.")
        return True

    if out_file and out_file.exists() and not force:
        print(f"\n  Step {step_num:02d} [{desc}]: Output exists — skipping.")
        return True

    print(f"\n{'='*60}")
    print(f"  Step {step_num:02d}: {desc}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-X", "utf8", script],
        cwd=Path(__file__).parent,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Step {step_num:02d} failed (exit code {result.returncode})")
        return False

    print(f"\n  Step {step_num:02d} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TMIL-ETH 12-Step Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this specific step")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    args = parser.parse_args()

    print("=" * 60)
    print("TMIL-ETH: Transaction-level MIL for Ethereum Phishing")
    print("12-Step Master Pipeline")
    print("=" * 60)

    print("\nPipeline Overview:")
    for n, desc in STEP_DESCRIPTIONS.items():
        print(f"  Step {n:02d}: {desc}")
    print()

    if args.step is not None:
        steps_to_run = [args.step]
    else:
        steps_to_run = list(range(args.from_step, 13))

    results = {}
    for step in steps_to_run:
        success = run_step(step, force=args.force)
        results[step] = "OK" if success else "FAILED"
        if not success and step in CRITICAL_STEPS:
            print(f"\n  Stopping: Step {step:02d} is a critical dependency.")
            break

    print("\n" + "=" * 60)
    print("Pipeline Summary:")
    for step, status in results.items():
        icon = "[OK]    " if status == "OK" else "[FAILED]"
        print(f"  {icon} Step {step:02d}: {STEP_DESCRIPTIONS[step]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
