"""
TMIL-ETH — Master Pipeline Runner
===================================
Runs all 10 steps in sequence.
Skips steps that have already produced output files.

Usage:
  python run_all.py            # run all steps
  python run_all.py --from 3  # start from step 3
  python run_all.py --step 7  # run only step 7
"""

import sys
import argparse
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = Path(__file__).parent / "results"


def step_outputs():
    """Map step number to expected output file (used to skip if already done)."""
    return {
        1: RESULTS_DIR / "step1_dataset_stats.json",
        2: RESULTS_DIR / "step2_features.pkl",
        3: RESULTS_DIR / "step3_orthogonality_r2.json",
        4: RESULTS_DIR / "step4_windows_stats.json",
        5: None,  # model code — always runnable
        6: None,  # part of model code
        7: RESULTS_DIR / "step7_training_curves.json",
        8: RESULTS_DIR / "step8_sidak_thresholds.json",
        9: RESULTS_DIR / "step9_nested_cv_results.json",
        10: RESULTS_DIR / "step10_ablation_table.csv",
    }


STEP_SCRIPTS = {
    1: "step1_dataset_analysis.py",
    2: "step2_feature_extraction.py",
    3: "step3_orthogonality_validation.py",
    4: "step4_sliding_window.py",
    5: "tmil_model.py",
    6: None,   # Embedded in tmil_model.py
    7: "step7_two_phase_training.py",
    8: "step8_sidak_correction.py",
    9: "step9_nested_cv_cpu.py",
    10: "step10_ablation_interpretability.py",
}


def run_step(step_num: int, force: bool = False) -> bool:
    outputs = step_outputs()
    script  = STEP_SCRIPTS.get(step_num)

    if script is None:
        print(f"\n  Step {step_num}: Embedded in model code — skipping separate run.")
        return True

    out_file = outputs.get(step_num)
    if out_file and out_file.exists() and not force:
        print(f"\n  Step {step_num}: Output exists ({out_file.name}) — skipping.")
        return True

    print(f"\n{'='*60}")
    print(f"  Running Step {step_num}: {script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-X", "utf8", script],
        cwd=Path(__file__).parent,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Step {step_num} failed (exit code {result.returncode})")
        return False

    print(f"\n  Step {step_num} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="TMIL-ETH Pipeline Runner")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this specific step")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    args = parser.parse_args()

    print("TMIL-ETH: Transaction-level MIL for Ethereum Phishing Detection")
    print("Master Pipeline Runner")
    print("=" * 60)

    if args.step is not None:
        steps_to_run = [args.step]
    else:
        steps_to_run = list(range(args.from_step, 11))

    results = {}
    for step in steps_to_run:
        success = run_step(step, force=args.force)
        results[step] = "OK" if success else "FAILED"
        if not success and step in [2, 3, 7, 9, 10]:  # critical dependencies
            print(f"\n  Stopping: Step {step} is a critical dependency for later steps.")
            break

    print("\n" + "=" * 60)
    print("Pipeline Summary:")
    for step, status in results.items():
        icon = "[OK]" if status == "OK" else "[FAILED]"
        print(f"  Step {step}: {icon}")
    print("=" * 60)


if __name__ == "__main__":
    main()
