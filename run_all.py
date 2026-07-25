"""
run_all.py — Full pipeline runner
===================================
Executes all phases in order. Safe to re-run; each phase checks for
existing outputs and skips if already complete.

Usage:
  python run_all.py             # run all phases
  python run_all.py --phase 1   # run a specific phase only
  python run_all.py --from 3    # run from phase 3 onward
"""

import argparse
import subprocess
import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PHASES = [
    (1, "src/01_data_generator.py",      "Synthetic Data Generator"),
    (2, "src/02_feature_engineering.py", "Feature Engineering"),
    (3, "src/03_isolation_forest.py",    "Isolation Forest"),
    (4, "src/04_bilstm_autoencoder.py",  "BiLSTM Autoencoder (GPU)"),
    (5, "src/05_classifier.py",          "Anomaly Classifier"),
    (6, "src/06_explainability.py",      "Explainability (SHAP)"),
    (7, "src/07_cold_start_drift.py",    "Cold-Start & Concept Drift"),
]

CHECK_FILES = {
    1: "data/data_with_labels.csv",
    2: "data/features_tabular.csv",
    3: "data/if_scores.csv",
    4: "data/blended_scores.csv",
    5: "data/classified_alerts.csv",
    6: "data/alerts_with_explanations.csv",
    7: "data/cold_start_entities.csv",
}

def run_phase(phase_num, script, description, force=False):
    check = CHECK_FILES.get(phase_num)
    if check and os.path.exists(check) and not force:
        print(f"\n  [SKIP] Phase {phase_num} - {description}")
        print(f"         Output already exists: {check}")
        return True

    print(f"\n{'='*60}")
    print(f"  Phase {phase_num} - {description}")
    print(f"{'='*60}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, script], capture_output=False, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] Phase {phase_num} FAILED (exit code {result.returncode})")
        return False

    print(f"\n[OK] Phase {phase_num} complete ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="UEBA Pipeline Runner")
    parser.add_argument("--phase", type=int, help="Run only this phase")
    parser.add_argument("--from",  type=int, dest="from_phase", help="Run from this phase onward")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    print("\nAI-Powered Behavioral Anomaly Detection - Pipeline")
    print(f"    Python: {sys.version.split()[0]}")
    print(f"    Working dir: {os.getcwd()}")

    phases_to_run = PHASES
    if args.phase:
        phases_to_run = [(n, s, d) for n, s, d in PHASES if n == args.phase]
    elif args.from_phase:
        phases_to_run = [(n, s, d) for n, s, d in PHASES if n >= args.from_phase]

    success = True
    for phase_num, script, description in phases_to_run:
        ok = run_phase(phase_num, script, description, force=args.force)
        if not ok:
            print(f"\n[✗] Stopping at Phase {phase_num}. Fix the error and re-run with --from {phase_num}")
            success = False
            break

    if success:
        print("\n" + "="*60)
        print("  ✅  ALL PHASES COMPLETE")
        print("="*60)
        print("\n  To launch the dashboard:")
        print("    streamlit run dashboard/dashboard.py\n")

if __name__ == "__main__":
    main()
