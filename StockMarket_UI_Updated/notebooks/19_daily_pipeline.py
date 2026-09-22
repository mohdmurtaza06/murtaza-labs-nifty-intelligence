import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


# ============================================================
# PIPELINE
# ============================================================

scripts = [
    "14_live_data.py",
    "15_live_prediction.py",
    "16_prediction_tracker.py",
    "17_verify_predictions.py",
    "18_evaluate_model.py"
]


# ============================================================
# RUN EACH STEP
# ============================================================

print("=" * 70)
print("NIFTY 50 DAILY ML PIPELINE")
print("=" * 70)

for script in scripts:

    script_path = NOTEBOOKS_DIR / script

    print("\n")
    print("=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:

        print("\n")
        print("=" * 70)
        print(f"PIPELINE STOPPED: {script}")
        print("=" * 70)

        sys.exit(result.returncode)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("DAILY PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
