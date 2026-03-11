"""Run all stages sequentially as separate processes.

Each stage runs in its own Python process to avoid memory accumulation.
Data is passed between stages via saved CSV files.
Each run gets a timestamped output folder.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Generate timestamped run ID
run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Run ID: {run_id}")
print(f"All outputs will be saved to: outputs/{run_id}/")

# Pass run_id to child processes via environment variable
env = os.environ.copy()
env["PIPELINE_RUN_ID"] = run_id

MAX_RETRIES = 5


def run_script(name: str, script: str, retries: int = MAX_RETRIES, timeout: int = 600):
    """Run a script with retry logic for segfaults."""
    for attempt in range(1, retries + 1):
        print(f"\n{'='*60}")
        print(f"Running {name}... (attempt {attempt}/{retries})")
        print(f"{'='*60}\n", flush=True)

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True
        elif result.returncode in (3221225477, 139, -11):
            print(f"\nWARNING: {name} segfaulted (exit code {result.returncode}), retrying...")
        else:
            print(f"\nERROR: {name} failed with exit code {result.returncode}")
            return False

    print(f"\nERROR: {name} failed after {retries} attempts")
    return False


# ── Core pipeline stages (must succeed) ──
stages = [
    ("Stage 1: Factor Construction", "scripts/stage_1_factors.py"),
    ("Stage 2: Factor Selection", "scripts/stage_2_selection.py"),
    ("Stage 3: Portfolio Optimization", "scripts/stage_3_optimize.py"),
]

for name, script in stages:
    if not run_script(name, script):
        sys.exit(1)

# ── Plotting stages (warnings only on failure) ──
plot_stages = [
    ("Stage 3 Comparison Plots", "scripts/plot_comparison.py", 120),
    ("Individual Fund Plots", "scripts/plot_individual_funds.py", 120),
    ("Factor-Level Plots", "scripts/plot_factors.py", 180),
]

for name, script, timeout in plot_stages:
    if not run_script(name, script, retries=1, timeout=timeout):
        print(f"WARNING: {name} failed, continuing...")

print(f"\n{'='*60}")
print(f"ALL STAGES COMPLETE — Run ID: {run_id}")
print(f"Results in: outputs/{run_id}/")
print("=" * 60)
