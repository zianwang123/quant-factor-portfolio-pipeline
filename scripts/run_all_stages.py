"""Run all 3 stages sequentially as separate processes.

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

stages = [
    ("Stage 1: Factor Construction", "scripts/stage_1_factors.py"),
    ("Stage 2: Factor Selection", "scripts/stage_2_selection.py"),
    ("Stage 3: Portfolio Optimization", "scripts/stage_3_optimize.py"),
]

MAX_RETRIES = 5

for name, script in stages:
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"Running {name}... (attempt {attempt}/{MAX_RETRIES})")
        print(f"{'='*60}\n", flush=True)

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=600,
        )

        if result.returncode == 0:
            break
        elif result.returncode in (3221225477, 139, -11):
            # Segfault — retry
            print(f"\nWARNING: {name} segfaulted (exit code {result.returncode}), retrying...")
        else:
            print(f"\nERROR: {name} failed with exit code {result.returncode}")
            sys.exit(1)
    else:
        print(f"\nERROR: {name} failed after {MAX_RETRIES} attempts")
        sys.exit(1)

# Run comparison plots
print(f"\n{'='*60}")
print("Generating comparison plots...")
print(f"{'='*60}\n", flush=True)

result = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "plot_comparison.py")],
    cwd=str(PROJECT_ROOT),
    env=env,
    timeout=120,
)
if result.returncode != 0:
    print(f"WARNING: Plot generation failed (exit code {result.returncode})")

# Run individual fund plots
result = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "plot_individual_funds.py")],
    cwd=str(PROJECT_ROOT),
    env=env,
    timeout=120,
)
if result.returncode != 0:
    print(f"WARNING: Individual fund plots failed (exit code {result.returncode})")

print(f"\n{'='*60}")
print(f"ALL STAGES COMPLETE — Run ID: {run_id}")
print(f"Results in: outputs/{run_id}/")
print("=" * 60)
