"""Run all 3 stages sequentially as separate processes.

Each stage runs in its own Python process to avoid memory accumulation.
Data is passed between stages via saved CSV files.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

stages = [
    ("Stage 1: Factor Construction", "scripts/stage_1_factors.py"),
    ("Stage 2: Factor Selection", "scripts/stage_2_selection.py"),
    ("Stage 3: Portfolio Optimization", "scripts/stage_3_optimize.py"),
]

for name, script in stages:
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}\n", flush=True)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=str(PROJECT_ROOT),
        timeout=600,
    )

    if result.returncode != 0:
        print(f"\nERROR: {name} failed with exit code {result.returncode}")
        sys.exit(1)

print("\n" + "=" * 60)
print("ALL STAGES COMPLETE")
print("=" * 60)
