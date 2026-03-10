"""Run Stages 1 and 2 together, passing data between them."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage_1_factors import run_stage_1
from scripts.stage_2_selection import run_stage_2

config_path = str(PROJECT_ROOT / "config" / "pipeline.yaml")

# Stage 1
s1 = run_stage_1(config_path)

# Stage 2 — pass factors and qspreads to avoid recomputation
s2 = run_stage_2(
    config_path,
    factors=s1["factors"],
    qspreads={n: r["qspread"] for n, r in s1["sort_results"].items()},
)
