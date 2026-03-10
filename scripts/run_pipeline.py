"""Master pipeline runner.

Usage:
    python scripts/run_pipeline.py                          # Run all stages
    python scripts/run_pipeline.py --stages 1,2             # Run specific stages
    python scripts/run_pipeline.py --config custom.yaml     # Custom config
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Quant Factor-Portfolio Pipeline")
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated stage numbers to run (e.g., '1,2,3,4')")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "config" / "pipeline.yaml"),
                        help="Path to pipeline config YAML")
    args = parser.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(",")]

    print("=" * 60)
    print("  QUANT FACTOR-PORTFOLIO PIPELINE")
    print("  From Alpha Discovery to Portfolio Deployment")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Stages: {stages}")
    print("=" * 60)

    factors = None
    sort_results = None
    selection_results = None

    if 1 in stages:
        from scripts.stage_1_factors import run_stage_1
        stage1 = run_stage_1(args.config)
        factors = stage1["factors"]
        sort_results = stage1["sort_results"]

    if 2 in stages:
        from scripts.stage_2_selection import run_stage_2
        selection_results = run_stage_2(args.config, factors=factors)

    if 3 in stages:
        from scripts.stage_3_optimize import run_stage_3
        run_stage_3(args.config)

    if 4 in stages:
        from scripts.stage_4_black_litterman import run_stage_4
        run_stage_4(args.config, factors=factors, sort_results=sort_results)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
