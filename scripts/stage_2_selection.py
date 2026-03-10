"""Stage 2: Factor Selection Engine.

Applies Fama-MacBeth, IC analysis, and LASSO to identify factors
with genuine predictive power, then selects an optimal factor combination
using greedy forward selection with decorrelation constraints.
"""
import faulthandler
faulthandler.enable()
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import load_config
from src.data.loader import DataPanel
from src.factors.registry import build_all_factors
from src.factors.validation import QuintileSorter
from src.selection.selector import FactorSelector
from src.selection.information_coefficient import compute_ic_series, ic_decay_analysis
from src.visualization.factor_plots import plot_ic_time_series


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_2(config_path: str = None, factors: dict = None, qspreads: dict = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_2")

    print("=" * 60)
    print("STAGE 2: Factor Selection Engine")
    print("=" * 60); _flush()

    # Load data
    print("\n[1/4] Loading data..."); _flush()
    panel = DataPanel(config)
    returns = panel.get_returns()
    is_sp500 = panel.get_sp500_membership()

    if factors is None:
        print("  Building factors from scratch (excluding Beta)..."); _flush()
        factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])

    # Free raw data to reduce memory pressure
    panel._raw = None
    gc.collect()
    print("  Raw data freed"); _flush()

    # Build QSpreads if not provided
    if qspreads is None:
        print("  Running quintile sorts for QSpreads..."); _flush()
        sorter = QuintileSorter(n_bins=config.factors.quintile_bins)
        sort_results = sorter.sort_all_factors(factors, returns, is_sp500)
        qspreads = {name: res["qspread"] for name, res in sort_results.items()}

    # Run selection
    print("\n[2/4] Running factor selection methods..."); _flush()
    selector = FactorSelector(config)
    results = selector.run_all(returns, factors, is_sp500, qspreads=qspreads)

    # *** INTERMEDIATE DUMP: save selection results immediately ***
    print("\n[3/4] Saving results..."); _flush()
    results["fama_macbeth"].to_csv(tables_path / "fama_macbeth_results.csv")
    results["ic_analysis"].to_csv(tables_path / "ic_analysis.csv")
    results["consensus_scores"].to_frame("Score").to_csv(tables_path / "factor_consensus_ranking.csv")

    selected = results["selected_factors"]
    selected_df = pd.DataFrame({
        "factor": selected,
        "score": [results["consensus_scores"].get(f, 0) for f in selected],
    })
    selected_df.to_csv(tables_path / "selected_factor_combination.csv", index=False)
    print("  CSVs saved"); _flush()

    # Report results
    print("\n--- Fama-MacBeth Results ---")
    print(results["fama_macbeth"].round(4)); _flush()

    print("\n--- IC Analysis ---")
    print(results["ic_analysis"].round(4)); _flush()

    print("\n--- LASSO Selection ---")
    lasso = results["lasso"]
    print(f"  Selected factors: {lasso['selected_factors']}")
    print(f"  Alpha: {lasso.get('alpha', 'N/A')}"); _flush()

    print("\n--- Consensus Ranking ---")
    print(results["consensus_scores"]); _flush()

    print(f"\n--- Selected Factor Combination ---")
    print(f"  Factors: {selected}"); _flush()

    # QSpread correlation among selected
    if len(selected) > 1:
        sel_spreads = pd.DataFrame({k: v for k, v in qspreads.items() if k in selected})
        if not sel_spreads.empty:
            print("\n  Pairwise QSpread correlations among selected:")
            print(sel_spreads.corr().round(3)); _flush()

    # Plots skipped in main run to avoid segfaults
    print("\n[4/4] Plots skipped (run plot_all.py separately if needed)"); _flush()

    print(f"\nStage 2 complete. Results saved to {tables_path} and {fig_path}"); _flush()
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_2(config_path)
