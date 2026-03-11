"""Stage 2: Factor Selection Engine.

Applies Fama-MacBeth, IC analysis, and LASSO to identify factors
with genuine predictive power, then selects an optimal factor combination
using greedy forward selection with decorrelation constraints.

Loads cached factors and QSpreads from Stage 1 to avoid redundant computation.
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


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_2(config_path: str = None, factors: dict = None, qspreads: dict = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_2")
    cache_dir = PROJECT_ROOT / "data" / "processed" / "cache"

    print("=" * 60)
    print("STAGE 2: Factor Selection Engine")
    print("=" * 60)
    _flush()

    # ── Load data ──
    _flush("\n[1/4] Loading data...")

    # Try to load cached factors from Stage 1
    if factors is None and (cache_dir / "factor_HL1M.pkl").exists():
        _flush("  Loading cached factors from Stage 1...")
        factors = {}
        for pkl in sorted(cache_dir.glob("factor_*.pkl")):
            fname = pkl.stem.replace("factor_", "")
            factors[fname] = pd.read_pickle(str(pkl))
        _flush(f"  Loaded {len(factors)} cached factors")
    elif factors is None:
        _flush("  No cache found, building factors from scratch...")
        panel = DataPanel(config)
        factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])
        panel._raw = None
        gc.collect()
        _flush(f"  Built {len(factors)} factors")

    # Load cached returns and SP500 membership
    if (cache_dir / "returns.pkl").exists():
        _flush("  Loading cached returns...")
        returns = pd.read_pickle(str(cache_dir / "returns.pkl"))
        is_sp500 = pd.read_pickle(str(cache_dir / "is_sp500.pkl"))
    else:
        _flush("  Loading returns from raw data...")
        panel = DataPanel(config)
        returns = panel.get_returns()
        is_sp500 = panel.get_sp500_membership()
        panel._raw = None
        gc.collect()

    _flush(f"  Returns: {returns.shape}, Factors: {len(factors)}")

    # Load QSpreads from Stage 1 CSV if available
    if qspreads is None:
        qspreads_csv = config.find_prior_output("factor_qspreads.csv")
        if qspreads_csv is not None:
            _flush(f"  Loading QSpreads from {qspreads_csv.parent.parent.name}...")
            qs_df = pd.read_csv(qspreads_csv, index_col=0)
            qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")
            qspreads = {col: qs_df[col] for col in qs_df.columns}
            _flush(f"  Loaded {len(qspreads)} QSpread series")
        else:
            _flush("  Running quintile sorts for QSpreads...")
            sorter = QuintileSorter(n_bins=config.factors.quintile_bins)
            sort_results = sorter.sort_all_factors(factors, returns, is_sp500)
            qspreads = {name: res["qspread"] for name, res in sort_results.items()}

    # ── Run selection ──
    _flush("\n[2/4] Running factor selection methods...")
    selector = FactorSelector(config)
    results = selector.run_all(returns, factors, is_sp500, qspreads=qspreads)

    # ── Save results ──
    _flush("\n[3/4] Saving results...")
    results["fama_macbeth"].to_csv(tables_path / "fama_macbeth_results.csv")
    results["ic_analysis"].to_csv(tables_path / "ic_analysis.csv")
    results["consensus_scores"].to_frame("Score").to_csv(tables_path / "factor_consensus_ranking.csv")

    selected = results["selected_factors"]
    selected_df = pd.DataFrame({
        "factor": selected,
        "score": [results["consensus_scores"].get(f, 0) for f in selected],
    })
    selected_df.to_csv(tables_path / "selected_factor_combination.csv", index=False)

    # Save IC series for factor plots
    if "ic_series" in results:
        import pickle
        with open(cache_dir / "ic_series.pkl", "wb") as f:
            pickle.dump(results["ic_series"], f)
        _flush("  Cached ic_series.pkl for plotting")

    _flush("  CSVs saved")

    # ── Report ──
    print("\n--- Fama-MacBeth Results ---")
    print(results["fama_macbeth"].round(4))
    _flush()

    print("\n--- IC Analysis ---")
    print(results["ic_analysis"].round(4))
    _flush()

    print("\n--- LASSO Selection ---")
    lasso = results["lasso"]
    print(f"  Selected factors: {lasso['selected_factors']}")
    print(f"  Alpha: {lasso.get('alpha', 'N/A')}")
    _flush()

    print("\n--- Consensus Ranking ---")
    print(results["consensus_scores"])
    _flush()

    print(f"\n--- Selected Factor Combination ---")
    print(f"  Factors: {selected}")
    _flush()

    if len(selected) > 1:
        sel_spreads = pd.DataFrame({k: v for k, v in qspreads.items() if k in selected})
        if not sel_spreads.empty:
            print("\n  Pairwise QSpread correlations among selected:")
            print(sel_spreads.corr().round(3))
            _flush()

    print("\n[4/4] Plots skipped (run plot_all.py separately if needed)")
    _flush(f"\nStage 2 complete. Results saved to {tables_path} and {fig_path}")
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_2(config_path)
