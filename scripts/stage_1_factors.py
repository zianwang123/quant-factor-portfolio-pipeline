"""Stage 1: Factor Construction & Validation.

Builds all 20 equity factors from Compustat/CRSP data, validates against
Capital IQ benchmarks, and produces comprehensive factor analytics.

Split into sub-stages with intermediate saves to survive segfaults.
"""
import faulthandler
faulthandler.enable()
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.config import load_config
from src.data.loader import DataPanel
from src.factors.registry import build_all_factors, validate_all_factors
from src.factors.validation import QuintileSorter
from src.analytics.performance import (
    compute_descriptive_stats, t_test_factor_spreads, compute_turnover_stats,
)


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_1(config_path: str = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_1")
    cache_dir = PROJECT_ROOT / "data" / "processed" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Factor Construction & Validation")
    print("=" * 60)
    _flush()

    # ── Step 1: Load data ──
    _flush("\n[1/6] Loading data...")
    panel = DataPanel(config)
    returns = panel.get_returns()
    is_sp500 = panel.get_sp500_membership()
    market_excess = panel.get_market_excess()
    capital_iq = panel.capital_iq
    _flush(f"  Data loaded: returns {returns.shape}")

    # Save returns and is_sp500 for potential sub-process use
    returns.to_pickle(str(cache_dir / "returns.pkl"))
    is_sp500.to_pickle(str(cache_dir / "is_sp500.pkl"))
    _flush("  Cached returns and SP500 membership")

    # ── Step 2: Build factors ──
    _flush("\n[2/6] Constructing factors...")
    factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])
    _flush(f"  Built {len(factors)} factors")

    # Save each factor to disk
    for fname, fdf in factors.items():
        fdf.to_pickle(str(cache_dir / f"factor_{fname}.pkl"))
    _flush(f"  Cached {len(factors)} factor DataFrames to disk")

    # Free raw CSV data
    panel._raw = None
    gc.collect()
    _flush("  Raw data freed")

    # ── Step 3: Quintile sorts (one factor at a time, with gc) ──
    _flush("\n[3/6] Running quintile sorts...")
    sorter = QuintileSorter(n_bins=config.factors.quintile_bins)

    sort_results = {}
    for name in list(factors.keys()):
        _flush(f"  Sorting {name}...")
        result = sorter.sort_single_factor(factors[name], returns, is_sp500)
        sort_results[name] = result
        gc.collect()

    # *** INTERMEDIATE DUMP: save QSpreads and sort results immediately ***
    qspreads = pd.DataFrame({name: res["qspread"] for name, res in sort_results.items()})
    qspreads.to_csv(tables_path / "s1_factor_qspreads.csv")
    _flush(f"  QSpreads saved: {qspreads.shape}")

    import pickle
    with open(cache_dir / "sort_results.pkl", "wb") as f:
        pickle.dump(sort_results, f)
    _flush("  Cached sort_results.pkl for plotting")

    # ── Step 4: Validate against Capital IQ ──
    _flush("\n[4/6] Validating against Capital IQ benchmarks...")
    validation = validate_all_factors(
        factors, {n: r["qspread"] for n, r in sort_results.items()},
        capital_iq, config.dates.validation_start, config.dates.end,
    )
    print(validation)
    validation.to_csv(tables_path / "s1_factor_benchmark_correlations.csv")
    _flush()

    # ── Step 5: Analytics ──
    _flush("\n[5/6] Computing analytics...")

    qspreads_pct = qspreads * 100
    sp500_excess_pct = market_excess * 100
    qspreads_pct["SP500"] = sp500_excess_pct.reindex(qspreads_pct.index)

    desc_stats = compute_descriptive_stats(qspreads_pct)
    print("\nDescriptive Statistics:")
    print(desc_stats.round(4))
    desc_stats.to_csv(tables_path / "s1_factor_descriptive_stats.csv")
    _flush()

    t_tests = t_test_factor_spreads(qspreads)
    print("\nt-Test for Mean QSpread != 0:")
    print(t_tests.round(4))
    t_tests.to_csv(tables_path / "s1_factor_t_tests.csv")
    _flush()

    turnover = {name: res["turnover"] for name, res in sort_results.items()}
    turnover_stats = compute_turnover_stats(turnover)
    print("\nAverage Turnover:")
    print(turnover_stats.round(4))
    _flush()

    print("\nIC analysis: deferred to Stage 2")
    print("\n[6/6] Plots skipped (run plot_all.py separately if needed)")

    _flush(f"\nStage 1 complete. Results saved to {tables_path} and {fig_path}")
    return {
        "factors": factors,
        "sort_results": sort_results,
        "qspreads": qspreads,
        "validation": validation,
        "desc_stats": desc_stats,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_1(config_path)
