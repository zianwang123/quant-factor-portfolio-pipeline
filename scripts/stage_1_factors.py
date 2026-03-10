"""Stage 1: Factor Construction & Validation.

Builds all 20 equity factors from Compustat/CRSP data, validates against
Capital IQ benchmarks, and produces comprehensive factor analytics.

Dumps intermediate results to CSV after each major step so progress
is never lost to intermittent segfaults.
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
from src.visualization.factor_plots import (
    plot_qspread_vs_benchmark, plot_long_short_legs,
    plot_quintile_monotonicity, plot_cumulative_qspread_vs_market,
    plot_factor_correlation_heatmap, plot_ic_time_series,
)
from src.selection.information_coefficient import compute_ic_series, ic_summary


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_1(config_path: str = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_1")

    print("=" * 60)
    print("STAGE 1: Factor Construction & Validation")
    print("=" * 60); _flush()

    # ── Step 1: Load data ──
    print("\n[1/6] Loading data..."); _flush()
    panel = DataPanel(config)
    returns = panel.get_returns()
    is_sp500 = panel.get_sp500_membership()
    market_excess = panel.get_market_excess()
    capital_iq = panel.capital_iq
    print(f"  Data loaded: returns {returns.shape}"); _flush()

    # ── Step 2: Build factors ──
    print("\n[2/6] Constructing factors..."); _flush()
    factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])
    print(f"  Built {len(factors)} factors"); _flush()

    # Free raw CSV data — factors are already computed
    panel._raw = None
    gc.collect()
    print("  Raw data freed"); _flush()

    # ── Step 3: Quintile sorts ──
    print("\n[3/6] Running quintile sorts..."); _flush()
    sorter = QuintileSorter(n_bins=config.factors.quintile_bins)
    sort_results = sorter.sort_all_factors(factors, returns, is_sp500)

    # *** INTERMEDIATE DUMP: save QSpreads immediately ***
    qspreads = pd.DataFrame({name: res["qspread"] for name, res in sort_results.items()})
    qspreads.to_csv(tables_path / "factor_qspreads.csv")
    print(f"  QSpreads saved: {qspreads.shape}"); _flush()

    # ── Step 4: Validate against Capital IQ ──
    print("\n[4/6] Validating against Capital IQ benchmarks..."); _flush()
    validation = validate_all_factors(
        factors, {n: r["qspread"] for n, r in sort_results.items()},
        capital_iq, config.dates.validation_start, config.dates.end,
    )
    print(validation)
    validation.to_csv(tables_path / "factor_benchmark_correlations.csv")
    _flush()

    # ── Step 5: Analytics ──
    print("\n[5/6] Computing analytics..."); _flush()

    # Descriptive statistics (in percentage points for readability)
    qspreads_pct = qspreads * 100
    sp500_excess_pct = market_excess * 100
    qspreads_pct["SP500"] = sp500_excess_pct.reindex(qspreads_pct.index)

    desc_stats = compute_descriptive_stats(qspreads_pct)
    print("\nDescriptive Statistics:")
    print(desc_stats.round(4))
    desc_stats.to_csv(tables_path / "factor_descriptive_stats.csv")
    _flush()

    t_tests = t_test_factor_spreads(qspreads)
    print("\nt-Test for Mean QSpread != 0:")
    print(t_tests.round(4))
    t_tests.to_csv(tables_path / "factor_t_tests.csv")
    _flush()

    turnover = {name: res["turnover"] for name, res in sort_results.items()}
    turnover_stats = compute_turnover_stats(turnover)
    print("\nAverage Turnover:")
    print(turnover_stats.round(4))
    _flush()

    # IC Analysis skipped here — done in Stage 2 to avoid pandas segfaults
    print("\nIC analysis: deferred to Stage 2"); _flush()

    # Plots skipped in main run to avoid segfaults — run plot_all.py separately
    print("\n[6/6] Plots skipped (run plot_all.py separately if needed)"); _flush()

    print(f"\nStage 1 complete. Results saved to {tables_path} and {fig_path}"); _flush()
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
