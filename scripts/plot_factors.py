"""Generate per-factor plots: QSpread vs benchmark, long/short legs,
quintile monotonicity, cumulative vs market, correlation heatmap.

Reads saved data from Stage 1/2 outputs (no recomputation needed).
"""
import os
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import load_config
from src.data.loader import load_sp500_returns, load_capital_iq
from src.visualization.factor_plots import (
    plot_qspread_vs_benchmark,
    plot_long_short_legs,
    plot_quintile_monotonicity,
    plot_cumulative_qspread_vs_market,
    plot_factor_correlation_heatmap,
    plot_ic_time_series,
)


def run_factor_plots(config_path: str = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_1")
    cache_dir = PROJECT_ROOT / "data" / "processed" / "cache"

    print("=" * 60)
    print("Generating Factor-Level Plots")
    print("=" * 60)

    # ── Load QSpreads from Stage 1 ──
    qspreads_csv = tables_path / "s1_factor_qspreads.csv"
    if not qspreads_csv.exists():
        print("ERROR: factor_qspreads.csv not found. Run Stage 1 first.")
        return
    qs_df = pd.read_csv(qspreads_csv, index_col=0)
    qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")
    print(f"  Loaded QSpreads: {qs_df.shape}")

    # ── Load sort results from cache (for long/short legs and quintile returns) ──
    sort_data_path = cache_dir / "sort_results.pkl"
    sort_results = None
    if sort_data_path.exists():
        with open(sort_data_path, "rb") as f:
            sort_results = pickle.load(f)
        print(f"  Loaded sort results: {len(sort_results)} factors")

    # ── Load Capital IQ benchmarks ──
    try:
        capital_iq = load_capital_iq(config)
        print(f"  Loaded Capital IQ benchmarks: {list(capital_iq.columns)}")
    except Exception:
        capital_iq = None
        print("  Capital IQ benchmarks not available")

    # ── Load market excess returns ──
    sp500_df = load_sp500_returns(config)
    market_excess = sp500_df["excess_return"]

    # ── Load IC series from Stage 2 ──
    ic_data_path = cache_dir / "ic_series.pkl"
    ic_series_dict = None
    if ic_data_path.exists():
        with open(ic_data_path, "rb") as f:
            ic_series_dict = pickle.load(f)
        print(f"  Loaded IC series: {len(ic_series_dict)} factors")

    # ── 1. Correlation heatmap (all factors) ──
    print("\n  Plotting correlation heatmap...")
    plot_factor_correlation_heatmap(qs_df, save_path=fig_path)

    # ── 2. Per-factor plots ──
    for factor_name in qs_df.columns:
        print(f"  Plotting {factor_name}...", end=" ", flush=True)
        qspread = qs_df[factor_name].dropna()
        plots_done = []

        # QSpread vs Capital IQ benchmark
        if capital_iq is not None and factor_name in capital_iq.columns:
            benchmark = capital_iq[factor_name].dropna()
            plot_qspread_vs_benchmark(qspread, benchmark, factor_name, save_path=fig_path)
            plots_done.append("benchmark")

        # Cumulative QSpread vs S&P 500
        plot_cumulative_qspread_vs_market(qspread, market_excess, factor_name, save_path=fig_path)
        plots_done.append("cumulative")

        # Long/short legs and quintile monotonicity (need sort_results)
        if sort_results and factor_name in sort_results:
            sr = sort_results[factor_name]
            if "long_return" in sr and "short_return" in sr:
                plot_long_short_legs(sr["long_return"], sr["short_return"], factor_name, save_path=fig_path)
                plots_done.append("long/short")
            if "quintile_returns" in sr:
                plot_quintile_monotonicity(sr["quintile_returns"], factor_name, save_path=fig_path)
                plots_done.append("quintiles")

        # IC time series (need IC data from Stage 2)
        if ic_series_dict and factor_name in ic_series_dict:
            ic = ic_series_dict[factor_name]
            if len(ic) > 12:
                plot_ic_time_series(ic, factor_name, save_path=fig_path)
                plots_done.append("IC")

        print(", ".join(plots_done))

    print(f"\nFactor plots saved to {fig_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_factor_plots(config_path)
