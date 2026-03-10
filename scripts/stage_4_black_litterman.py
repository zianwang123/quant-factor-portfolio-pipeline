"""Stage 4: Black-Litterman View Integration.

Incorporates factor-based views into equilibrium returns
using the Black-Litterman framework. Supports multiple views
and sensitivity analysis.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import DataPanel
from src.factors.registry import build_all_factors
from src.factors.validation import QuintileSorter
from src.portfolio.covariance import ledoit_wolf_shrinkage
from src.black_litterman.equilibrium import implied_equilibrium_returns, market_cap_weights
from src.black_litterman.views import build_factor_view, build_multi_view
from src.black_litterman.model import black_litterman_posterior
from src.black_litterman.sensitivity import tau_delta_grid, view_impact_analysis
from src.analytics.performance import compute_descriptive_stats
from src.analytics.statistical_tests import diebold_mariano_test


def run_stage_4(config_path: str = None, factors: dict = None, sort_results: dict = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    print("=" * 60)
    print("STAGE 4: Black-Litterman View Integration")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    panel = DataPanel(config)
    returns = panel.get_returns()
    is_sp500 = panel.get_sp500_membership()

    if factors is None:
        factors = build_all_factors(panel, config)

    # Define universe: S&P 500 members at in-sample end with sufficient history
    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start

    # Get market cap weights
    raw = panel.raw
    mv_data = panel.pivot("cshom") * panel.pivot("prccm")  # Market value proxy
    mv_data = mv_data.replace([np.inf, -np.inf], np.nan)

    # Filter universe
    if is_end in is_sp500.columns:
        sp500_members = is_sp500[is_sp500[is_end] == 1].index
    else:
        # Fallback: use last available date
        last_date = is_sp500.columns[-1]
        sp500_members = is_sp500[is_sp500[last_date] == 1].index

    # Require sufficient return history
    ret_count_pre = returns.loc[:is_end].count()
    ret_count_post = returns.loc[oos_start:].count()
    eligible = ret_count_pre[ret_count_pre > 200].index.intersection(
        ret_count_post[ret_count_post > 50].index
    ).intersection(sp500_members)

    print(f"  Universe: {len(eligible)} stocks")

    # Prepare matrices
    returns_is = returns.loc[:is_end, eligible].dropna(axis=1)
    returns_oos = returns.loc[oos_start:, eligible].dropna(axis=1)

    # Ensure same columns
    common_stocks = returns_is.columns.intersection(returns_oos.columns)
    returns_is = returns_is[common_stocks]
    returns_oos = returns_oos[common_stocks]
    stock_list = common_stocks.tolist()
    n_stocks = len(stock_list)
    print(f"  Final universe: {n_stocks} stocks")

    # Covariance and market cap weights
    print("\n[2/5] Estimating covariance and equilibrium returns...")
    sigma = ledoit_wolf_shrinkage(returns_is).values

    # Market cap weights at in-sample end
    if is_end in mv_data.index:
        mv = mv_data.loc[is_end, common_stocks].fillna(0)
    else:
        mv = mv_data.iloc[-1].reindex(common_stocks).fillna(0)

    w_mkt = (mv / mv.sum()).values
    delta = config.black_litterman.delta
    tau = config.black_litterman.tau

    pi = implied_equilibrium_returns(delta, sigma, w_mkt)
    print(f"  Equilibrium returns: mean={pi.mean():.4f}, std={pi.std():.4f}")

    # Build views from selected factors
    print("\n[3/5] Building factor views...")

    if sort_results is None:
        sorter = QuintileSorter()
        sort_results = sorter.sort_all_factors(factors, returns, is_sp500)

    # Build views for factors with significant QSpread
    view_factors = ["HL1M", "MOM", "BP"]  # Factors most likely to have signal
    views = []

    for fname in view_factors:
        if fname not in sort_results:
            continue

        qspread = sort_results[fname]["qspread"].loc[:is_end]
        if qspread.mean() == 0 or len(qspread) < 12:
            continue

        # Identify winners/losers at in-sample end
        factor_df = factors[fname]
        if is_end not in factor_df.index:
            continue

        cs = factor_df.loc[is_end].reindex(common_stocks).dropna()
        cs = cs.replace([np.inf, -np.inf], np.nan).dropna()
        cs = cs.sort_values(ascending=False)

        n_q = len(cs) // 5
        winners = cs.iloc[:n_q].index.tolist()
        losers = cs.iloc[-n_q:].index.tolist()

        P_row, Q_val, omega_val = build_factor_view(
            qspread, winners, losers, stock_list
        )
        views.append((P_row, Q_val, omega_val))
        print(f"  {fname}: Q={Q_val:.4f}, omega={omega_val:.4f}, "
              f"long={len(winners)}, short={len(losers)}")

    if not views:
        print("  No valid views found. Using equilibrium only.")
        return

    P, Q, Omega = build_multi_view(views)
    print(f"  Combined: {P.shape[0]} views x {P.shape[1]} stocks")

    # Run BL model
    print("\n[4/5] Computing BL posterior...")
    bl = black_litterman_posterior(delta, sigma, w_mkt, tau, P, Q, Omega)

    w_star = bl["weights"]
    mu_post = bl["mu_posterior"]

    print(f"  Prior returns: mean={pi.mean():.4f}")
    print(f"  Posterior returns: mean={mu_post.mean():.4f}")
    print(f"  Max weight change: {np.max(np.abs(w_star - w_mkt)):.6f}")

    # View impact analysis
    impact = view_impact_analysis(
        sigma, w_mkt, tau, delta, P, Q, Omega,
        stock_names=stock_list, n_top=5,
    )

    print("\n  Most Bullish Stocks:")
    print(impact["bullish"].round(6))
    print("\n  Most Bearish Stocks:")
    print(impact["bearish"].round(6))

    # Out-of-sample evaluation
    print("\n[5/5] Out-of-sample evaluation...")

    w_mkt_s = pd.Series(w_mkt, index=common_stocks)
    w_star_s = pd.Series(w_star, index=common_stocks)

    oos_mkt_ret = returns_oos @ w_mkt_s
    oos_bl_ret = returns_oos @ w_star_s

    perf = pd.DataFrame({
        "Market Cap Weighted": {
            "Mean (monthly)": oos_mkt_ret.mean(),
            "Std (monthly)": oos_mkt_ret.std(),
            "Sharpe (monthly)": oos_mkt_ret.mean() / oos_mkt_ret.std(),
            "Annualized Sharpe": oos_mkt_ret.mean() / oos_mkt_ret.std() * np.sqrt(12),
        },
        "Black-Litterman": {
            "Mean (monthly)": oos_bl_ret.mean(),
            "Std (monthly)": oos_bl_ret.std(),
            "Sharpe (monthly)": oos_bl_ret.mean() / oos_bl_ret.std(),
            "Annualized Sharpe": oos_bl_ret.mean() / oos_bl_ret.std() * np.sqrt(12),
        },
    })
    print("\n  Out-of-Sample Performance:")
    print(perf.round(6))

    # Diebold-Mariano test
    dm = diebold_mariano_test(oos_bl_ret, oos_mkt_ret)
    print(f"\n  Diebold-Mariano test: stat={dm['DM Statistic']:.4f}, p={dm['p-value']:.4f}")

    # Sensitivity analysis
    print("\n  Running sensitivity grid...")
    grid = tau_delta_grid(
        sigma, w_mkt, P, Q, Omega,
        tau_values=[0.01, 0.1, 0.5, 1.0],
        delta_values=[5, 10, 25, 50],
        stock_names=stock_list,
    )

    # Save results
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_4")

    perf.to_csv(tables_path / "bl_oos_performance.csv")
    impact["bullish"].to_csv(tables_path / "bl_bullish_stocks.csv")
    impact["bearish"].to_csv(tables_path / "bl_bearish_stocks.csv")
    if not grid["weights"].empty:
        grid["weights"].to_csv(tables_path / "bl_sensitivity_weights.csv")

    print(f"\nStage 4 complete. Results saved to {tables_path} and {fig_path}")

    return {
        "bl_result": bl,
        "impact": impact,
        "performance": perf,
        "sensitivity": grid,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_4(config_path)
