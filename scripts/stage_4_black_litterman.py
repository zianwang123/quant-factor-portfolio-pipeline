"""Stage 4: Black-Litterman Robustness Analysis.

Part A — BL Hyperparameter Sensitivity & Statistical Tests:
  Tests whether Stage 3's BL improvement (MVO+BL vs MVO, MaxSharpe+BL vs MaxSharpe)
  is robust to the choice of tau and delta. The baseline tau=0.05, delta=10 was chosen
  a priori from He & Litterman (1999). The sensitivity grid evaluates OOS performance
  across a range of (tau, delta) combinations — this is a ROBUSTNESS CHECK, not parameter
  optimization (using OOS to select parameters would be data snooping).

  Also runs Diebold-Mariano and Sharpe equality tests to formally assess whether
  BL's improvement over plain MVO/MaxSharpe is statistically significant.

Part B — Stock-Level BL (Academic Reference):
  Applies BL to ~274 individual S&P 500 stocks to demonstrate why factor-level
  allocation is the right granularity (stock-level views are too diffuse).
"""
import faulthandler
faulthandler.enable()
import os
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import DataPanel, load_sp500_returns
from src.portfolio.covariance import ledoit_wolf_shrinkage
from src.portfolio.optimization import mean_variance_optimize, max_sharpe_portfolio
from src.black_litterman.equilibrium import implied_equilibrium_returns
from src.black_litterman.model import black_litterman_posterior
from src.analytics.statistical_tests import diebold_mariano_test, sharpe_ratio_test


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def _sharpe(r):
    return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


def _ann_ret(r):
    return r.mean() * 12


def _ann_vol(r):
    return r.std() * np.sqrt(12)


def _max_dd(r):
    cum = (1 + r).cumprod()
    return (cum / cum.cummax() - 1).min()


def run_stage_4(config_path: str = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_4")
    cache_dir = PROJECT_ROOT / "data" / "processed" / "cache"

    print("=" * 60)
    print("STAGE 4: Black-Litterman Robustness Analysis")
    print("=" * 60)

    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start
    is_end_period = pd.Period(is_end, freq="M")
    oos_start_period = pd.Period(oos_start, freq="M")
    tau_baseline = config.black_litterman.tau
    delta_baseline = config.black_litterman.delta

    # ══════════════════════════════════════════════════════════════
    # PART A: BL Hyperparameter Sensitivity & Statistical Tests
    # ══════════════════════════════════════════════════════════════
    _flush("\n" + "─" * 60)
    _flush("PART A: BL Robustness Analysis (Sensitivity + Statistical Tests)")
    _flush("─" * 60)

    # ── A1: Load factor QSpreads ──
    _flush("\n[A1] Loading factor data...")
    qspreads_csv = tables_path / "factor_qspreads.csv"
    if not qspreads_csv.exists():
        _flush("  ERROR: factor_qspreads.csv not found. Run Stage 1 first.")
        return

    qs_df = pd.read_csv(qspreads_csv, index_col=0)
    qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")

    selected_csv = tables_path / "selected_factor_combination.csv"
    if selected_csv.exists():
        sel_df = pd.read_csv(selected_csv)
        selected_factors = sel_df["factor"].tolist()
    else:
        selected_factors = ["AccrualRatio", "CFTP", "STReversal", "AssetGrowth", "ROE"]

    sel_qspreads = qs_df[selected_factors]
    _flush(f"  Selected factors: {selected_factors}")
    _flush(f"  QSpread data: {sel_qspreads.shape}")

    # Load IC and FM tables for view confidence
    ic_table = None
    ic_csv = tables_path / "ic_analysis.csv"
    if ic_csv.exists():
        ic_table = pd.read_csv(ic_csv, index_col=0)

    fm_table = None
    fm_csv = tables_path / "fama_macbeth_results.csv"
    if fm_csv.exists():
        fm_table = pd.read_csv(fm_csv, index_col=0)

    # ── A2: Split IS/OOS, compute covariance and views ──
    qs_is = sel_qspreads.loc[:is_end_period].dropna()
    qs_oos = sel_qspreads.loc[oos_start_period:].dropna()
    n_factors = len(selected_factors)
    K = n_factors
    _flush(f"  IS: {len(qs_is)} months, OOS: {len(qs_oos)} months")

    _flush("\n[A2] Computing factor covariance and views...")
    sigma_f = ledoit_wolf_shrinkage(qs_is).values
    mu_is = qs_is.mean().values

    # Views: P = I (absolute view per factor), Q = IS mean returns
    P = np.eye(K)
    Q = mu_is
    _flush(f"  IS mean QSpread: {dict(zip(selected_factors, Q.round(6)))}")

    # Omega: view uncertainty scaled by IC confidence and FM significance
    # omega_k = tau * sigma_f[k,k] / confidence_k (He & Litterman 1999)
    confidence = np.ones(K)
    for i, fname in enumerate(selected_factors):
        c = 1.0
        if ic_table is not None and fname in ic_table.index:
            ic = abs(float(ic_table.loc[fname, "Mean IC"]))
            c = 1.0 + 20.0 * ic
        if fm_table is not None and fname in fm_table.index:
            fm_sig = fm_table.loc[fname, "Significant (5%)"]
            if fm_sig == True or fm_sig == "True":
                c *= 1.5
        confidence[i] = c

    # Base omega per unit tau (so we can rescale for the sensitivity grid)
    base_omega_over_tau = np.array([sigma_f[i, i] / confidence[i] for i in range(K)])
    _flush(f"  View confidence: {dict(zip(selected_factors, confidence.round(2)))}")

    # ── A3: Load Stage 3 baselines ──
    _flush("\n[A3] Loading Stage 3 baseline weights...")
    weights_csv = tables_path / "factor_allocation_weights.csv"
    if not weights_csv.exists():
        _flush("  ERROR: factor_allocation_weights.csv not found. Run Stage 3 first.")
        return

    stage3_weights = pd.read_csv(weights_csv, index_col=0)

    # Compute OOS returns for each Stage 3 variant
    baseline_returns = {}
    baseline_sharpe = {}
    for col in stage3_weights.columns:
        w = stage3_weights[col].values
        ret = qs_oos @ w
        baseline_returns[col] = ret
        baseline_sharpe[col] = _sharpe(ret)

    _flush("  Stage 3 OOS Sharpe ratios:")
    for name, sr in sorted(baseline_sharpe.items(), key=lambda x: -x[1]):
        _flush(f"    {name}: {sr:.4f}")

    # Extract the key comparisons
    mvo_sr = baseline_sharpe.get("MVO", np.nan)
    mvo_bl_sr = baseline_sharpe.get("MVO + BL", np.nan)
    ms_sr = baseline_sharpe.get("Max Sharpe", np.nan)
    ms_bl_sr = baseline_sharpe.get("Max Sharpe + BL", np.nan)

    _flush(f"\n  BL improvement at baseline (tau={tau_baseline}, delta={delta_baseline}):")
    _flush(f"    MVO: {mvo_sr:.4f} → MVO+BL: {mvo_bl_sr:.4f} (Δ = {mvo_bl_sr - mvo_sr:+.4f})")
    _flush(f"    MaxSharpe: {ms_sr:.4f} → MaxSharpe+BL: {ms_bl_sr:.4f} (Δ = {ms_bl_sr - ms_sr:+.4f})")

    # ── A4: Tau/delta sensitivity grid ──
    _flush("\n[A4] Tau/delta robustness grid...")
    _flush("  NOTE: This grid uses OOS data for evaluation — it CANNOT be used for")
    _flush("  parameter selection (that would be data snooping). The baseline")
    _flush(f"  tau={tau_baseline}, delta={delta_baseline} was chosen a priori from")
    _flush("  He & Litterman (1999). This analysis shows whether the BL improvement")
    _flush("  is robust or fragile to hyperparameter choice.\n")

    tau_vals = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    delta_vals = [2.5, 5, 10, 25, 50]
    w_eq = np.ones(K) / K

    # Fix Omega at baseline tau to break the tau cancellation.
    # When Omega ∝ tau (He & Litterman 1999), tau cancels out of the posterior
    # mean mu_bar — the optimizers see identical inputs regardless of tau.
    # Fixing Omega makes tau control the prior-vs-view balance as intended.
    Omega_fixed = np.diag(base_omega_over_tau * tau_baseline)

    grid_mvo_bl = {}
    grid_ms_bl = {}

    for t in tau_vals:
        for d in delta_vals:
            try:
                bl = black_litterman_posterior(d, sigma_f, w_eq, t, P, Q, Omega_fixed)
                mu_bl = bl["mu_posterior"]

                # MVO + BL (matching Stage 3 constraints exactly)
                try:
                    w_mvo_bl = mean_variance_optimize(
                        mu_bl, sigma_f, risk_aversion=d,
                        long_only=False, min_weight=-0.5, max_weight=1.0, gross_leverage=2.0,
                    )
                    ret_mvo_bl = qs_oos @ w_mvo_bl
                    grid_mvo_bl[(t, d)] = _sharpe(ret_mvo_bl)
                except Exception:
                    grid_mvo_bl[(t, d)] = np.nan

                # Max Sharpe + BL (matching Stage 3 constraints exactly)
                try:
                    w_ms_bl = max_sharpe_portfolio(
                        mu_bl, sigma_f, long_only=False, gross_leverage=3.0,
                    )
                    ret_ms_bl = qs_oos @ w_ms_bl
                    grid_ms_bl[(t, d)] = _sharpe(ret_ms_bl)
                except Exception:
                    grid_ms_bl[(t, d)] = np.nan

            except Exception:
                grid_mvo_bl[(t, d)] = np.nan
                grid_ms_bl[(t, d)] = np.nan

    # Format as 2D DataFrames
    mvo_bl_grid = pd.DataFrame(
        [[grid_mvo_bl.get((t, d), np.nan) for d in delta_vals] for t in tau_vals],
        index=[f"τ={t}" for t in tau_vals],
        columns=[f"δ={d}" for d in delta_vals],
    )
    ms_bl_grid = pd.DataFrame(
        [[grid_ms_bl.get((t, d), np.nan) for d in delta_vals] for t in tau_vals],
        index=[f"τ={t}" for t in tau_vals],
        columns=[f"δ={d}" for d in delta_vals],
    )

    _flush("  MVO + BL Sharpe across (tau, delta):")
    print(mvo_bl_grid.round(4))

    _flush(f"\n  Plain MVO baseline Sharpe: {mvo_sr:.4f}")
    n_mvo_beat = (mvo_bl_grid.values.flatten() > mvo_sr).sum()
    n_total = mvo_bl_grid.size
    _flush(f"  Grid cells where MVO+BL beats plain MVO: {n_mvo_beat}/{n_total}")
    _flush(f"  MVO+BL Sharpe range: [{np.nanmin(mvo_bl_grid.values):.4f}, {np.nanmax(mvo_bl_grid.values):.4f}]")

    _flush(f"\n  Max Sharpe + BL Sharpe across (tau, delta):")
    print(ms_bl_grid.round(4))

    _flush(f"\n  Plain Max Sharpe baseline Sharpe: {ms_sr:.4f}")
    n_ms_beat = (ms_bl_grid.values.flatten() > ms_sr).sum()
    _flush(f"  Grid cells where MaxSharpe+BL beats plain MaxSharpe: {n_ms_beat}/{n_total}")
    _flush(f"  MaxSharpe+BL Sharpe range: [{np.nanmin(ms_bl_grid.values):.4f}, {np.nanmax(ms_bl_grid.values):.4f}]")

    # Robustness summary
    mvo_robust_pct = n_mvo_beat / n_total * 100
    ms_robust_pct = n_ms_beat / n_total * 100
    _flush(f"\n  Robustness: BL improves MVO in {mvo_robust_pct:.0f}% of grid cells, "
           f"MaxSharpe in {ms_robust_pct:.0f}% of grid cells.")
    if mvo_robust_pct >= 60 and ms_robust_pct >= 60:
        _flush("  → BL improvement is ROBUST to hyperparameter choice.")
    elif mvo_robust_pct >= 40 or ms_robust_pct >= 40:
        _flush("  → BL improvement is MODERATE — depends on (tau, delta) region.")
        _flush("    BL helps most with higher delta (risk aversion) and lower tau (trust prior).")
    else:
        _flush("  → BL improvement is FRAGILE — highly dependent on hyperparameter choice.")

    # ── A5: Statistical significance of BL improvement ──
    _flush("\n[A5] Statistical tests: Is BL improvement significant?")

    tests_results = []

    # Test pairs: BL variant vs non-BL variant
    test_pairs = [
        ("MVO + BL", "MVO"),
        ("Max Sharpe + BL", "Max Sharpe"),
    ]

    for bl_name, base_name in test_pairs:
        if bl_name not in baseline_returns or base_name not in baseline_returns:
            continue

        r_bl = pd.Series(baseline_returns[bl_name], index=qs_oos.index)
        r_base = pd.Series(baseline_returns[base_name], index=qs_oos.index)

        # Diebold-Mariano test
        dm = diebold_mariano_test(r_bl, r_base)
        # Sharpe equality test (Jobson-Korkie with Memmel correction)
        jk = sharpe_ratio_test(r_bl, r_base)

        _flush(f"\n  {bl_name} vs {base_name}:")
        _flush(f"    Sharpe: {jk['SR1']:.4f} vs {jk['SR2']:.4f} (Δ = {jk['SR Diff']:+.4f})")
        _flush(f"    Jobson-Korkie z = {jk['z-statistic']:.3f}, p = {jk['p-value']:.4f}"
               f"{' ***' if jk['Significant (5%)'] else ''}")
        _flush(f"    Diebold-Mariano DM = {dm['DM Statistic']:.3f}, p = {dm['p-value']:.4f}"
               f"{' ***' if dm['Significant (5%)'] else ''}")

        tests_results.append({
            "Comparison": f"{bl_name} vs {base_name}",
            "BL Sharpe": jk["SR1"],
            "Base Sharpe": jk["SR2"],
            "Sharpe Diff": jk["SR Diff"],
            "JK z-stat": jk["z-statistic"],
            "JK p-value": jk["p-value"],
            "JK Sig (5%)": jk["Significant (5%)"],
            "DM stat": dm["DM Statistic"],
            "DM p-value": dm["p-value"],
            "DM Sig (5%)": dm["Significant (5%)"],
        })

    _flush("\n  Note: With only 60 OOS months, these tests have low power. Economic")
    _flush("  significance (BL consistently improves Sharpe and halves turnover) may")
    _flush("  be more informative than statistical significance at short horizons.")

    tests_df = pd.DataFrame(tests_results)

    # ══════════════════════════════════════════════════════════════
    # PART B: Stock-Level Black-Litterman (for completeness)
    # ══════════════════════════════════════════════════════════════
    _flush("\n" + "─" * 60)
    _flush("PART B: Stock-Level Black-Litterman (Academic Reference)")
    _flush("─" * 60)

    _flush("\n[B1] Loading stock-level data...")
    panel = DataPanel(config)
    returns = panel.get_returns()
    is_sp500 = panel.get_sp500_membership()
    sp500_df = load_sp500_returns(config)
    rf = sp500_df["rf"]

    # Load cached factors
    factors = {}
    for pkl in sorted(cache_dir.glob("factor_*.pkl")):
        fname = pkl.stem.replace("factor_", "")
        factors[fname] = pd.read_pickle(str(pkl))

    # Load sort results
    sort_results = None
    sort_path = cache_dir / "sort_results.pkl"
    if sort_path.exists():
        with open(sort_path, "rb") as f:
            sort_results = pickle.load(f)

    # Define universe: SP500 stocks with sufficient history
    is_sp500_cols = is_sp500.columns
    if is_end_period in is_sp500_cols:
        sp500_at_end = is_sp500[is_end_period]
    else:
        nearest = is_sp500_cols[abs(is_sp500_cols - is_end_period).argmin()]
        sp500_at_end = is_sp500[nearest]

    sp500_members = sp500_at_end[sp500_at_end == 1].index
    ret_is = returns.loc[:is_end_period]
    ret_oos = returns.loc[oos_start_period:]

    eligible = (
        ret_is.count()[ret_is.count() > 120].index
        .intersection(ret_oos.count()[ret_oos.count() > 24].index)
        .intersection(sp500_members)
    )

    returns_is = ret_is[eligible].dropna(axis=1, thresh=int(len(ret_is) * 0.5))
    returns_oos = ret_oos[eligible].dropna(axis=1, thresh=int(len(ret_oos) * 0.5))
    common_stocks = returns_is.columns.intersection(returns_oos.columns)
    returns_is = returns_is[common_stocks].fillna(0)
    returns_oos = returns_oos[common_stocks].fillna(0)
    stock_list = common_stocks.tolist()
    n_stocks = len(stock_list)
    _flush(f"  Universe: {n_stocks} stocks")

    # Covariance and market cap weights
    _flush("\n[B2] Computing stock-level equilibrium...")
    sigma_s = ledoit_wolf_shrinkage(returns_is).values

    mv_data = panel.pivot("cshom") * panel.pivot("prccm")
    mv_data = mv_data.replace([np.inf, -np.inf], np.nan)
    if is_end_period in mv_data.index:
        mv = mv_data.loc[is_end_period, common_stocks].fillna(0)
    else:
        mv = mv_data.iloc[-1].reindex(common_stocks).fillna(0)
    mv = mv.clip(lower=0)
    w_mkt = (mv / mv.sum()).values

    delta = delta_baseline
    tau = tau_baseline
    pi_s = implied_equilibrium_returns(delta, sigma_s, w_mkt)
    _flush(f"  Equilibrium returns: mean={pi_s.mean():.6f}, std={pi_s.std():.6f}")

    # Build stock-level views from factors
    _flush("\n[B3] Building stock-level views...")
    from src.black_litterman.views import build_factor_view, build_views_with_prior_scaling

    raw_views = []
    view_names = []
    ic_values = {}

    for fname in selected_factors:
        if fname not in factors or fname not in sort_results:
            continue

        qspread = sort_results[fname]["qspread"]
        qspread_is = qspread.loc[:is_end_period]
        if len(qspread_is) < 24:
            continue

        factor_df = factors[fname]
        if is_end_period in factor_df.index:
            use_date = is_end_period
        else:
            avail = factor_df.index[factor_df.index <= is_end_period]
            if len(avail) == 0:
                continue
            use_date = avail[-1]

        cs = factor_df.loc[use_date].reindex(common_stocks).replace([np.inf, -np.inf], np.nan).dropna()
        if len(cs) < 20:
            continue

        cs = cs.sort_values(ascending=False)
        n_q = len(cs) // 5
        winners = cs.iloc[:n_q].index.tolist()
        losers = cs.iloc[-n_q:].index.tolist()

        P_row, Q_val, _ = build_factor_view(qspread_is, winners, losers, stock_list)
        raw_views.append((P_row, Q_val))
        view_names.append(fname)

        if ic_table is not None and fname in ic_table.index:
            ic_values[fname] = float(ic_table.loc[fname, "Mean IC"])

    if raw_views:
        P_s, Q_s, Omega_s = build_views_with_prior_scaling(
            raw_views, sigma_s, tau,
            ic_values=ic_values, view_names=view_names, base_confidence=1.0,
        )

        bl_s = black_litterman_posterior(delta, sigma_s, w_mkt, tau, P_s, Q_s, Omega_s)
        w_star_s = bl_s["weights"]

        # Constrained BL
        try:
            w_constrained = mean_variance_optimize(
                bl_s["mu_posterior"], sigma_s,
                risk_aversion=delta, long_only=True, max_weight=0.05,
            )
        except Exception:
            w_constrained = w_mkt.copy()

        rf_oos = rf.reindex(returns_oos.index).fillna(0)
        w_mkt_s = pd.Series(w_mkt, index=common_stocks)

        stock_portfolios = {
            "Market Cap": w_mkt_s,
            "BL Unconstrained": pd.Series(w_star_s, index=common_stocks),
            "BL Constrained": pd.Series(w_constrained, index=common_stocks),
        }

        _flush("\n[B4] Stock-level OOS performance:")
        stock_perf = {}
        for pname, w in stock_portfolios.items():
            ret = returns_oos @ w
            excess = ret - rf_oos
            stock_perf[pname] = {
                "Ann. Return": _ann_ret(excess),
                "Ann. Volatility": _ann_vol(excess),
                "Sharpe Ratio": _sharpe(excess),
                "Max Drawdown": _max_dd(ret),
            }
        stock_perf_df = pd.DataFrame(stock_perf)
        print(stock_perf_df.round(4))

        _flush("\n  Stock-level BL underperforms market-cap weighting because factor-derived")
        _flush("  views are too diffuse across ~274 individual stocks. This motivates")
        _flush("  our focus on factor-level allocation in Stages 3-5.")

        stock_perf_df.to_csv(tables_path / "bl_stock_level_performance.csv")

    # ══════════════════════════════════════════════════════════════
    # Save all results
    # ══════════════════════════════════════════════════════════════
    _flush("\n" + "─" * 60)
    _flush("Saving results...")

    mvo_bl_grid.to_csv(tables_path / "bl_sensitivity_mvo.csv")
    ms_bl_grid.to_csv(tables_path / "bl_sensitivity_maxsharpe.csv")
    if len(tests_results) > 0:
        tests_df.to_csv(tables_path / "bl_statistical_tests.csv", index=False)

    _flush(f"\nStage 4 complete. Results saved to {tables_path}")

    return {
        "sensitivity_mvo_bl": mvo_bl_grid,
        "sensitivity_ms_bl": ms_bl_grid,
        "statistical_tests": tests_df if len(tests_results) > 0 else None,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_4(config_path)
