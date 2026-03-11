"""Stage 4: Black-Litterman Factor Allocation.

Applies Black-Litterman to factor-level portfolio allocation:
- Prior: equal-weight factor allocation (market equilibrium for factors)
- Views: factor return forecasts from IC analysis and Fama-MacBeth
- Posterior: BL-optimal factor weights

This is a natural extension of Stage 3 — instead of ad-hoc factor weighting
(equal, IC-weighted, MVO), BL provides a Bayesian-principled allocation
that blends prior beliefs with factor-level signals.

Also runs a stock-level BL analysis for completeness.
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
from src.portfolio.optimization import mean_variance_optimize
from src.black_litterman.equilibrium import implied_equilibrium_returns
from src.black_litterman.model import black_litterman_posterior
from src.black_litterman.sensitivity import tau_delta_grid
from src.analytics.statistical_tests import diebold_mariano_test


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
    print("STAGE 4: Black-Litterman Factor Allocation")
    print("=" * 60)

    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start
    is_end_period = pd.Period(is_end, freq="M")
    oos_start_period = pd.Period(oos_start, freq="M")
    tau = config.black_litterman.tau
    delta = config.black_litterman.delta

    # ══════════════════════════════════════════════════════════════
    # PART A: Factor-Level Black-Litterman
    # ══════════════════════════════════════════════════════════════
    _flush("\n" + "─" * 60)
    _flush("PART A: Factor-Level Black-Litterman Allocation")
    _flush("─" * 60)

    # ── Load factor QSpreads ──
    _flush("\n[A1] Loading factor data...")
    qspreads_csv = tables_path / "factor_qspreads.csv"
    if not qspreads_csv.exists():
        _flush("  ERROR: factor_qspreads.csv not found. Run Stage 1 first.")
        return

    qs_df = pd.read_csv(qspreads_csv, index_col=0)
    qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")

    # Load selected factors
    selected_csv = tables_path / "selected_factor_combination.csv"
    if selected_csv.exists():
        sel_df = pd.read_csv(selected_csv)
        selected_factors = sel_df["factor"].tolist()
    else:
        selected_factors = ["AccrualRatio", "CFTP", "STReversal", "AssetGrowth", "ROE"]

    sel_qspreads = qs_df[selected_factors]
    _flush(f"  Selected factors: {selected_factors}")
    _flush(f"  QSpread data: {sel_qspreads.shape}")

    # Load IC analysis for views
    ic_table = None
    ic_csv = tables_path / "ic_analysis.csv"
    if ic_csv.exists():
        ic_table = pd.read_csv(ic_csv, index_col=0)

    # Load FM results for views
    fm_table = None
    fm_csv = tables_path / "fama_macbeth_results.csv"
    if fm_csv.exists():
        fm_table = pd.read_csv(fm_csv, index_col=0)

    # ── Split IS/OOS ──
    qs_is = sel_qspreads.loc[:is_end_period].dropna()
    qs_oos = sel_qspreads.loc[oos_start_period:].dropna()
    n_factors = len(selected_factors)
    _flush(f"  IS: {len(qs_is)} months, OOS: {len(qs_oos)} months")

    # ── Factor covariance (Ledoit-Wolf shrinkage) ──
    _flush("\n[A2] Computing factor covariance and equilibrium...")
    sigma_f = ledoit_wolf_shrinkage(qs_is).values
    _flush(f"  Factor covariance: {sigma_f.shape}")

    # Prior: equal-weight allocation (the "market portfolio" for factors)
    w_eq = np.ones(n_factors) / n_factors
    pi_f = implied_equilibrium_returns(delta, sigma_f, w_eq)
    _flush(f"  Equilibrium returns (equal-weight prior): {dict(zip(selected_factors, pi_f.round(6)))}")

    # ── Build views from IC and FM analysis ──
    _flush("\n[A3] Building factor views...")

    # View 1: Each factor's historical mean QSpread as a view
    # P = identity matrix (one view per factor: "factor k will earn Q_k")
    P = np.eye(n_factors)
    Q = qs_is.mean().values
    _flush(f"  Historical mean QSpread: {dict(zip(selected_factors, Q.round(6)))}")

    # Omega: view uncertainty
    # Use prior-scaled approach: omega_k = tau * sigma_f[k,k] / confidence_k
    # Confidence from IC: higher |IC| → more confident
    omega_diag = np.zeros(n_factors)
    for i, fname in enumerate(selected_factors):
        prior_var = tau * sigma_f[i, i]
        confidence = 1.0
        if ic_table is not None and fname in ic_table.index:
            ic = abs(float(ic_table.loc[fname, "Mean IC"]))
            # IC of 0.03 → confidence 1.6, IC of 0.05 → 2.0
            confidence = 1.0 + 20.0 * ic
        # Also boost confidence if FM risk premium is significant
        if fm_table is not None and fname in fm_table.index:
            fm_sig = fm_table.loc[fname, "Significant (5%)"]
            if fm_sig == True or fm_sig == "True":
                confidence *= 1.5
        omega_diag[i] = prior_var / confidence
        _flush(f"  {fname}: Q={Q[i]:.6f}, confidence={confidence:.2f}, omega={omega_diag[i]:.8f}")

    Omega = np.diag(omega_diag)

    # ── Run BL model ──
    _flush("\n[A4] Computing BL posterior...")
    bl_f = black_litterman_posterior(delta, sigma_f, w_eq, tau, P, Q, Omega)
    mu_post_f = bl_f["mu_posterior"]
    w_bl_f = bl_f["weights"]

    _flush(f"  Prior (equilibrium) returns: {dict(zip(selected_factors, pi_f.round(6)))}")
    _flush(f"  Posterior returns: {dict(zip(selected_factors, mu_post_f.round(6)))}")
    _flush(f"  BL weights: {dict(zip(selected_factors, w_bl_f.round(4)))}")
    _flush(f"  Sum of weights: {w_bl_f.sum():.4f}")

    # Compare with Stage 3 allocations
    weights_csv = tables_path / "factor_allocation_weights.csv"
    if weights_csv.exists():
        stage3_weights = pd.read_csv(weights_csv, index_col=0)
        stage3_weights["BL"] = w_bl_f
        _flush("\n  All Factor Allocation Weights:")
        print(stage3_weights.round(4))

    # ── OOS evaluation ──
    _flush("\n[A5] Out-of-sample evaluation...")
    bl_oos_ret = qs_oos @ w_bl_f

    # Also compute Stage 3 portfolio returns for comparison
    all_oos_returns = {"BL": bl_oos_ret}
    if weights_csv.exists():
        for col in stage3_weights.columns:
            if col != "BL":
                w = stage3_weights[col].values
                all_oos_returns[col] = qs_oos @ w

    perf_rows = {}
    for pname, ret in all_oos_returns.items():
        perf_rows[pname] = {
            "Ann. Return": _ann_ret(ret),
            "Ann. Volatility": _ann_vol(ret),
            "Sharpe Ratio": _sharpe(ret),
            "Max Drawdown": _max_dd(ret),
        }

    perf_f = pd.DataFrame(perf_rows)
    _flush("\n  Factor-Level OOS Performance:")
    print(perf_f.round(4))

    # Statistical test: BL vs IC-Weighted
    if "IC-Weighted" in all_oos_returns:
        try:
            dm = diebold_mariano_test(
                all_oos_returns["BL"],
                all_oos_returns["IC-Weighted"],
            )
            _flush(f"\n  Diebold-Mariano (BL vs IC-Weighted): stat={dm['DM Statistic']:.4f}, p={dm['p-value']:.4f}")
        except Exception as e:
            _flush(f"  DM test failed: {e}")

    # ── Sensitivity analysis ──
    _flush("\n[A6] Tau/delta sensitivity for factor-level BL...")
    tau_vals = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    delta_vals = [2.5, 5, 10, 25, 50]

    # Scale Omega proportionally for each tau in the grid
    # (since Omega was built with current tau, need to rescale)
    base_omega_over_tau = omega_diag / tau  # prior_var_per_unit_tau / confidence

    grid_sharpe = {}
    grid_weights = {}
    for t in tau_vals:
        for d in delta_vals:
            omega_scaled = np.diag(base_omega_over_tau * t)
            try:
                bl_g = black_litterman_posterior(d, sigma_f, w_eq, t, P, Q, omega_scaled)
                w_g = bl_g["weights"]
                ret_g = qs_oos @ w_g
                key = f"tau={t}, delta={d}"
                grid_sharpe[key] = _sharpe(ret_g)
                grid_weights[key] = dict(zip(selected_factors, w_g.round(4)))
            except Exception:
                pass

    grid_sharpe_s = pd.Series(grid_sharpe)
    _flush("\n  Sensitivity grid (Sharpe):")
    print(grid_sharpe_s.round(4))

    best_key = grid_sharpe_s.idxmax()
    _flush(f"\n  Best: {best_key} → Sharpe={grid_sharpe_s[best_key]:.4f}")
    _flush(f"  Weights: {grid_weights[best_key]}")

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

        # Save stock-level results
        stock_perf_df.to_csv(tables_path / "bl_stock_level_performance.csv")

    # ══════════════════════════════════════════════════════════════
    # Save all results
    # ══════════════════════════════════════════════════════════════
    _flush("\n" + "─" * 60)
    _flush("Saving results...")

    perf_f.to_csv(tables_path / "bl_factor_performance.csv")

    bl_factor_weights = pd.DataFrame({
        "Equal Weight": w_eq,
        "BL Posterior": w_bl_f,
    }, index=selected_factors)
    bl_factor_weights.to_csv(tables_path / "bl_factor_weights.csv")

    grid_sharpe_s.to_csv(tables_path / "bl_sensitivity_sharpe.csv")

    # Save BL OOS returns alongside Stage 3 returns
    bl_ret_df = pd.DataFrame(all_oos_returns)
    bl_ret_df.to_csv(tables_path / "bl_oos_returns.csv")

    _flush(f"\nStage 4 complete. Results saved to {tables_path}")

    return {
        "bl_factor": bl_f,
        "performance": perf_f,
        "sensitivity_sharpe": grid_sharpe_s,
        "oos_returns": all_oos_returns,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_4(config_path)
