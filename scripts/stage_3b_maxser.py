"""Stage 3B: MAXSER Portfolio — Factor + Stock Investing.

Implements Ao, Li, Zheng (RFS 2019) MAXSER with Scenario 2:
  - Factor component: our 5 selected factor QSpreads (low-dim, plug-in)
  - Stock component: S&P 500 individual stocks, sparse LASSO/Ridge on idiosyncratic returns
  - Combined portfolio maximizes Sharpe while controlling risk

Comparison: same asset universe (5 factors + 50 stocks), three weight methods:
  1. MVO plug-in (our Stage 3 optimizer, constrained)
  2. MAXSER-Lasso (Proposition 3 decomposition + LASSO on idiosyncratic)
  3. MAXSER-Ridge (Proposition 3 decomposition + Ridge on idiosyncratic)

This stage runs independently after Stage 1 and Stage 2 are complete.
"""
import faulthandler
faulthandler.enable()
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import load_sp500_returns
from src.portfolio.maxser import maxser_scenario2
from src.portfolio.optimization import mean_variance_optimize, max_sharpe_portfolio
from sklearn.covariance import LedoitWolf
from src.analytics.performance import max_drawdown, sortino_ratio
from src.analytics.risk import historical_var, cornish_fisher_var

SEED = 42


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_3b(config_path: str = None):
    np.random.seed(SEED)

    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    print("=" * 60)
    print("STAGE 3B: MAXSER Portfolio (Ao, Li, Zheng RFS 2019)")
    print("=" * 60); _flush()

    cache_dir = Path(config.project_root) / config.data.processed_dir / "cache"

    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_3b")
    print(f"  Loading from: {tables_path}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")

    returns = pd.read_pickle(str(cache_dir / "returns.pkl"))
    is_sp500 = pd.read_pickle(str(cache_dir / "is_sp500.pkl"))
    print(f"  Stock returns: {returns.shape}")

    sp500_df = load_sp500_returns(config)
    rf = sp500_df["rf"]

    qs_path = tables_path / "factor_qspreads.csv"
    if not qs_path.exists():
        print("  ERROR: factor_qspreads.csv not found. Run Stage 1 first.")
        return
    qs_df = pd.read_csv(qs_path, index_col=0)
    qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")

    sel_path = tables_path / "selected_factor_combination.csv"
    if sel_path.exists():
        selected_factors = pd.read_csv(sel_path)["factor"].tolist()
    else:
        selected_factors = ["AccrualRatio", "CFTP", "STReversal", "AssetGrowth", "ROE"]

    factor_names = [f for f in selected_factors if f in qs_df.columns]
    factor_qs = qs_df[factor_names].dropna()
    K = len(factor_names)
    print(f"  Our factor QSpreads (K={K}): {factor_qs.shape}")
    _flush()

    # ── Prepare IS/OOS data ───────────────────────────────────────────
    print("\n[2/6] Preparing stock universe...")

    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start
    oos_end = config.dates.end  # 2019-12

    is_returns = returns.loc["1987-01":is_end]

    # S&P 500 stocks with sufficient IS data
    sp500_end = is_sp500[pd.Period(is_end, "M")]
    sp500_stocks = sp500_end[sp500_end == 1].index.tolist()
    min_obs = int(len(is_returns) * 0.8)
    valid_stocks = [g for g in sp500_stocks
                    if g in is_returns.columns and is_returns[g].notna().sum() >= min_obs]

    print(f"  S&P 500 stocks at IS end: {len(sp500_stocks)}")
    print(f"  Stocks with >=80% IS data: {len(valid_stocks)}")

    # Align IS dates
    factor_is = factor_qs.loc["1987-01":is_end]
    common_dates = factor_is.index.intersection(is_returns.index)
    factor_is = factor_is.loc[common_dates]
    factor_is_arr = factor_is.values  # (T, K)

    stock_is = is_returns.loc[common_dates, valid_stocks].fillna(0).values
    rf_is = rf.reindex(common_dates).fillna(0).values
    stock_is_excess = stock_is - rf_is[:, None]
    T_is, N_stocks = stock_is_excess.shape

    print(f"  IS: T={T_is}, N_stocks={N_stocks}, K={K}")

    # Target sigma for MAXSER
    sp500_is_excess = sp500_df["excess_return"].reindex(common_dates).fillna(0)
    sigma_target = sp500_is_excess.std()
    print(f"  MAXSER target sigma (S&P 500 IS monthly std): {sigma_target:.4f}")
    _flush()

    # ── Select subpool (fixed for all methods) ────────────────────────
    print("\n[3/6] Selecting stock subpool...")

    subpool_size = 50
    # Select top 50 stocks by idiosyncratic IR (deterministic, optimization-driven)
    from src.portfolio.maxser import _select_subpool
    sel_idx = _select_subpool(stock_is_excess, factor_is_arr, sigma_target, subpool_size)
    sub_stock_is = stock_is_excess[:, sel_idx]
    sub_stock_names = [valid_stocks[i] for i in sel_idx]

    print(f"  Selected {len(sel_idx)} stocks by idiosyncratic IR")
    _flush()

    # ── Method 1: MVO plug-in on [factors + stocks] ───────────────────
    print("\n[4/6] Running portfolio optimization methods...")
    print("  (a) MVO plug-in on combined [factors + stocks]...")

    combined_is = np.hstack([factor_is_arr, sub_stock_is])  # (T, K+N_sub)
    mu_combined = combined_is.mean(axis=0)
    cov_combined = LedoitWolf().fit(combined_is).covariance_

    # Same constraints as Stage 3 MVO: long_only=False, leverage<=3, weights in [-2, 2]
    try:
        w_mvo = mean_variance_optimize(
            mu_combined, cov_combined,
            risk_aversion=10.0,
            long_only=False,
            min_weight=-2.0, max_weight=2.0,
            gross_leverage=3.0,
        )
        w_mvo_f = w_mvo[:K]
        w_mvo_s = w_mvo[K:]
        print(f"    Factor weights: {dict(zip(factor_names, np.round(w_mvo_f, 4)))}")
        print(f"    Nonzero stock positions: {np.sum(np.abs(w_mvo_s) > 1e-4)}")
        print(f"    Gross leverage: {np.sum(np.abs(w_mvo)):.2f}x")
    except Exception as e:
        print(f"    MVO failed: {e}")
        w_mvo = None
    _flush()

    # ── Method 1b: Max Sharpe on [factors + stocks] ─────────────────────
    print("  (a2) Max Sharpe on combined [factors + stocks]...")
    try:
        w_maxsharpe = max_sharpe_portfolio(
            mu_combined, cov_combined,
            rf=0.0,
            long_only=False,
            gross_leverage=3.0,
        )
        w_maxsharpe_f = w_maxsharpe[:K]
        w_maxsharpe_s = w_maxsharpe[K:]
        print(f"    Factor weights: {dict(zip(factor_names, np.round(w_maxsharpe_f, 4)))}")
        print(f"    Nonzero stock positions: {np.sum(np.abs(w_maxsharpe_s) > 1e-4)}")
        print(f"    Gross leverage: {np.sum(np.abs(w_maxsharpe)):.2f}x")
    except Exception as e:
        print(f"    Max Sharpe failed: {e}")
        w_maxsharpe = None
    _flush()

    # ── Method 1d: MVO + BL on [factors + stocks] ──────────────────────
    print("  (a4) MVO + BL on combined [factors + stocks]...")
    from src.black_litterman.model import black_litterman_posterior
    tau = config.black_litterman.tau
    delta_bl = config.black_litterman.delta
    try:
        n_combined = K + len(sel_idx)
        w_eq_combined = np.ones(n_combined) / n_combined
        P_combined = np.eye(n_combined)
        Q_combined = mu_combined  # views = IS sample means
        omega_diag = np.array([tau * cov_combined[i, i] for i in range(n_combined)])
        Omega_combined = np.diag(omega_diag)

        bl_combined = black_litterman_posterior(
            delta_bl, cov_combined, w_eq_combined, tau, P_combined, Q_combined, Omega_combined)
        mu_bl_combined = bl_combined["mu_posterior"]
        print(f"    BL posterior computed (tau={tau}, delta={delta_bl})")

        # MVO + BL
        w_mvo_bl = mean_variance_optimize(
            mu_bl_combined, cov_combined,
            risk_aversion=delta_bl,
            long_only=False, min_weight=-0.5, max_weight=1.0,
            gross_leverage=2.0,
        )
        w_mvo_bl_f = w_mvo_bl[:K]
        w_mvo_bl_s = w_mvo_bl[K:]
        print(f"    MVO+BL factor weights: {dict(zip(factor_names, np.round(w_mvo_bl_f, 4)))}")
        print(f"    MVO+BL gross leverage: {np.sum(np.abs(w_mvo_bl)):.2f}x")

        # Max Sharpe + BL
        w_ms_bl = max_sharpe_portfolio(
            mu_bl_combined, cov_combined,
            rf=0.0, long_only=False, gross_leverage=3.0,
        )
        w_ms_bl_f = w_ms_bl[:K]
        w_ms_bl_s = w_ms_bl[K:]
        print(f"    MaxSharpe+BL factor weights: {dict(zip(factor_names, np.round(w_ms_bl_f, 4)))}")
        print(f"    MaxSharpe+BL gross leverage: {np.sum(np.abs(w_ms_bl)):.2f}x")
    except Exception as e:
        print(f"    BL variants failed: {e}")
        import traceback; traceback.print_exc()
        w_mvo_bl = None
        w_ms_bl = None
    _flush()

    # ── Method 1c: IC-Weighted on [factors + stocks] ──────────────────
    print("  (a3) IC-Weighted on combined [factors + stocks]...")
    # IC for factors: use saved IC values; for stocks: use rank IC of stock returns
    ic_path = tables_path / "ic_analysis.csv"
    ic_values = np.zeros(K + len(sel_idx))
    if ic_path.exists():
        ic_table = pd.read_csv(ic_path, index_col=0)
        for j, f in enumerate(factor_names):
            if f in ic_table.index:
                ic_values[j] = abs(ic_table.loc[f, "Mean IC"])
            else:
                ic_values[j] = 0.02  # default
    else:
        ic_values[:K] = 0.02

    # For stocks: use correlation of IS returns with their mean as a proxy for IC
    for j in range(len(sel_idx)):
        stock_ret = sub_stock_is[:, j]
        ic_values[K + j] = abs(np.corrcoef(stock_ret[:-1], stock_ret[1:])[0, 1])

    # Normalize to sum to 1
    ic_values = ic_values / ic_values.sum()
    w_ic_f = ic_values[:K]
    w_ic_s = ic_values[K:]
    print(f"    Factor weight sum: {w_ic_f.sum():.4f}, Stock weight sum: {w_ic_s.sum():.4f}")
    _flush()

    # ── Method 2: MAXSER-Lasso (Scenario 2) ───────────────────────────
    print("  (b) MAXSER-Lasso (Proposition 3 decomposition + LASSO)...")
    try:
        # Use fixed subpool (selected in step 3), no internal subpool selection
        res_lasso = maxser_scenario2(
            sub_stock_is, factor_is_arr, sigma_target,
            n_folds=10, method="lasso",
            subpool_size=None,
        )
        res_lasso["selected_stock_idx"] = sel_idx
        w_lasso_f = res_lasso['w_factors']
        w_lasso_s = res_lasso['w_stocks']
        print(f"    Factor weights: {dict(zip(factor_names, np.round(w_lasso_f, 4)))}")
        print(f"    Nonzero stock positions: {res_lasso['n_nonzero_stocks']}")
        print(f"    Factor leverage: {np.sum(np.abs(w_lasso_f)):.2f}x, "
              f"Stock leverage: {np.sum(np.abs(w_lasso_s)):.2f}x")
        print(f"    theta_f={res_lasso['theta_f']:.4f}, theta_u={res_lasso['theta_u']:.4f}, "
              f"theta_all={res_lasso['theta_all']:.4f}")
    except Exception as e:
        print(f"    MAXSER-Lasso failed: {e}")
        import traceback; traceback.print_exc()
        res_lasso = None
    _flush()

    # ── Method 3: MAXSER-Ridge (Scenario 2) ───────────────────────────
    print("  (c) MAXSER-Ridge (Proposition 3 decomposition + Ridge)...")
    try:
        res_ridge = maxser_scenario2(
            sub_stock_is, factor_is_arr, sigma_target,
            n_folds=10, method="ridge",
            subpool_size=None,
        )
        res_ridge["selected_stock_idx"] = sel_idx
        w_ridge_f = res_ridge['w_factors']
        w_ridge_s = res_ridge['w_stocks']
        print(f"    Factor weights: {dict(zip(factor_names, np.round(w_ridge_f, 4)))}")
        print(f"    Nonzero stock positions: {res_ridge['n_nonzero_stocks']}")
        print(f"    Factor leverage: {np.sum(np.abs(w_ridge_f)):.2f}x, "
              f"Stock leverage: {np.sum(np.abs(w_ridge_s)):.2f}x")
        print(f"    theta_f={res_ridge['theta_f']:.4f}, theta_u={res_ridge['theta_u']:.4f}, "
              f"theta_all={res_ridge['theta_all']:.4f}")
    except Exception as e:
        print(f"    MAXSER-Ridge failed: {e}")
        import traceback; traceback.print_exc()
        res_ridge = None
    _flush()

    # ── Compute OOS performance ───────────────────────────────────────
    print("\n[5/6] Computing out-of-sample performance...")

    factor_oos = factor_qs.loc[oos_start:oos_end].dropna()
    factor_oos_arr = factor_oos.values
    oos_stock_returns = returns.loc[oos_start:oos_end, valid_stocks].reindex(factor_oos.index).fillna(0)
    rf_oos = rf.reindex(factor_oos.index).fillna(0)
    oos_stock_excess = oos_stock_returns.sub(rf_oos, axis=0)
    sub_stock_oos = oos_stock_excess.iloc[:, sel_idx].values

    sp500_oos_excess = sp500_df["excess_return"].reindex(factor_oos.index).fillna(0)

    portfolio_oos = {}

    # Benchmarks
    portfolio_oos["S&P 500"] = sp500_oos_excess

    # Our existing Stage 3 portfolios for reference
    our_ret_path = tables_path / "our_portfolio_returns.csv"
    if our_ret_path.exists():
        our_ret = pd.read_csv(our_ret_path, index_col=0)
        our_ret.index = pd.PeriodIndex(our_ret.index, freq="M")
        for col in ["IC-Weighted", "Equal Weight", "MVO", "Max Sharpe",
                    "MVO + BL", "Max Sharpe + BL", "Risk Parity"]:
            if col in our_ret.columns:
                oos_col = our_ret[col].loc[oos_start:oos_end].dropna()
                if len(oos_col) > 0:
                    portfolio_oos[f"{col} (factors only)"] = oos_col

    # Method 1a: MVO on [factors + stocks]
    if w_mvo is not None:
        mvo_oos = factor_oos_arr @ w_mvo_f + sub_stock_oos @ w_mvo_s
        portfolio_oos["MVO (factors+stocks)"] = pd.Series(mvo_oos, index=factor_oos.index)

    # Method 1b: Max Sharpe on [factors + stocks]
    if w_maxsharpe is not None:
        maxsharpe_oos = factor_oos_arr @ w_maxsharpe_f + sub_stock_oos @ w_maxsharpe_s
        portfolio_oos["Max Sharpe (factors+stocks)"] = pd.Series(maxsharpe_oos, index=factor_oos.index)

    # Method 1d: MVO + BL on [factors + stocks]
    if w_mvo_bl is not None:
        mvo_bl_oos = factor_oos_arr @ w_mvo_bl_f + sub_stock_oos @ w_mvo_bl_s
        portfolio_oos["MVO + BL (factors+stocks)"] = pd.Series(mvo_bl_oos, index=factor_oos.index)

    # Method 1e: Max Sharpe + BL on [factors + stocks]
    if w_ms_bl is not None:
        ms_bl_oos = factor_oos_arr @ w_ms_bl_f + sub_stock_oos @ w_ms_bl_s
        portfolio_oos["Max Sharpe + BL (factors+stocks)"] = pd.Series(ms_bl_oos, index=factor_oos.index)

    # Method 1c: IC-Weighted on [factors + stocks]
    ic_oos = factor_oos_arr @ w_ic_f + sub_stock_oos @ w_ic_s
    portfolio_oos["IC-Weighted (factors+stocks)"] = pd.Series(ic_oos, index=factor_oos.index)

    # Method 2: MAXSER-Lasso
    if res_lasso is not None:
        lasso_oos = factor_oos_arr @ w_lasso_f + sub_stock_oos @ w_lasso_s
        portfolio_oos["MAXSER-Lasso (factors+stocks)"] = pd.Series(lasso_oos, index=factor_oos.index)

    # Method 3: MAXSER-Ridge
    if res_ridge is not None:
        ridge_oos = factor_oos_arr @ w_ridge_f + sub_stock_oos @ w_ridge_s
        portfolio_oos["MAXSER-Ridge (factors+stocks)"] = pd.Series(ridge_oos, index=factor_oos.index)

    # ── Performance table ──
    print("\n  Out-of-Sample Performance (2015-2019):")
    print(f"  Same universe: {K} factors + {subpool_size} stocks\n")
    perf = {}
    for name, ret in portfolio_oos.items():
        r = ret.dropna() if isinstance(ret, pd.Series) else pd.Series(ret).dropna()
        if len(r) < 6:
            continue
        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        perf[name] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sr,
            "Sortino": sortino_ratio(r),
            "Max DD": max_drawdown(r),
            "VaR 95%": historical_var(r, 0.95),
            "CF-VaR 95%": cornish_fisher_var(r, 0.95),
        }

    perf_df = pd.DataFrame(perf).T.sort_values("Sharpe", ascending=False)
    print(perf_df.round(4))

    # ── Save results ──────────────────────────────────────────────────
    print("\n[6/6] Saving results...")

    perf_df.to_csv(tables_path / "maxser_performance.csv")
    print(f"  Saved maxser_performance.csv")

    oos_df = pd.DataFrame({k: v for k, v in portfolio_oos.items() if isinstance(v, pd.Series)})
    oos_df.to_csv(tables_path / "maxser_oos_returns.csv")
    print(f"  Saved maxser_oos_returns.csv")

    # Save weight details
    weight_details = {}
    if w_mvo is not None:
        weight_details["MVO"] = {"factors": w_mvo_f, "stocks": w_mvo_s}
    if w_maxsharpe is not None:
        weight_details["Max-Sharpe"] = {"factors": w_maxsharpe_f, "stocks": w_maxsharpe_s}
    if w_mvo_bl is not None:
        weight_details["MVO-BL"] = {"factors": w_mvo_bl_f, "stocks": w_mvo_bl_s}
    if w_ms_bl is not None:
        weight_details["MaxSharpe-BL"] = {"factors": w_ms_bl_f, "stocks": w_ms_bl_s}
    weight_details["IC-Weighted"] = {"factors": w_ic_f, "stocks": w_ic_s}
    if res_lasso is not None:
        weight_details["MAXSER-Lasso"] = {"factors": w_lasso_f, "stocks": w_lasso_s}
    if res_ridge is not None:
        weight_details["MAXSER-Ridge"] = {"factors": w_ridge_f, "stocks": w_ridge_s}

    for method, wd in weight_details.items():
        safe = method.lower().replace(" ", "_").replace("-", "_")
        # Factor weights
        pd.DataFrame({"factor": factor_names, "weight": wd["factors"]}).to_csv(
            tables_path / f"maxser_{safe}_factor_weights.csv", index=False)
        # Stock weights (nonzero only)
        sw = pd.DataFrame({"gvkey": sub_stock_names, "weight": wd["stocks"]})
        sw = sw[sw["weight"].abs() > 1e-8].sort_values("weight", key=abs, ascending=False)
        sw.to_csv(tables_path / f"maxser_{safe}_stock_weights.csv", index=False)
        print(f"  Saved {safe} weights")

    # Save subpool composition
    pd.DataFrame({"gvkey": sub_stock_names, "idx": sel_idx}).to_csv(
        tables_path / "maxser_subpool.csv", index=False)
    print(f"  Saved subpool composition ({len(sel_idx)} stocks)")

    # ── Plot cumulative returns ──
    try:
        from src.visualization.portfolio_plots import plot_portfolio_cumulative_comparison
        plot_portfolio_cumulative_comparison(
            {k: v for k, v in portfolio_oos.items() if isinstance(v, pd.Series)},
            title="MAXSER vs MVO vs Our Factors (OOS Cumulative Returns)",
            save_path=fig_path,
            filename="maxser_cumulative.png",
        )
        print(f"  Saved cumulative returns plot")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\nStage 3B complete. Results saved to {tables_path}")
    return {"performance": perf_df, "returns": portfolio_oos}


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_3b(config_path)
