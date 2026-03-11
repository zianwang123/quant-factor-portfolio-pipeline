"""Stage 3: Factor Portfolio Construction, MAXSER, & Benchmark Comparison.

Takes the selected factors from Stage 2, constructs:
  Part A: Factor-only portfolios (EW, IC-Weighted, MVO, MaxSharpe, RiskParity, BL variants)
  Part B: Factor+stock portfolios via MAXSER (Ao, Li, Zheng 2019) — Lasso and Ridge variants
and benchmarks all of them against mutual funds, smart beta ETFs, hedge fund indices,
and Fama-French factors in a unified comparison table.

Note: We intentionally do NOT use plug-in optimizers (MVO, MaxSharpe, etc.) on the joint
[factors + stocks] space. Stock returns contain factor exposures (r_i = beta*f + alpha + eps),
so joint optimization double-counts factor risk and produces uninterpretable weights.
MAXSER correctly decomposes into factor leg + idiosyncratic leg with beta-adjustment.
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
from src.data.loader import DataPanel, load_sp500_returns
from src.factors.registry import build_all_factors
from src.factors.validation import QuintileSorter
from src.portfolio.covariance import ledoit_wolf_shrinkage
from src.portfolio.optimization import (
    mean_variance_optimize, max_sharpe_portfolio, risk_parity,
)
from src.portfolio.maxser import maxser_scenario2
from src.analytics.performance import (
    max_drawdown, sortino_ratio, calmar_ratio,
)
from src.analytics.risk import (
    historical_var, cvar, cornish_fisher_var, drawdown_stats,
)
from src.analytics.statistical_tests import sharpe_ratio_test
from src.black_litterman.model import black_litterman_posterior
from src.visualization.portfolio_plots import plot_efficient_frontier
from src.portfolio.maxser import _select_subpool


def _parse_ff_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a Fama-French Excel sheet: set Date as Period index."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m").dt.to_period("M")
    df = df.set_index("Date").dropna(how="all")
    return df


def _detect_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Detect if returns are in percent and convert to decimal."""
    avg_abs = df.abs().mean().mean()
    if avg_abs > 0.5:
        return df / 100.0
    return df


def _fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data anomalies: replace spike+reversal pairs with neutral values."""
    df = df.copy()
    for col in df.columns:
        series = df[col].dropna()
        for i in range(len(series) - 1):
            curr, nxt = series.iloc[i], series.iloc[i + 1]
            if curr > 5.0 and nxt < -0.8:
                net = (1 + curr) * (1 + nxt) - 1
                monthly = (1 + net) ** 0.5 - 1
                df.loc[series.index[i], col] = monthly
                df.loc[series.index[i + 1], col] = monthly
                _flush(f"  Fixed anomaly in {col}: {series.index[i]} ({curr:.2f}) "
                       f"+ {series.index[i+1]} ({nxt:.2f}) -> {monthly:.4f} each")
    return df


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def run_stage_3(config_path: str = None, factors: dict = None, qspreads: dict = None,
                selected_factors: list = None, ic_table: pd.DataFrame = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    print("=" * 60)
    print("STAGE 3: Factor Portfolio Construction & Benchmark Comparison")
    print("=" * 60); _flush()

    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start
    oos_end = config.dates.end
    tables_path = config.tables_path()
    cache_dir = Path(config.project_root) / config.data.processed_dir / "cache"

    # ══════════════════════════════════════════════════════════════════
    # [1/7] Load factor data
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/7] Loading factor data...")

    qspreads_path = config.find_prior_output("factor_qspreads.csv")
    if qspreads is None and qspreads_path is not None:
        print(f"  Loading QSpreads from {qspreads_path.parent.parent.name}...")
        qs_df = pd.read_csv(qspreads_path, index_col=0)
        qs_df.index = pd.PeriodIndex(qs_df.index, freq="M")
        qspreads = {col: qs_df[col] for col in qs_df.columns}
    elif qspreads is None:
        print("  Recomputing factors (no saved output found)...")
        panel = DataPanel(config)
        returns = panel.get_returns()
        is_sp500 = panel.get_sp500_membership()
        if factors is None:
            factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])
        panel._raw = None
        sorter = QuintileSorter(n_bins=config.factors.quintile_bins)
        sort_results = sorter.sort_all_factors(factors, returns, is_sp500)
        qspreads = {name: res["qspread"] for name, res in sort_results.items()}

    if selected_factors is None:
        sel_path = config.find_prior_output("selected_factor_combination.csv")
        if sel_path is not None:
            sel_df = pd.read_csv(sel_path)
            selected_factors = sel_df["factor"].tolist()
            print(f"  Loaded selected factors from {sel_path.parent.parent.name}: {selected_factors}")
        else:
            selected_factors = ["HL1M", "MOM", "BP"]
            print(f"  WARNING: No Stage 2 output found, using defaults: {selected_factors}")

    if ic_table is None:
        ic_path = config.find_prior_output("ic_analysis.csv")
        if ic_path is not None:
            ic_table = pd.read_csv(ic_path, index_col=0)

    sel_qspreads = pd.DataFrame({f: qspreads[f] for f in selected_factors if f in qspreads})
    sel_qspreads = sel_qspreads.dropna()
    n_factors = len(selected_factors)
    K = n_factors

    print(f"  Selected factors: {selected_factors}")
    print(f"  QSpread data: {sel_qspreads.shape[0]} months, {K} factors")

    sel_qspreads.to_csv(tables_path / "selected_qspreads.csv")
    print(f"  [DUMP] selected_qspreads.csv saved"); _flush()

    # ══════════════════════════════════════════════════════════════════
    # [2/7] Factor-only portfolio optimization (Part A)
    # ══════════════════════════════════════════════════════════════════
    print("\n[2/7] Optimizing factor-only allocation...")

    is_data = sel_qspreads.loc[:is_end]
    oos_data = sel_qspreads.loc[oos_start:]

    print(f"  In-sample: {len(is_data)} months ({is_data.index[0]}–{is_data.index[-1]})")
    print(f"  Out-of-sample: {len(oos_data)} months ({oos_data.index[0]}–{oos_data.index[-1]})")

    mu_is = is_data.mean().values
    sigma_is = ledoit_wolf_shrinkage(is_data).values

    factor_only = {}

    # 1. Equal-weight
    ew = np.ones(K) / K
    factor_only["Equal Weight"] = ew

    # 2. IC-weighted
    if ic_table is not None:
        ic_weights = {}
        for f in selected_factors:
            if f in ic_table.index:
                ic_weights[f] = abs(ic_table.loc[f, "Mean IC"])
            else:
                ic_weights[f] = 1.0 / K
        w_ic = np.array([ic_weights[f] for f in selected_factors])
        w_ic = w_ic / w_ic.sum()
        factor_only["IC-Weighted"] = w_ic
    else:
        factor_only["IC-Weighted"] = ew.copy()

    # 3. MVO
    try:
        mvo = mean_variance_optimize(
            mu_is, sigma_is, risk_aversion=config.optimization.risk_aversion,
            long_only=False, min_weight=-2.0, max_weight=2.0, gross_leverage=3.0,
        )
        factor_only["MVO"] = mvo
        print(f"  MVO: done")
    except Exception as e:
        print(f"  MVO failed: {e}")

    # 4. Max Sharpe
    try:
        ms = max_sharpe_portfolio(mu_is, sigma_is, long_only=False, gross_leverage=3.0)
        factor_only["Max Sharpe"] = ms
        print(f"  Max Sharpe: done")
    except Exception as e:
        print(f"  Max Sharpe failed: {e}")

    # 5. Risk Parity
    try:
        rp = risk_parity(sigma_is)
        factor_only["Risk Parity"] = rp
        print(f"  Risk Parity: done")
    except Exception as e:
        print(f"  Risk Parity failed: {e}")

    # 6-7. BL variants
    tau = config.black_litterman.tau
    delta = config.black_litterman.delta
    try:
        w_eq = np.ones(K) / K
        P = np.eye(K)
        Q = mu_is
        omega_diag = np.array([tau * sigma_is[i, i] for i in range(K)])
        Omega = np.diag(omega_diag)
        bl = black_litterman_posterior(delta, sigma_is, w_eq, tau, P, Q, Omega)
        mu_bl = bl["mu_posterior"]
        print(f"  BL posterior computed (tau={tau}, delta={delta})")

        try:
            mvo_bl = mean_variance_optimize(
                mu_bl, sigma_is, risk_aversion=delta,
                long_only=False, min_weight=-0.5, max_weight=1.0, gross_leverage=2.0,
            )
            factor_only["MVO + BL"] = mvo_bl
            print(f"  MVO + BL: done")
        except Exception as e:
            print(f"  MVO + BL failed: {e}")

        try:
            ms_bl = max_sharpe_portfolio(mu_bl, sigma_is, long_only=False, gross_leverage=3.0)
            factor_only["Max Sharpe + BL"] = ms_bl
            print(f"  Max Sharpe + BL: done")
        except Exception as e:
            print(f"  Max Sharpe + BL failed: {e}")
    except Exception as e:
        print(f"  BL posterior failed: {e}")

    # Print and save factor-only weights
    weight_df = pd.DataFrame(factor_only, index=selected_factors)
    print(f"\n  Factor-Only Allocation Weights:")
    print(weight_df.round(4))
    weight_df.to_csv(tables_path / "factor_allocation_weights.csv")
    print(f"  [DUMP] factor_allocation_weights.csv saved")

    # Compute factor-only return series
    factor_only_returns = {}
    for pname, weights in factor_only.items():
        factor_only_returns[pname] = sel_qspreads @ weights

    our_ret_df = pd.DataFrame(factor_only_returns)
    our_ret_df.to_csv(tables_path / "our_portfolio_returns.csv")
    print(f"  [DUMP] our_portfolio_returns.csv saved"); _flush()

    # ══════════════════════════════════════════════════════════════════
    # [3/7] Factor+stock portfolio optimization (Part B — MAXSER)
    # ══════════════════════════════════════════════════════════════════
    print("\n[3/7] Building factor+stock portfolios...")

    # Load stock-level data
    stock_returns = pd.read_pickle(str(cache_dir / "returns.pkl"))
    is_sp500 = pd.read_pickle(str(cache_dir / "is_sp500.pkl"))
    sp500_df = load_sp500_returns(config)
    rf = sp500_df["rf"]

    # S&P 500 stocks with sufficient IS data
    is_stock_returns = stock_returns.loc["1987-01":is_end]
    sp500_end = is_sp500[pd.Period(is_end, "M")]
    sp500_stocks = sp500_end[sp500_end == 1].index.tolist()
    min_obs = int(len(is_stock_returns) * 0.8)
    valid_stocks = [g for g in sp500_stocks
                    if g in is_stock_returns.columns and is_stock_returns[g].notna().sum() >= min_obs]
    print(f"  S&P 500 stocks at IS end: {len(sp500_stocks)}")
    print(f"  Stocks with >=80% IS data: {len(valid_stocks)}")

    # Align IS dates between factors and stocks
    factor_is = sel_qspreads.loc["1987-01":is_end]
    common_dates = factor_is.index.intersection(is_stock_returns.index)
    factor_is = factor_is.loc[common_dates]
    factor_is_arr = factor_is.values  # (T, K)

    stock_is = is_stock_returns.loc[common_dates, valid_stocks].fillna(0).values
    rf_is = rf.reindex(common_dates).fillna(0).values
    stock_is_excess = stock_is - rf_is[:, None]
    T_is, N_stocks = stock_is_excess.shape
    print(f"  IS: T={T_is}, N_stocks={N_stocks}, K={K}")

    # Target sigma for MAXSER
    sp500_is_excess = sp500_df["excess_return"].reindex(common_dates).fillna(0)
    sigma_target = sp500_is_excess.std()
    print(f"  MAXSER target sigma: {sigma_target:.4f}")

    factor_stock = {}       # name -> {"w_f": array, "w_s": array}

    # MAXSER (Ao, Li, Zheng 2019) decomposes into factor + idiosyncratic legs:
    #   1. Factor leg: plug-in tangency portfolio (K is small, sample estimates are fine)
    #   2. Stock leg: regress out factor exposures, apply sparse regression on residuals
    #   3. Combine with beta-adjustment to avoid double-counting factor exposure
    # This is the only principled way to combine factors and stocks in one portfolio.
    # Plug-in methods (MVO, MaxSharpe on stacked [factors+stocks]) are excluded because
    # they double-count factor exposure embedded in stock returns.

    # MAXSER needs N << T, so select a subpool of 50 stocks by idiosyncratic IR
    maxser_subpool_size = 50
    maxser_sel_idx = _select_subpool(stock_is_excess, factor_is_arr, sigma_target, maxser_subpool_size)
    maxser_stock_is = stock_is_excess[:, maxser_sel_idx]
    maxser_stock_names = [valid_stocks[i] for i in maxser_sel_idx]
    print(f"  MAXSER subpool: {maxser_subpool_size} stocks by idiosyncratic IR (N << T required)")

    # (a) MAXSER-Lasso
    print("  (a) MAXSER-Lasso (factors+stocks)...")
    try:
        res = maxser_scenario2(maxser_stock_is, factor_is_arr, sigma_target,
                               n_folds=10, method="lasso", subpool_size=None)
        # Store with full-universe-sized weight vector (zeros for non-subpool stocks)
        w_s_full = np.zeros(N_stocks)
        w_s_full[maxser_sel_idx] = res["w_stocks"]
        factor_stock["MAXSER-Lasso"] = {"w_f": res["w_factors"], "w_s": w_s_full}
        print(f"      Factor leverage: {np.sum(np.abs(res['w_factors'])):.2f}x, "
              f"Stock leverage: {np.sum(np.abs(res['w_stocks'])):.2f}x, "
              f"Nonzero stocks: {res['n_nonzero_stocks']}")
        print(f"      theta_f={res['theta_f']:.4f}, theta_u={res['theta_u']:.4f}")
    except Exception as e:
        print(f"      Failed: {e}")
        import traceback; traceback.print_exc()

    # (b) MAXSER-Ridge
    print("  (b) MAXSER-Ridge (factors+stocks)...")
    try:
        res = maxser_scenario2(maxser_stock_is, factor_is_arr, sigma_target,
                               n_folds=10, method="ridge", subpool_size=None)
        w_s_full = np.zeros(N_stocks)
        w_s_full[maxser_sel_idx] = res["w_stocks"]
        factor_stock["MAXSER-Ridge"] = {"w_f": res["w_factors"], "w_s": w_s_full}
        print(f"      Factor leverage: {np.sum(np.abs(res['w_factors'])):.2f}x, "
              f"Stock leverage: {np.sum(np.abs(res['w_stocks'])):.2f}x, "
              f"Nonzero stocks: {res['n_nonzero_stocks']}")
    except Exception as e:
        print(f"      Failed: {e}")
        import traceback; traceback.print_exc()

    _flush()

    # Save factor+stock weights
    for method, wd in factor_stock.items():
        safe = method.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace("/", "_")
        pd.DataFrame({"factor": selected_factors, "weight": wd["w_f"]}).to_csv(
            tables_path / f"fs_{safe}_factor_weights.csv", index=False)
        sw = pd.DataFrame({"gvkey": valid_stocks, "weight": wd["w_s"]})
        sw = sw[sw["weight"].abs() > 1e-8].sort_values("weight", key=abs, ascending=False)
        sw.to_csv(tables_path / f"fs_{safe}_stock_weights.csv", index=False)

    print(f"  Saved factor+stock weight files ({N_stocks} stocks)")

    # ══════════════════════════════════════════════════════════════════
    # [4/7] Load benchmark data
    # ══════════════════════════════════════════════════════════════════
    print("\n[4/7] Loading benchmark data...")

    benchmarks = {}
    benchmarks["S&P 500"] = sp500_df["ret_sp500"] - rf  # excess return

    processed_dir = config.project_root / config.data.processed_dir
    benchmark_csvs = {
        "fama_french_factor": "Fama-French Factor",
        "mutual_fund": "Mutual Fund",
        "smart_beta": "Smart Beta",
        "hedge_fund_index": "Hedge Fund Index",
    }
    for csv_name, sheet_name in benchmark_csvs.items():
        csv_path = processed_dir / f"{csv_name}.csv"
        if not csv_path.exists():
            _flush(f"  WARNING: {csv_path} not found, skipping")
            continue
        raw_df = pd.read_csv(csv_path)
        if "Date" not in raw_df.columns:
            continue
        df = _parse_ff_sheet(raw_df)
        df = _detect_and_normalize(df.select_dtypes(include=[np.number]))
        df = _fix_anomalies(df)
        df = df.dropna(how="all")

        if "fama" in sheet_name.lower() or "ff" in sheet_name.lower():
            if "mktrf" in df.columns:
                benchmarks["FF Market (Mkt-RF)"] = df["mktrf"]
            if "smb" in df.columns:
                benchmarks["FF SMB"] = df["smb"]
            if "hml" in df.columns:
                benchmarks["FF HML"] = df["hml"]
            if "umd" in df.columns:
                benchmarks["FF Momentum (UMD)"] = df["umd"]
        else:
            rf_aligned = rf.reindex(df.index).fillna(0)
            df_excess = df.sub(rf_aligned, axis=0)
            ew_ret = df_excess.mean(axis=1)
            benchmarks[f"{sheet_name} (EW)"] = ew_ret
            sharpes = df_excess.mean() / df_excess.std()
            best_fund = sharpes.idxmax()
            benchmarks[f"{sheet_name} Best ({best_fund})"] = df_excess[best_fund]

    print(f"  Benchmarks loaded: {list(benchmarks.keys())}")

    bench_df = pd.DataFrame(benchmarks)
    bench_df.to_csv(tables_path / "benchmark_raw_returns.csv")
    print(f"  [DUMP] benchmark_raw_returns.csv saved"); _flush()

    # ══════════════════════════════════════════════════════════════════
    # [5/7] Compute OOS returns for factor+stock portfolios
    # ══════════════════════════════════════════════════════════════════
    print("\n[5/7] Computing factor+stock OOS returns...")

    factor_oos = sel_qspreads.loc[oos_start:oos_end].dropna()
    factor_oos_arr = factor_oos.values

    oos_stock_returns = stock_returns.loc[oos_start:oos_end, valid_stocks].reindex(factor_oos.index).fillna(0)
    rf_oos = rf.reindex(factor_oos.index).fillna(0)
    oos_stock_excess = oos_stock_returns.sub(rf_oos, axis=0).values  # (T_oos, N_stocks)

    factor_stock_returns = {}
    for name, wd in factor_stock.items():
        r = factor_oos_arr @ wd["w_f"] + oos_stock_excess @ wd["w_s"]
        factor_stock_returns[name] = pd.Series(r, index=factor_oos.index)

    # Save MAXSER OOS returns
    pd.DataFrame(factor_stock_returns).to_csv(tables_path / "maxser_oos_returns.csv")
    print(f"  [DUMP] maxser_oos_returns.csv saved"); _flush()

    # ══════════════════════════════════════════════════════════════════
    # [6/7] Unified performance comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n[6/7] Computing unified performance comparison...")

    all_series = {}

    # Factor-only portfolios
    for name, ret in factor_only_returns.items():
        oos = ret.loc[oos_start:]
        if len(oos) > 0:
            all_series[f"Our: {name} (factors only)"] = oos

    # Factor+stock portfolios
    for name, ret in factor_stock_returns.items():
        oos = ret.loc[oos_start:]
        if len(oos) > 0:
            all_series[f"Our: {name} (factors+stocks)"] = oos

    # Benchmarks
    for name, ret in benchmarks.items():
        oos = ret.loc[oos_start:]
        if len(oos) > 0:
            all_series[name] = oos

    # Find common date range
    common_start = max(s.index[0] for s in all_series.values())
    common_end = min(s.index[-1] for s in all_series.values())
    print(f"  Common OOS period: {common_start} to {common_end}")

    comparison = {}
    for name, ret in all_series.items():
        r = ret.loc[common_start:common_end].dropna()
        if len(r) < 6:
            continue

        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        mdd = max_drawdown(r)
        sort_r = sortino_ratio(r)
        cal_r = calmar_ratio(r)
        dd = drawdown_stats(r)
        var_95 = historical_var(r, 0.95)
        cvar_95 = cvar(r, 0.95)
        cf_var_95 = cornish_fisher_var(r, 0.95)

        comparison[name] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sr,
            "Sortino": sort_r,
            "Calmar": cal_r,
            "Max DD": mdd,
            "DD Duration (mo)": dd.get("Max Drawdown Duration (months)", 0),
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "CF-VaR 95%": cf_var_95,
        }

    comp_df = pd.DataFrame(comparison).T
    comp_df = comp_df.sort_values("Sharpe", ascending=False)
    print("\n  Out-of-Sample Performance Comparison (All Portfolios vs Benchmarks):")
    print(comp_df.round(4))

    # Save comparison and OOS returns
    comp_df.to_csv(tables_path / "benchmark_comparison.csv")
    oos_returns_all = {}
    for name, ret in all_series.items():
        r = ret.loc[common_start:common_end].dropna()
        if len(r) > 0:
            oos_returns_all[name] = r
    pd.DataFrame(oos_returns_all).to_csv(tables_path / "oos_returns_all.csv")
    print(f"  [DUMP] benchmark_comparison.csv + oos_returns_all.csv saved"); _flush()

    # Save separate MAXSER performance table for backward compatibility
    maxser_names = [n for n in comp_df.index if "F+S" in n or "MAXSER" in n]
    if maxser_names:
        comp_df.loc[maxser_names].to_csv(tables_path / "maxser_performance.csv")

    # ── Statistical tests: our best vs each benchmark ────────────────
    print("\n  Statistical Tests (Sharpe Ratio Equality):")
    our_names = [n for n in all_series if n.startswith("Our:")]
    if our_names:
        best_our_name = max(our_names, key=lambda n: comp_df.loc[n, "Sharpe"] if n in comp_df.index else -999)
        best_our = all_series[best_our_name].loc[common_start:common_end].dropna()

        for bname in benchmarks:
            if bname in all_series:
                b_ret = all_series[bname].loc[common_start:common_end].dropna()
                common = best_our.index.intersection(b_ret.index)
                if len(common) < 12:
                    continue
                try:
                    test = sharpe_ratio_test(best_our.loc[common], b_ret.loc[common])
                    sig = "***" if test["Significant (5%)"] else ""
                    print(f"    {best_our_name} vs {bname}: "
                          f"SR diff={test['SR Diff']:.4f}, z={test['z-statistic']:.3f}, "
                          f"p={test['p-value']:.4f} {sig}")
                except Exception as e:
                    print(f"    {best_our_name} vs {bname}: test failed ({e})")

    # ══════════════════════════════════════════════════════════════════
    # [7/7] Save final results & plots
    # ══════════════════════════════════════════════════════════════════
    print("\n[7/7] Saving final results...")
    fig_path = config.figures_path("stage_3")

    # In-sample performance for factor-only portfolios
    is_comparison = {}
    for pname, weights in factor_only.items():
        is_ret = is_data @ weights
        is_comparison[pname] = {
            "Ann. Return": is_ret.mean() * 12,
            "Ann. Vol": is_ret.std() * np.sqrt(12),
            "Sharpe": (is_ret.mean() * 12) / (is_ret.std() * np.sqrt(12)),
        }
    is_df = pd.DataFrame(is_comparison).T
    is_df.to_csv(tables_path / "factor_portfolio_in_sample.csv")
    print(f"  [DUMP] factor_portfolio_in_sample.csv saved"); _flush()

    # Efficient frontier plot (factor-only)
    print("  Generating efficient frontier plot...")
    try:
        plot_efficient_frontier(
            mu_is, sigma_is,
            portfolios=factor_only,
            asset_names=selected_factors,
            gross_leverage=3.0,
            title="Efficient Frontier (In-Sample, Selected Factors)",
            save_path=fig_path,
            filename="efficient_frontier.png",
        )
        print("  Saved efficient_frontier.html")
    except Exception as e:
        print(f"  Efficient frontier plot failed: {e}")

    # Cumulative returns plot (all portfolios)
    try:
        from src.visualization.portfolio_plots import plot_portfolio_cumulative_comparison
        all_oos_series = {k: v for k, v in oos_returns_all.items() if isinstance(v, pd.Series)}
        plot_portfolio_cumulative_comparison(
            all_oos_series,
            title="All Portfolios vs Benchmarks (OOS Cumulative Returns)",
            save_path=fig_path,
            filename="all_portfolios_cumulative.png",
        )
        print("  Saved all_portfolios_cumulative plot")
    except Exception as e:
        print(f"  Cumulative plot failed: {e}")

    print(f"\nStage 3 complete. Results saved to {tables_path} and {fig_path}")
    return {
        "factor_only_portfolios": factor_only,
        "factor_only_returns": factor_only_returns,
        "factor_stock_portfolios": factor_stock,
        "factor_stock_returns": factor_stock_returns,
        "benchmarks": benchmarks,
        "comparison": comp_df,
        "weights": weight_df,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_3(config_path)
