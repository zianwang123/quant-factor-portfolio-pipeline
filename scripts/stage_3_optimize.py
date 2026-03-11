"""Stage 3: Factor Portfolio Construction & Benchmark Comparison.

Takes the selected factors from Stage 2, constructs optimized factor portfolios,
and benchmarks against mutual funds, smart beta ETFs, hedge fund indices,
and Fama-French factors.
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
from src.analytics.performance import (
    max_drawdown, sortino_ratio, calmar_ratio,
)
from src.analytics.risk import (
    historical_var, cvar, cornish_fisher_var, drawdown_stats,
)
from src.analytics.statistical_tests import sharpe_ratio_test
from src.visualization.portfolio_plots import plot_efficient_frontier


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
    """Fix data anomalies: replace spike+reversal pairs with neutral values.

    Detects months where return > 500% followed by > -80% drop (or vice versa),
    which indicates a split/distribution adjustment error.
    """
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

    # ── Part A: Load factor data (from saved CSVs or recompute) ─────────
    print("\n[1/5] Loading factor data...")

    # Try loading from saved Stage 1 output first (avoid recomputation)
    qspreads_path = config.tables_path() / "factor_qspreads.csv"
    if qspreads is None and qspreads_path.exists():
        print("  Loading QSpreads from saved Stage 1 output...")
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

    # Load selected factors from Stage 2 output if not provided
    if selected_factors is None:
        sel_path = config.tables_path() / "selected_factor_combination.csv"
        if sel_path.exists():
            sel_df = pd.read_csv(sel_path)
            selected_factors = sel_df["factor"].tolist()
            print(f"  Loaded selected factors from Stage 2: {selected_factors}")
        else:
            selected_factors = ["HL1M", "MOM", "BP"]  # fallback
            print(f"  WARNING: No Stage 2 output found, using defaults: {selected_factors}")

    # Load IC table for weighting
    if ic_table is None:
        ic_path = config.tables_path() / "ic_analysis.csv"
        if ic_path.exists():
            ic_table = pd.read_csv(ic_path, index_col=0)

    # Build QSpread DataFrame for selected factors
    sel_qspreads = pd.DataFrame({f: qspreads[f] for f in selected_factors if f in qspreads})
    sel_qspreads = sel_qspreads.dropna()

    print(f"  Selected factors: {selected_factors}")
    print(f"  QSpread data: {sel_qspreads.shape[0]} months, {sel_qspreads.shape[1]} factors")

    # INTERMEDIATE DUMP: save selected QSpreads
    tables_path = config.tables_path()
    sel_qspreads.to_csv(tables_path / "selected_qspreads.csv")
    print(f"  [DUMP] selected_qspreads.csv saved"); _flush()

    # ── Part B: Optimize factor allocation ───────────────────────────────
    print("\n[2/5] Optimizing factor allocation...")

    # Split in-sample / out-of-sample
    is_end = config.dates.in_sample_end
    oos_start = config.dates.out_of_sample_start

    is_data = sel_qspreads.loc[:is_end]
    oos_data = sel_qspreads.loc[oos_start:]

    print(f"  In-sample: {len(is_data)} months ({is_data.index[0]}–{is_data.index[-1]})")
    print(f"  Out-of-sample: {len(oos_data)} months ({oos_data.index[0]}–{oos_data.index[-1]})")

    mu_is = is_data.mean().values
    sigma_is = ledoit_wolf_shrinkage(is_data).values

    # Build multiple portfolio strategies
    our_portfolios = {}

    # 1. Equal-weight factor combination
    n_factors = len(selected_factors)
    ew = np.ones(n_factors) / n_factors
    our_portfolios["Equal Weight"] = ew

    # 2. IC-weighted
    if ic_table is not None:
        ic_weights = {}
        for f in selected_factors:
            if f in ic_table.index:
                ic_weights[f] = abs(ic_table.loc[f, "Mean IC"])
            else:
                ic_weights[f] = 1.0 / n_factors
        w_ic = np.array([ic_weights[f] for f in selected_factors])
        w_ic = w_ic / w_ic.sum()
        our_portfolios["IC-Weighted"] = w_ic
    else:
        our_portfolios["IC-Weighted"] = ew.copy()

    # 3. MVO (hedge fund style: allow shorting, gross leverage up to 3x)
    try:
        mvo = mean_variance_optimize(
            mu_is, sigma_is, risk_aversion=config.optimization.risk_aversion,
            long_only=False, min_weight=-2.0, max_weight=2.0,
            gross_leverage=3.0,
        )
        our_portfolios["MVO"] = mvo
        print(f"  MVO: done")
    except Exception as e:
        print(f"  MVO failed: {e}")

    # 4. Max Sharpe (allow shorting, gross leverage capped at 3x like MVO)
    try:
        ms = max_sharpe_portfolio(mu_is, sigma_is, long_only=False, gross_leverage=3.0)
        our_portfolios["Max Sharpe"] = ms
        print(f"  Max Sharpe: done")
    except Exception as e:
        print(f"  Max Sharpe failed: {e}")

    # 5. Risk Parity (requires positive weights, use absolute QSpreads)
    try:
        rp = risk_parity(sigma_is)
        our_portfolios["Risk Parity"] = rp
        print(f"  Risk Parity: done")
    except Exception as e:
        print(f"  Risk Parity failed: {e}")

    # Print weights
    weight_df = pd.DataFrame(our_portfolios, index=selected_factors)
    print(f"\n  Factor Allocation Weights:")
    print(weight_df.round(4))

    # INTERMEDIATE DUMP: save weights
    weight_df.to_csv(tables_path / "factor_allocation_weights.csv")
    print(f"  [DUMP] factor_allocation_weights.csv saved"); _flush()

    # Compute portfolio return series
    our_returns = {}
    for pname, weights in our_portfolios.items():
        our_returns[pname] = sel_qspreads @ weights

    # INTERMEDIATE DUMP: save our portfolio returns
    our_ret_df = pd.DataFrame(our_returns)
    our_ret_df.to_csv(tables_path / "our_portfolio_returns.csv")
    print(f"  [DUMP] our_portfolio_returns.csv saved"); _flush()

    # ── Part C: Load benchmark data ──────────────────────────────────────
    print("\n[3/5] Loading benchmark data...")

    # S&P 500 index return as primary benchmark
    sp500_df = load_sp500_returns(config)
    rf = sp500_df["rf"]  # monthly risk-free rate
    benchmarks = {}
    benchmarks["S&P 500"] = sp500_df["ret_sp500"] - rf  # excess return

    # Load benchmark data from pre-converted CSVs
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
            # Subtract risk-free rate to get excess returns for long-only benchmarks
            rf_aligned = rf.reindex(df.index).fillna(0)
            df_excess = df.sub(rf_aligned, axis=0)

            ew_ret = df_excess.mean(axis=1)
            benchmarks[f"{sheet_name} (EW)"] = ew_ret

            sharpes = df_excess.mean() / df_excess.std()
            best_fund = sharpes.idxmax()
            benchmarks[f"{sheet_name} Best ({best_fund})"] = df_excess[best_fund]

    print(f"  Benchmarks loaded: {list(benchmarks.keys())}")

    # INTERMEDIATE DUMP: save raw benchmark returns
    bench_df = pd.DataFrame(benchmarks)
    bench_df.to_csv(tables_path / "benchmark_raw_returns.csv")
    print(f"  [DUMP] benchmark_raw_returns.csv saved"); _flush()

    # ── Part D: Performance comparison ───────────────────────────────────
    print("\n[4/5] Computing performance comparison...")

    # Align all series to common OOS period
    all_series = {}
    for name, ret in our_returns.items():
        oos = ret.loc[oos_start:]
        if len(oos) > 0:
            all_series[f"Our: {name}"] = oos

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
    print("\n  Out-of-Sample Performance Comparison:")
    print(comp_df.round(4))

    # INTERMEDIATE DUMP: save comparison and OOS returns NOW (before stat tests)
    comp_df.to_csv(tables_path / "benchmark_comparison.csv")
    oos_returns_early = {}
    for name, ret in all_series.items():
        r = ret.loc[common_start:common_end].dropna()
        if len(r) > 0:
            oos_returns_early[name] = r
    pd.DataFrame(oos_returns_early).to_csv(tables_path / "oos_returns_all.csv")
    print(f"  [DUMP] benchmark_comparison.csv + oos_returns_all.csv saved"); _flush()

    # ── Statistical tests: our best vs each benchmark ────────────────────
    print("\n  Statistical Tests (Sharpe Ratio Equality):")
    # Find our best strategy
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

    # ── Part E: Save remaining results ──────────────────────────────────
    print("\n[5/5] Saving final results...")
    fig_path = config.figures_path("stage_3")

    # In-sample performance for our factor portfolios
    is_comparison = {}
    for pname, weights in our_portfolios.items():
        is_ret = is_data @ weights
        is_comparison[pname] = {
            "Ann. Return": is_ret.mean() * 12,
            "Ann. Vol": is_ret.std() * np.sqrt(12),
            "Sharpe": (is_ret.mean() * 12) / (is_ret.std() * np.sqrt(12)),
        }
    is_df = pd.DataFrame(is_comparison).T
    is_df.to_csv(tables_path / "factor_portfolio_in_sample.csv")
    print(f"  [DUMP] factor_portfolio_in_sample.csv saved"); _flush()

    # Efficient frontier plot
    print("  Generating efficient frontier plot...")
    try:
        plot_efficient_frontier(
            mu_is, sigma_is,
            gmv_weights=our_portfolios.get("MVO"),
            mvp_weights=our_portfolios.get("Max Sharpe"),
            rp_weights=our_portfolios.get("Risk Parity"),
            asset_names=selected_factors,
            title="Efficient Frontier (In-Sample, Selected Factors)",
            save_path=fig_path,
            filename="efficient_frontier.png",
        )
        print("  Saved efficient_frontier.html")
    except Exception as e:
        print(f"  Efficient frontier plot failed: {e}")

    print(f"\nStage 3 complete. Results saved to {tables_path} and {fig_path}")
    return {
        "our_portfolios": our_portfolios,
        "our_returns": our_returns,
        "benchmarks": benchmarks,
        "comparison": comp_df,
        "weights": weight_df,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_3(config_path)
