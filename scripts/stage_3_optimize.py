"""Stage 3: Factor Portfolio Construction & Benchmark Comparison.

Takes the selected factors from Stage 2, constructs optimized factor portfolios,
and benchmarks against mutual funds, smart beta ETFs, hedge fund indices,
and Fama-French factors.
"""
import faulthandler
faulthandler.enable()
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config
from src.data.loader import DataPanel, load_fama_french, load_sp500_returns
from src.factors.registry import build_all_factors
from src.factors.validation import QuintileSorter
from src.portfolio.covariance import ledoit_wolf_shrinkage
from src.portfolio.optimization import (
    mean_variance_optimize, global_minimum_variance,
    max_sharpe_portfolio, risk_parity,
)
from src.analytics.performance import (
    compute_descriptive_stats, max_drawdown, cumulative_returns,
    rolling_sharpe, sortino_ratio, calmar_ratio,
)
from src.analytics.risk import (
    parametric_var, historical_var, cornish_fisher_var, cvar, drawdown_stats,
)
from src.analytics.statistical_tests import sharpe_ratio_test, diebold_mariano_test
from src.visualization.portfolio_plots import (
    plot_efficient_frontier, plot_portfolio_cumulative_comparison,
    plot_rolling_sharpe_comparison,
)


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


def _build_composite_qspread(
    qspreads: pd.DataFrame,
    ic_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Build a composite long-short return from multiple factor QSpreads.

    IC-weighted combination: weight each factor's QSpread by its IC magnitude.
    """
    selected = qspreads.columns.tolist()
    if ic_weights:
        weights = np.array([abs(ic_weights.get(f, 1.0)) for f in selected])
    else:
        weights = np.ones(len(selected))

    weights = weights / weights.sum()
    composite = (qspreads * weights).sum(axis=1)
    return composite


def _flush():
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
        gc.collect()

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

    # 3. MVO
    try:
        # Allow short since factors can be long-short
        mvo = mean_variance_optimize(
            mu_is, sigma_is, risk_aversion=config.optimization.risk_aversion,
            long_only=False, min_weight=-0.5, max_weight=1.0,
        )
        our_portfolios["MVO"] = mvo
        print(f"  MVO: done")
    except Exception as e:
        print(f"  MVO failed: {e}")

    # 4. Max Sharpe (long-only to avoid extreme leverage)
    try:
        ms = max_sharpe_portfolio(mu_is, sigma_is, long_only=True)
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

    # Compute portfolio return series
    our_returns = {}
    for pname, weights in our_portfolios.items():
        our_returns[pname] = sel_qspreads @ weights

    # ── Part C: Load benchmark data ──────────────────────────────────────
    print("\n[3/5] Loading benchmark data...")

    # S&P 500 index return as primary benchmark
    sp500_df = load_sp500_returns(config)
    benchmarks = {}
    benchmarks["S&P 500"] = sp500_df["ret_sp500"]  # already in decimal

    ff_data = load_fama_french(config)
    for sheet_name, raw_df in ff_data.items():
        if "Date" not in raw_df.columns:
            continue
        df = _parse_ff_sheet(raw_df)
        df = _detect_and_normalize(df.select_dtypes(include=[np.number]))
        df = df.dropna(how="all")

        if "fama" in sheet_name.lower() or "ff" in sheet_name.lower():
            # Use market factor as benchmark
            if "mktrf" in df.columns:
                benchmarks["FF Market (Mkt-RF)"] = df["mktrf"]
            if "smb" in df.columns:
                benchmarks["FF SMB"] = df["smb"]
            if "hml" in df.columns:
                benchmarks["FF HML"] = df["hml"]
            if "umd" in df.columns:
                benchmarks["FF Momentum (UMD)"] = df["umd"]
        else:
            # Use equal-weight portfolio of funds as benchmark
            ew_ret = df.mean(axis=1)
            benchmarks[f"{sheet_name} (EW)"] = ew_ret

            # Also include best individual fund (highest Sharpe)
            sharpes = df.mean() / df.std()
            best_fund = sharpes.idxmax()
            benchmarks[f"{sheet_name} Best ({best_fund})"] = df[best_fund]

    print(f"  Benchmarks loaded: {list(benchmarks.keys())}")

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
        }

    comp_df = pd.DataFrame(comparison).T
    comp_df = comp_df.sort_values("Sharpe", ascending=False)
    print("\n  Out-of-Sample Performance Comparison:")
    print(comp_df.round(4))

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

    # ── Part E: Save results and plots ───────────────────────────────────
    print("\n[5/5] Saving results and generating plots...")
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_3")

    comp_df.to_csv(tables_path / "benchmark_comparison.csv")
    weight_df.to_csv(tables_path / "factor_allocation_weights.csv")

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

    # Save all OOS return series for plotting separately
    oos_returns = {}
    for name, ret in all_series.items():
        r = ret.loc[common_start:common_end].dropna()
        if len(r) > 0:
            oos_returns[name] = r
    oos_df = pd.DataFrame(oos_returns)
    oos_df.to_csv(tables_path / "oos_returns_all.csv")
    print(f"  OOS return series saved for plotting"); _flush()

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
