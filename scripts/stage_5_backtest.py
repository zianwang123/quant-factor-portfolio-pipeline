"""Stage 5: Rolling Backtest with Transaction Costs.

Simulates realistic portfolio management:
- Rolling lookback window to re-estimate factor covariance and expected returns
- Rebalances quarterly (configurable)
- Deducts transaction costs (10bps) on every weight change
- Compares all portfolio variants (with and without BL)

Produces:
- Gross vs net-of-cost Sharpe comparison table
- Cumulative return plots (gross and net)
- Rolling Sharpe comparison
- Weight evolution over time
- Turnover analysis
"""
import faulthandler
faulthandler.enable()
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import load_config
from src.portfolio.backtest import RollingBacktest
from src.portfolio.optimization import (
    mean_variance_optimize, max_sharpe_portfolio, risk_parity,
)
from src.black_litterman.equilibrium import implied_equilibrium_returns
from src.black_litterman.model import black_litterman_posterior
from src.analytics.performance import (
    cumulative_returns, rolling_sharpe, max_drawdown, sortino_ratio, calmar_ratio,
)
from src.analytics.risk import historical_var, cvar, cornish_fisher_var
from src.visualization.portfolio_plots import (
    plot_rolling_sharpe_comparison,
    plot_weight_evolution,
)


def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()
    sys.stderr.flush()


def _sharpe(r):
    return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


# ── Optimizer functions for RollingBacktest ──

def equal_weight_optimizer(mu, sigma, **kwargs):
    n = len(mu)
    return np.ones(n) / n


def ic_weighted_optimizer(mu, sigma, **kwargs):
    """Weight by |mean return| (proxy for IC from lookback window)."""
    w = np.abs(mu)
    if w.sum() == 0:
        return np.ones(len(mu)) / len(mu)
    return w / w.sum()


def mvo_optimizer(mu, sigma, **kwargs):
    return mean_variance_optimize(
        mu, sigma,
        risk_aversion=kwargs.get("risk_aversion", 10),
        long_only=False, min_weight=-2.0, max_weight=2.0,
        gross_leverage=3.0,
    )


def max_sharpe_optimizer(mu, sigma, **kwargs):
    return max_sharpe_portfolio(
        mu, sigma,
        long_only=False, gross_leverage=3.0,
    )


def risk_parity_optimizer(mu, sigma, **kwargs):
    return risk_parity(sigma)


def bl_optimizer(mu, sigma, **kwargs):
    """Black-Litterman: prior=equal-weight, views=historical means.

    Computes BL posterior returns, then uses constrained MVO to get
    well-behaved weights. This is the practitioner approach — BL provides
    better return estimates, MVO provides portfolio constraints.
    """
    n = len(mu)
    w_eq = np.ones(n) / n
    tau = kwargs.get("tau", 0.05)
    delta = kwargs.get("delta", 50)

    P = np.eye(n)
    Q = mu  # views = lookback mean returns

    omega_diag = np.array([tau * sigma[i, i] for i in range(n)])
    Omega = np.diag(omega_diag)

    try:
        bl = black_litterman_posterior(delta, sigma, w_eq, tau, P, Q, Omega)
        # Use constrained MVO with BL posterior returns
        return mean_variance_optimize(
            bl["mu_posterior"], sigma,
            risk_aversion=delta,
            long_only=False, min_weight=-0.5, max_weight=1.0,
            gross_leverage=2.0,
        )
    except Exception:
        return w_eq


def run_stage_5(config_path: str = None):
    config = load_config(config_path, project_root=str(PROJECT_ROOT))
    tables_path = config.tables_path()
    fig_path = config.figures_path("stage_5")

    print("=" * 60)
    print("STAGE 5: Rolling Backtest with Transaction Costs")
    print("=" * 60)

    oos_start = config.dates.out_of_sample_start
    oos_start_period = pd.Period(oos_start, freq="M")

    # ── Load factor QSpreads ──
    _flush("\n[1/4] Loading factor data...")
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

    sel_qspreads = qs_df[selected_factors].dropna()
    _flush(f"  Selected factors: {selected_factors}")
    _flush(f"  QSpread data: {sel_qspreads.shape}")

    # ── Run rolling backtests ──
    _flush("\n[2/4] Running rolling backtests...")

    lookback = 60  # 5 years
    rebalance_freq = "quarterly"
    tc_bps = config.optimization.transaction_cost_bps
    tau = config.black_litterman.tau
    delta = config.black_litterman.delta

    _flush(f"  Lookback: {lookback} months")
    _flush(f"  Rebalance: {rebalance_freq}")
    _flush(f"  Transaction cost: {tc_bps} bps")

    portfolios = {
        "Equal Weight": (equal_weight_optimizer, {}),
        "IC-Weighted": (ic_weighted_optimizer, {}),
        "MVO": (mvo_optimizer, {"risk_aversion": config.optimization.risk_aversion}),
        "Max Sharpe": (max_sharpe_optimizer, {}),
        "Risk Parity": (risk_parity_optimizer, {}),
        "BL": (bl_optimizer, {"tau": tau, "delta": delta}),
    }

    # Run each backtest: once with costs, once without
    results_gross = {}
    results_net = {}

    for pname, (opt_func, opt_kwargs) in portfolios.items():
        _flush(f"\n  Running {pname}...")

        # Gross (no costs)
        bt_gross = RollingBacktest(
            sel_qspreads, opt_func,
            lookback_months=lookback,
            rebalance_freq=rebalance_freq,
            transaction_cost_bps=0,
        )
        res_gross = bt_gross.run(**opt_kwargs)

        # Net (with costs)
        bt_net = RollingBacktest(
            sel_qspreads, opt_func,
            lookback_months=lookback,
            rebalance_freq=rebalance_freq,
            transaction_cost_bps=tc_bps,
        )
        res_net = bt_net.run(**opt_kwargs)

        if res_gross and len(res_gross["portfolio_returns"]) > 0:
            results_gross[pname] = res_gross
            results_net[pname] = res_net

            ret_g = res_gross["portfolio_returns"]
            ret_n = res_net["portfolio_returns"]
            turnover = res_net.get("turnover", pd.Series(dtype=float))
            costs = res_net.get("transaction_costs", pd.Series(dtype=float))

            # Only report OOS portion
            ret_g_oos = ret_g.loc[ret_g.index >= oos_start_period]
            ret_n_oos = ret_n.loc[ret_n.index >= oos_start_period]

            _flush(f"    Total months: {len(ret_g)}, OOS months: {len(ret_g_oos)}")
            _flush(f"    Gross Sharpe (OOS): {_sharpe(ret_g_oos):.3f}")
            _flush(f"    Net Sharpe (OOS):   {_sharpe(ret_n_oos):.3f}")
            _flush(f"    Avg turnover:       {turnover.mean():.3f}")
            _flush(f"    Total costs:        {costs.sum()*100:.2f}%")
        else:
            _flush(f"    FAILED — no returns generated")

    # ── Compute performance table ──
    _flush("\n\n[3/4] Computing performance comparison...")

    perf_rows = []
    oos_returns_gross = {}
    oos_returns_net = {}

    for pname in results_gross:
        ret_g = results_gross[pname]["portfolio_returns"]
        ret_n = results_net[pname]["portfolio_returns"]
        turnover = results_net[pname].get("turnover", pd.Series(dtype=float))

        # Filter to OOS
        ret_g_oos = ret_g.loc[ret_g.index >= oos_start_period]
        ret_n_oos = ret_n.loc[ret_n.index >= oos_start_period]

        oos_returns_gross[pname] = ret_g_oos
        oos_returns_net[pname] = ret_n_oos

        perf_rows.append({
            "Portfolio": pname,
            "Gross Ann. Return": ret_g_oos.mean() * 12,
            "Gross Ann. Vol": ret_g_oos.std() * np.sqrt(12),
            "Gross Sharpe": _sharpe(ret_g_oos),
            "Net Ann. Return": ret_n_oos.mean() * 12,
            "Net Ann. Vol": ret_n_oos.std() * np.sqrt(12),
            "Net Sharpe": _sharpe(ret_n_oos),
            "Sharpe Cost": _sharpe(ret_g_oos) - _sharpe(ret_n_oos),
            "Max DD (Net)": max_drawdown(ret_n_oos),
            "Sortino (Net)": sortino_ratio(ret_n_oos),
            "VaR 95%": historical_var(ret_n_oos, 0.95),
            "CF-VaR 95%": cornish_fisher_var(ret_n_oos, 0.95),
            "CVaR 95%": cvar(ret_n_oos, 0.95),
            "Avg Turnover": turnover.mean(),
            "Total Cost (bps)": results_net[pname].get("transaction_costs", pd.Series(dtype=float)).sum() * 10000,
        })

    perf_df = pd.DataFrame(perf_rows).set_index("Portfolio")
    perf_df = perf_df.sort_values("Net Sharpe", ascending=False)

    _flush("\n  Rolling Backtest Performance (OOS):")
    print(perf_df.round(4))

    # Also show the static (non-rolling) Sharpe for comparison
    static_csv = tables_path / "benchmark_comparison.csv"
    if static_csv.exists():
        static = pd.read_csv(static_csv, index_col=0)
        _flush("\n  Static (non-rolling) Sharpe for reference:")
        for pname in perf_df.index:
            static_name = f"Our: {pname}"
            if static_name in static.index:
                _flush(f"    {pname}: static={static.loc[static_name, 'Sharpe']:.3f}, "
                       f"rolling_gross={perf_df.loc[pname, 'Gross Sharpe']:.3f}, "
                       f"rolling_net={perf_df.loc[pname, 'Net Sharpe']:.3f}")

    # ── Save tables ──
    perf_df.to_csv(tables_path / "backtest_performance.csv")
    _flush(f"\n  Saved backtest_performance.csv")

    # Save OOS returns for plotting
    gross_df = pd.DataFrame(oos_returns_gross)
    net_df = pd.DataFrame(oos_returns_net)
    gross_df.to_csv(tables_path / "backtest_returns_gross.csv")
    net_df.to_csv(tables_path / "backtest_returns_net.csv")

    # ── Generate plots ──
    _flush("\n[4/4] Generating plots...")

    # 1. Cumulative returns comparison (net of costs)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    COLORS = {
        "Equal Weight": "#2563eb", "IC-Weighted": "#0ea5e9",
        "MVO": "#7c3aed", "Max Sharpe": "#db2777",
        "Risk Parity": "#059669", "BL": "#f59e0b",
    }

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[
            "Rolling Backtest: Cumulative Returns (Net of Transaction Costs)",
            "Drawdowns",
        ],
    )

    for pname, ret in oos_returns_net.items():
        cum = (1 + ret).cumprod()
        dates = ret.index.to_timestamp()
        color = COLORS.get(pname, "#94a3b8")
        lw = 3 if pname in ("IC-Weighted", "BL") else 1.5

        fig.add_trace(go.Scatter(
            x=dates, y=cum.values,
            mode="lines", name=f"{pname} (Sharpe:{_sharpe(ret):.2f})",
            line=dict(color=color, width=lw),
            legendgroup=pname,
        ), row=1, col=1)

        # Drawdown
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        fig.add_trace(go.Scatter(
            x=dates, y=dd.values,
            mode="lines", name=pname,
            line=dict(color=color, width=1),
            fill="tozeroy", opacity=0.4,
            showlegend=False, legendgroup=pname,
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_white", font=dict(size=14),
        height=800, width=1200,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)

    fig.write_html(str(fig_path / "backtest_cumulative.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(str(fig_path / "backtest_cumulative.png"), width=1200, height=800, scale=2)
    except Exception:
        pass
    _flush("  Saved backtest_cumulative.html")

    # 2. Rolling Sharpe comparison
    plot_rolling_sharpe_comparison(
        oos_returns_net, window=24,
        title="Rolling 24M Sharpe Ratio (Net of Costs)",
        save_path=fig_path, filename="backtest_rolling_sharpe.png",
    )
    _flush("  Saved backtest_rolling_sharpe.html")

    # 3. Gross vs Net Sharpe bar chart
    fig2 = go.Figure()

    sorted_names = perf_df.index.tolist()
    fig2.add_trace(go.Bar(
        y=sorted_names,
        x=perf_df["Gross Sharpe"].values,
        orientation="h", name="Gross Sharpe",
        marker_color="#93c5fd",
        text=[f"{v:.3f}" for v in perf_df["Gross Sharpe"]],
        textposition="outside",
    ))
    fig2.add_trace(go.Bar(
        y=sorted_names,
        x=perf_df["Net Sharpe"].values,
        orientation="h", name="Net Sharpe",
        marker_color="#2563eb",
        text=[f"{v:.3f}" for v in perf_df["Net Sharpe"]],
        textposition="outside",
    ))

    fig2.update_layout(
        title="Gross vs Net Sharpe Ratio (Rolling Backtest, OOS)",
        xaxis_title="Annualized Sharpe Ratio",
        template="plotly_white",
        height=500, width=900,
        font=dict(size=13),
        margin=dict(l=150),
        barmode="group",
    )

    fig2.write_html(str(fig_path / "backtest_sharpe_comparison.html"), include_plotlyjs="cdn")
    try:
        fig2.write_image(str(fig_path / "backtest_sharpe_comparison.png"), width=900, height=500, scale=2)
    except Exception:
        pass
    _flush("  Saved backtest_sharpe_comparison.html")

    # 4. Weight evolution for BL and IC-Weighted
    for pname in ["BL", "IC-Weighted", "MVO"]:
        if pname in results_net and "weights_history" in results_net[pname]:
            wh = results_net[pname]["weights_history"]
            if not wh.empty:
                plot_weight_evolution(
                    wh, top_n=5,
                    title=f"{pname}: Factor Weight Evolution (Rolling)",
                    save_path=fig_path,
                    filename=f"backtest_weights_{pname.lower().replace(' ', '_').replace('-', '_')}.png",
                )
                _flush(f"  Saved weight evolution for {pname}")

    # 5. Turnover comparison
    fig3 = go.Figure()
    for pname in sorted_names:
        if pname in results_net:
            turnover = results_net[pname].get("turnover", pd.Series(dtype=float))
            if not turnover.empty:
                dates = turnover.index.to_timestamp()
                color = COLORS.get(pname, "#94a3b8")
                fig3.add_trace(go.Bar(
                    x=dates, y=turnover.values,
                    name=pname, marker_color=color, opacity=0.7,
                ))

    fig3.update_layout(
        title="Portfolio Turnover at Each Rebalance",
        yaxis_title="Turnover (sum of |weight changes|)",
        template="plotly_white",
        height=500, width=1200,
        font=dict(size=13),
        barmode="group",
    )

    fig3.write_html(str(fig_path / "backtest_turnover.html"), include_plotlyjs="cdn")
    try:
        fig3.write_image(str(fig_path / "backtest_turnover.png"), width=1200, height=500, scale=2)
    except Exception:
        pass
    _flush("  Saved backtest_turnover.html")

    # 6. BL vs Non-BL side-by-side comparison
    if "BL" in oos_returns_net:
        _flush("\n  Generating BL vs Non-BL comparison...")

        # Pair each non-BL strategy with the BL strategy
        non_bl_names = [n for n in oos_returns_net if n != "BL"]
        bl_ret = oos_returns_net["BL"]

        fig_bl = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Cumulative Returns: BL vs Non-BL (Net)",
                "Rolling 24M Sharpe: BL vs Non-BL",
            ],
            horizontal_spacing=0.08,
        )

        # Left: Cumulative returns
        cum_bl = (1 + bl_ret).cumprod()
        dates_bl = bl_ret.index.to_timestamp()
        fig_bl.add_trace(go.Scatter(
            x=dates_bl, y=cum_bl.values,
            mode="lines", name="BL",
            line=dict(color=COLORS["BL"], width=3),
            legendgroup="BL",
        ), row=1, col=1)

        for pname in non_bl_names:
            ret = oos_returns_net[pname]
            cum = (1 + ret).cumprod()
            dates = ret.index.to_timestamp()
            color = COLORS.get(pname, "#94a3b8")
            fig_bl.add_trace(go.Scatter(
                x=dates, y=cum.values,
                mode="lines", name=pname,
                line=dict(color=color, width=1.5),
                legendgroup=pname,
            ), row=1, col=1)

        # Right: Rolling Sharpe
        window = 24
        rs_bl = bl_ret.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(12), raw=True)
        fig_bl.add_trace(go.Scatter(
            x=dates_bl, y=rs_bl.values,
            mode="lines", name="BL",
            line=dict(color=COLORS["BL"], width=3),
            showlegend=False, legendgroup="BL",
        ), row=1, col=2)

        for pname in non_bl_names:
            ret = oos_returns_net[pname]
            rs = ret.rolling(window).apply(lambda x: x.mean()/x.std()*np.sqrt(12), raw=True)
            dates = ret.index.to_timestamp()
            color = COLORS.get(pname, "#94a3b8")
            fig_bl.add_trace(go.Scatter(
                x=dates, y=rs.values,
                mode="lines", name=pname,
                line=dict(color=color, width=1.5),
                showlegend=False, legendgroup=pname,
            ), row=1, col=2)

        fig_bl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

        fig_bl.update_layout(
            template="plotly_white", font=dict(size=13),
            height=500, width=1400,
            title_text="Black-Litterman vs Traditional Portfolios (Rolling Backtest, Net of Costs)",
            legend=dict(x=0.02, y=0.98),
            hovermode="x unified",
        )
        fig_bl.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig_bl.update_yaxes(title_text="Rolling 24M Sharpe", row=1, col=2)

        fig_bl.write_html(str(fig_path / "bl_vs_nonbl_comparison.html"), include_plotlyjs="cdn")
        try:
            fig_bl.write_image(str(fig_path / "bl_vs_nonbl_comparison.png"), width=1400, height=500, scale=2)
        except Exception:
            pass
        _flush("  Saved bl_vs_nonbl_comparison.html")

        # BL vs non-BL summary table
        bl_summary = []
        bl_sharpe = _sharpe(bl_ret)
        for pname in non_bl_names:
            ret = oos_returns_net[pname]
            s = _sharpe(ret)
            bl_summary.append({
                "Portfolio": pname,
                "Non-BL Sharpe": s,
                "BL Sharpe": bl_sharpe,
                "BL Improvement": bl_sharpe - s,
                "BL Better?": "Yes" if bl_sharpe > s else "No",
            })
        bl_summary_df = pd.DataFrame(bl_summary).set_index("Portfolio")
        bl_summary_df.to_csv(tables_path / "bl_vs_nonbl_summary.csv")
        _flush("\n  BL vs Non-BL Summary:")
        print(bl_summary_df.round(4))
        _flush("  Saved bl_vs_nonbl_summary.csv")

    _flush(f"\nStage 5 complete. Results saved to {tables_path} and {fig_path}")

    return {
        "performance": perf_df,
        "gross_returns": oos_returns_gross,
        "net_returns": oos_returns_net,
        "results": results_net,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "config" / "pipeline.yaml")
    run_stage_5(config_path)
