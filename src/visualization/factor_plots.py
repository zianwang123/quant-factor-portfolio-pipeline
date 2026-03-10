"""Factor visualization using Plotly.

Generates interactive HTML charts and static PNG exports for factor analytics.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from src.analytics.performance import cumulative_returns


PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6",
    "#06b6d4", "#ec4899", "#78716c", "#84cc16", "#f97316",
]
QUINTILE_COLORS = ["#2563eb", "#60a5fa", "#94a3b8", "#f97316", "#dc2626"]


def _to_dt(idx):
    """Convert PeriodIndex to DatetimeIndex for plotly."""
    if hasattr(idx, "to_timestamp"):
        return idx.to_timestamp()
    return idx


def _save(fig, save_path, filename):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path / filename.replace(".png", ".html")), include_plotlyjs="cdn")
    try:
        fig.write_image(str(save_path / filename), width=1400, height=500, scale=2)
    except Exception:
        pass


def plot_qspread_vs_benchmark(
    qspread: pd.Series,
    benchmark: pd.Series,
    factor_name: str,
    save_path: Path = None,
):
    """Plot replicated QSpread against Capital IQ benchmark."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=_to_dt(qspread.index), y=qspread.values,
        mode="lines", name="Replicated",
        line=dict(color=PALETTE[0], width=1.5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=_to_dt(benchmark.index), y=benchmark.values,
        mode="lines", name="Capital IQ",
        line=dict(color=PALETTE[1], width=1.5),
    ), secondary_y=True)

    fig.update_layout(
        title=f"{factor_name} QSpread: Replicated vs Benchmark",
        template="plotly_white", height=450,
    )
    fig.update_yaxes(title_text="Replicated QSpread", secondary_y=False)
    fig.update_yaxes(title_text="Capital IQ Benchmark", secondary_y=True)

    _save(fig, save_path, f"{factor_name}_qspread_vs_benchmark.png")


def plot_long_short_legs(
    long_ret: pd.Series,
    short_ret: pd.Series,
    factor_name: str,
    save_path: Path = None,
):
    """Plot long and short leg cumulative returns."""
    fig = go.Figure()
    cum_long = cumulative_returns(long_ret)
    cum_short = cumulative_returns(short_ret)

    fig.add_trace(go.Scatter(
        x=_to_dt(cum_long.index), y=cum_long.values,
        mode="lines", name="Long (Top Quintile)",
        line=dict(color=PALETTE[2], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=_to_dt(cum_short.index), y=cum_short.values,
        mode="lines", name="Short (Bottom Quintile)",
        line=dict(color=PALETTE[1], width=2),
    ))

    fig.update_layout(
        title=f"{factor_name}: Long vs Short Leg Cumulative Returns",
        yaxis_title="Cumulative Return",
        template="plotly_white", height=450,
        hovermode="x unified",
    )
    _save(fig, save_path, f"{factor_name}_long_short.png")


def plot_quintile_monotonicity(
    quintile_returns: dict[int, pd.Series],
    factor_name: str,
    save_path: Path = None,
):
    """Plot average returns by quintile — tests for monotonic factor structure."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"{factor_name}: Avg Return by Quintile",
        f"{factor_name}: Cumulative by Quintile",
    ])

    avg_returns = {q: r.mean() for q, r in quintile_returns.items() if len(r) > 0}
    quintiles = sorted(avg_returns.keys())
    avgs = [avg_returns[q] for q in quintiles]

    fig.add_trace(go.Bar(
        x=[f"Q{q}" for q in quintiles], y=avgs,
        marker_color=[QUINTILE_COLORS[q - 1] for q in quintiles],
        showlegend=False,
    ), row=1, col=1)

    for q in quintiles:
        if len(quintile_returns[q]) > 0:
            cum = cumulative_returns(quintile_returns[q])
            fig.add_trace(go.Scatter(
                x=_to_dt(cum.index), y=cum.values,
                mode="lines", name=f"Q{q}",
                line=dict(color=QUINTILE_COLORS[q - 1], width=1.5),
            ), row=1, col=2)

    fig.update_layout(template="plotly_white", height=450, hovermode="x unified")
    _save(fig, save_path, f"{factor_name}_quintile_monotonicity.png")


def plot_cumulative_qspread_vs_market(
    qspread: pd.Series,
    market_excess: pd.Series,
    factor_name: str,
    save_path: Path = None,
):
    """Plot cumulative QSpread vs S&P 500 excess return."""
    fig = go.Figure()
    common = qspread.index.intersection(market_excess.index)
    cum_qs = cumulative_returns(qspread.loc[common])
    cum_mkt = cumulative_returns(market_excess.loc[common])

    fig.add_trace(go.Scatter(
        x=_to_dt(cum_qs.index), y=cum_qs.values,
        mode="lines", name=f"{factor_name} QSpread",
        line=dict(color=PALETTE[0], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=_to_dt(cum_mkt.index), y=cum_mkt.values,
        mode="lines", name="S&P 500 Excess",
        line=dict(color=PALETTE[3], width=2),
    ))

    fig.update_layout(
        title=f"{factor_name} QSpread vs S&P 500 Cumulative Returns",
        yaxis_title="Cumulative Return",
        template="plotly_white", height=450,
        hovermode="x unified",
    )
    _save(fig, save_path, f"{factor_name}_cumulative_vs_sp500.png")


def plot_factor_correlation_heatmap(
    qspreads: pd.DataFrame,
    save_path: Path = None,
):
    """Plot correlation heatmap of factor QSpreads."""
    corr = qspreads.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z = corr.values.copy()
    z[mask] = np.nan

    fig = go.Figure(data=go.Heatmap(
        z=z, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1, zmid=0,
        text=np.round(z, 2), texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
    ))

    fig.update_layout(
        title="Factor QSpread Correlation Matrix",
        template="plotly_white",
        height=700, width=800,
        yaxis=dict(autorange="reversed"),
    )
    _save(fig, save_path, "factor_correlation_heatmap.png")


def plot_ic_time_series(
    ic_series: pd.Series,
    factor_name: str,
    save_path: Path = None,
):
    """Plot IC time series with rolling average."""
    fig = go.Figure()

    dates = _to_dt(ic_series.index)

    # Monthly IC bars
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in ic_series.values]
    fig.add_trace(go.Bar(
        x=dates, y=ic_series.values,
        marker_color=colors, opacity=0.4,
        name="Monthly IC",
    ))

    # 12M rolling mean
    rolling_ic = ic_series.rolling(12).mean()
    fig.add_trace(go.Scatter(
        x=_to_dt(rolling_ic.index), y=rolling_ic.values,
        mode="lines", name="12M Rolling IC",
        line=dict(color=PALETTE[1], width=2.5),
    ))

    # Mean line
    mean_ic = ic_series.mean()
    fig.add_hline(y=mean_ic, line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"Mean IC = {mean_ic:.4f}")
    fig.add_hline(y=0, line_color="gray", opacity=0.3)

    fig.update_layout(
        title=f"{factor_name}: Monthly IC Time Series",
        yaxis_title="Information Coefficient (Rank Correlation)",
        template="plotly_white", height=450,
        hovermode="x unified",
    )
    _save(fig, save_path, f"{factor_name}_ic_series.png")
