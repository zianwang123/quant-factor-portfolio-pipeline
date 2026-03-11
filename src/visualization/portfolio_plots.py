"""Portfolio visualization using Plotly.

Generates interactive HTML charts and static PNG exports.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from src.analytics.performance import cumulative_returns, rolling_sharpe
from src.analytics.risk import drawdown_series


# Color palette
PALETTE = [
    "#2563eb",  # blue (our portfolio)
    "#dc2626",  # red
    "#16a34a",  # green
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#78716c",  # stone
    "#84cc16",  # lime
    "#f97316",  # orange
]


def _to_datetime_index(idx):
    """Convert PeriodIndex to DatetimeIndex for plotly."""
    if hasattr(idx, "to_timestamp"):
        return idx.to_timestamp()
    return idx


def _save_figure(fig, save_path, filename):
    """Save as both HTML (interactive) and PNG (static)."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Always save HTML (no dependencies needed)
    html_name = filename.replace(".png", ".html")
    fig.write_html(str(save_path / html_name), include_plotlyjs="cdn")

    # Try PNG export (requires kaleido)
    try:
        fig.write_image(str(save_path / filename), width=1400, height=800, scale=2)
    except Exception:
        pass  # kaleido not installed or not working


def plot_efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    portfolios: dict[str, np.ndarray] = None,
    rf: float = 0.0,
    asset_names: list = None,
    gross_leverage: float = 3.0,
    title: str = "Efficient Frontier",
    save_path: Path = None,
    filename: str = "efficient_frontier.png",
):
    """Plot mean-variance efficient frontier with key portfolios.

    The frontier is generated with the same constraints as the actual optimizers:
    allows shorting (w >= -2) and gross leverage up to `gross_leverage`.
    """
    fig = go.Figure()

    # Generate frontier points with realistic constraints (matching our optimizers)
    n = len(mu)
    n_points = 100
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret * 0.5, max_ret * 2.5, n_points)

    frontier_vols = []
    frontier_rets = []

    import cvxpy as cp
    for target in target_returns:
        try:
            w = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(w, sigma))
            constraints = [
                cp.sum(w) == 1,
                mu @ w >= target,
                w >= -2.0,
                w <= 2.0,
                cp.norm(w, 1) <= gross_leverage,
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            if prob.status == "optimal":
                port_vol = np.sqrt(w.value @ sigma @ w.value) * np.sqrt(12)
                port_ret = mu @ w.value * 12
                frontier_vols.append(port_vol)
                frontier_rets.append(port_ret)
        except Exception:
            continue

    if frontier_vols:
        fig.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_rets,
            mode="lines", name="Efficient Frontier",
            line=dict(color=PALETTE[0], width=3),
        ))

    # Individual assets
    asset_vols = np.sqrt(np.diag(sigma)) * np.sqrt(12)
    asset_rets = mu * 12
    labels = asset_names or [f"Asset {i}" for i in range(n)]
    fig.add_trace(go.Scatter(
        x=asset_vols, y=asset_rets,
        mode="markers+text", name="Individual Factors",
        marker=dict(size=10, color="#94a3b8"),
        text=labels, textposition="top center",
    ))

    # Plot all portfolio points
    SYMBOLS = ["circle", "square", "diamond", "star", "triangle-up",
               "pentagon", "hexagon", "cross"]
    COLORS_PORT = ["#16a34a", "#f59e0b", "#7c3aed", "#dc2626", "#06b6d4",
                   "#ec4899", "#0e7490", "#f97316"]

    if portfolios:
        for i, (pname, weights) in enumerate(portfolios.items()):
            if weights is not None:
                vol = np.sqrt(weights @ sigma @ weights) * np.sqrt(12)
                ret = mu @ weights * 12
                fig.add_trace(go.Scatter(
                    x=[vol], y=[ret],
                    mode="markers", name=pname,
                    marker=dict(
                        size=14, color=COLORS_PORT[i % len(COLORS_PORT)],
                        symbol=SYMBOLS[i % len(SYMBOLS)],
                        line=dict(width=2, color="black"),
                    ),
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
        font=dict(size=14),
        legend=dict(x=0.02, y=0.98),
    )

    _save_figure(fig, save_path, filename)


def plot_portfolio_cumulative_comparison(
    portfolio_returns: dict[str, pd.Series],
    title: str = "Portfolio Cumulative Returns",
    save_path: Path = None,
    filename: str = "portfolio_comparison.png",
):
    """Plot cumulative returns and drawdowns for multiple portfolios."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[title, "Drawdowns"],
    )

    for i, (name, rets) in enumerate(portfolio_returns.items()):
        color = PALETTE[i % len(PALETTE)]
        cum = cumulative_returns(rets)
        dates = _to_datetime_index(cum.index)

        # Cumulative returns
        fig.add_trace(go.Scatter(
            x=dates, y=cum.values,
            mode="lines", name=name,
            line=dict(color=color, width=2.5 if i == 0 else 1.5),
            legendgroup=name,
        ), row=1, col=1)

        # Drawdowns
        dd = drawdown_series(rets)
        dd_dates = _to_datetime_index(dd.index)
        fig.add_trace(go.Scatter(
            x=dd_dates, y=dd.values,
            mode="lines", name=name,
            line=dict(color=color, width=1),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else None,
            opacity=0.5,
            showlegend=False,
            legendgroup=name,
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        height=750,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)

    _save_figure(fig, save_path, filename)


def plot_rolling_sharpe_comparison(
    portfolio_returns: dict[str, pd.Series],
    window: int = 36,
    title: str = "Rolling Sharpe Ratio Comparison",
    save_path: Path = None,
    filename: str = "rolling_sharpe.png",
):
    """Plot rolling Sharpe ratios for multiple portfolios."""
    fig = go.Figure()

    for i, (name, rets) in enumerate(portfolio_returns.items()):
        color = PALETTE[i % len(PALETTE)]
        rs = rolling_sharpe(rets, window=window)
        dates = _to_datetime_index(rs.index)

        fig.add_trace(go.Scatter(
            x=dates, y=rs.values,
            mode="lines", name=name,
            line=dict(color=color, width=2 if i == 0 else 1.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        yaxis_title=f"Rolling {window}M Sharpe Ratio (Annualized)",
        template="plotly_white",
        font=dict(size=13),
        height=500,
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
    )

    _save_figure(fig, save_path, filename)


def plot_weight_evolution(
    weights_history: pd.DataFrame,
    top_n: int = 10,
    title: str = "Portfolio Weight Evolution",
    save_path: Path = None,
    filename: str = "weight_evolution.png",
):
    """Plot evolution of portfolio weights over time (top N assets)."""
    avg_weights = weights_history.mean().abs().nlargest(top_n)
    top_assets = avg_weights.index.tolist()

    fig = go.Figure()
    for i, asset in enumerate(top_assets):
        color = PALETTE[i % len(PALETTE)]
        dates = _to_datetime_index(weights_history.index)
        fig.add_trace(go.Scatter(
            x=dates, y=weights_history[asset].values,
            mode="lines", name=asset,
            stackgroup="one",
            line=dict(color=color),
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Portfolio Weight",
        template="plotly_white",
        font=dict(size=13),
        height=500,
    )

    _save_figure(fig, save_path, filename)
