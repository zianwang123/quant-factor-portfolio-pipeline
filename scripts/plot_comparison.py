"""Generate the key benchmark comparison plot from saved OOS returns."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load OOS returns
tables = PROJECT_ROOT / "outputs" / "tables"
fig_path = PROJECT_ROOT / "outputs" / "figures" / "stage_3"
fig_path.mkdir(parents=True, exist_ok=True)

oos = pd.read_csv(tables / "oos_returns_all.csv", index_col=0)
oos.index = pd.PeriodIndex(oos.index, freq="M")

# Pick key series for comparison
key_names = [
    col for col in oos.columns
    if any(k in col for k in ["IC-Weighted", "S&P 500", "Hedge Fund Index (EW)",
                               "Mutual Fund (EW)", "Smart Beta (EW)"])
]
print(f"Plotting: {key_names}")

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6", "#06b6d4"]

# ── Chart 1: Cumulative Returns Comparison ──
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=["Our Factor Portfolio vs Key Benchmarks (Out-of-Sample)", "Drawdowns"],
)

for i, name in enumerate(key_names):
    r = oos[name].dropna()
    cum = (1 + r).cumprod()
    dates = r.index.to_timestamp()
    color = COLORS[i % len(COLORS)]
    lw = 3 if "Our" in name else 1.8

    fig.add_trace(go.Scatter(
        x=dates, y=cum.values,
        mode="lines", name=name,
        line=dict(color=color, width=lw),
        legendgroup=name,
    ), row=1, col=1)

    # Drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    fig.add_trace(go.Scatter(
        x=dates, y=dd.values,
        mode="lines", name=name,
        line=dict(color=color, width=1),
        fill="tozeroy", opacity=0.4,
        showlegend=False, legendgroup=name,
    ), row=2, col=1)

fig.update_layout(
    template="plotly_white",
    font=dict(size=14),
    height=800, width=1200,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    hovermode="x unified",
)
fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
fig.update_yaxes(title_text="Drawdown", row=2, col=1)

fig.write_html(str(fig_path / "our_portfolio_vs_benchmarks.html"), include_plotlyjs="cdn")
try:
    fig.write_image(str(fig_path / "our_portfolio_vs_benchmarks.png"), width=1200, height=800, scale=2)
except Exception as e:
    print(f"  PNG export failed (kaleido): {e}")

print(f"Saved to {fig_path / 'our_portfolio_vs_benchmarks.html'}")

# ── Chart 2: All strategies bar chart (Sharpe ratios) ──
comp = pd.read_csv(tables / "benchmark_comparison.csv", index_col=0)
comp_sorted = comp.sort_values("Sharpe", ascending=True)

fig2 = go.Figure()
colors_bar = ["#2563eb" if "Our" in name else "#94a3b8" for name in comp_sorted.index]
fig2.add_trace(go.Bar(
    y=comp_sorted.index,
    x=comp_sorted["Sharpe"],
    orientation="h",
    marker_color=colors_bar,
    text=[f"{v:.3f}" for v in comp_sorted["Sharpe"]],
    textposition="outside",
))
fig2.update_layout(
    title="Out-of-Sample Sharpe Ratio Comparison",
    xaxis_title="Annualized Sharpe Ratio",
    template="plotly_white",
    height=600, width=900,
    font=dict(size=13),
    margin=dict(l=250),
)

fig2.write_html(str(fig_path / "sharpe_comparison_bar.html"), include_plotlyjs="cdn")
try:
    fig2.write_image(str(fig_path / "sharpe_comparison_bar.png"), width=900, height=600, scale=2)
except Exception as e:
    print(f"  PNG export failed (kaleido): {e}")

print(f"Saved to {fig_path / 'sharpe_comparison_bar.html'}")
print("Done!")
