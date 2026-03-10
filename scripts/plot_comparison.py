"""Generate the key benchmark comparison plot from saved OOS returns."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use timestamped run folder if set: outputs/run_TIMESTAMP/tables, figures
run_id = os.environ.get("PIPELINE_RUN_ID", "")
if run_id:
    run_base = PROJECT_ROOT / "outputs" / run_id
else:
    run_base = PROJECT_ROOT / "outputs"
tables = run_base / "tables"
fig_path = run_base / "figures" / "stage_3"
fig_path.mkdir(parents=True, exist_ok=True)

oos = pd.read_csv(tables / "oos_returns_all.csv", index_col=0)
oos.index = pd.PeriodIndex(oos.index, freq="M")

# Create "Market + Alpha" strategy: S&P 500 return + our factor alpha
# This shows what a long-only investor would get by overlaying our factor signals
oos["Our: Market + Factor Alpha"] = oos["S&P 500"] + oos["Our: IC-Weighted"]

# Pick key series: all "Our:" variants + key benchmarks
key_names = [
    col for col in oos.columns
    if col.startswith("Our:") or any(k in col for k in [
        "S&P 500", "Hedge Fund Index (EW)", "Mutual Fund (EW)", "Smart Beta (EW)",
    ])
]
print(f"Plotting: {key_names}")

# Colors: blues for our portfolios, other colors for benchmarks
OUR_COLORS = {"Equal Weight": "#2563eb", "IC-Weighted": "#0ea5e9", "MVO": "#7c3aed",
              "Max Sharpe": "#db2777", "Risk Parity": "#059669", "Market + Factor Alpha": "#0d9488"}
BENCH_COLORS = {"S&P 500": "#dc2626", "Mutual Fund (EW)": "#f59e0b",
                "Smart Beta (EW)": "#84cc16", "Hedge Fund Index (EW)": "#f97316"}

def _get_color(name):
    for k, c in OUR_COLORS.items():
        if k in name:
            return c
    for k, c in BENCH_COLORS.items():
        if k in name:
            return c
    return "#94a3b8"

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
    color = _get_color(name)
    lw = 2.5 if "Our" in name else 1.5

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
