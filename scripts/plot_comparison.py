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

oos = pd.read_csv(tables / "s3_oos_returns.csv", index_col=0)
oos.index = pd.PeriodIndex(oos.index, freq="M")

# Create "Market + Alpha" strategy: S&P 500 return + our factor alpha
# This shows what a long-only investor would get by overlaying our factor signals
# Create "Market + Alpha" strategy: S&P 500 return + our factor alpha
if "S&P 500" in oos.columns and "Our: IC-Weighted (factors only)" in oos.columns:
    oos["Our: Market + Factor Alpha"] = oos["S&P 500"] + oos["Our: IC-Weighted (factors only)"]
elif "S&P 500" in oos.columns:
    ic_col = [c for c in oos.columns if "IC-Weighted" in c and c.startswith("Our:")]
    if ic_col:
        oos["Our: Market + Factor Alpha"] = oos["S&P 500"] + oos[ic_col[0]]

# Categorize columns
our_names = [c for c in oos.columns if c.startswith("Our:")]
bench_names = [c for c in oos.columns if not c.startswith("Our:")]
print(f"Our portfolios: {len(our_names)}, Benchmarks: {len(bench_names)}")

# Colors for our portfolios (thick colored) and benchmarks (thin gray for individuals)
OUR_COLORS = {"Equal Weight": "#2563eb", "IC-Weighted": "#0ea5e9", "MVO": "#7c3aed",
              "Max Sharpe": "#db2777", "Risk Parity": "#059669", "MAXSER": "#0d9488",
              "Market + Factor Alpha": "#0d9488"}

def _get_our_color(name):
    for k, c in OUR_COLORS.items():
        if k in name:
            return c
    return "#3b82f6"

# ── Chart 1: Cumulative Returns Comparison ──
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=["All Portfolios vs Benchmarks (Out-of-Sample Cumulative Returns)", "Drawdowns"],
)

# Plot individual benchmarks first (thin gray lines)
for name in bench_names:
    r = oos[name].dropna()
    if len(r) < 6:
        continue
    cum = (1 + r).cumprod()
    dates = r.index.to_timestamp()
    # Key benchmarks get color; individual funds get thin gray
    is_key = any(k in name for k in ["S&P 500", "(EW)", "FF "])
    color = "#dc2626" if "S&P 500" in name else "#f59e0b" if "Mutual Fund (EW)" in name else \
            "#84cc16" if "Smart Beta (EW)" in name else "#f97316" if "Hedge Fund Index (EW)" in name else \
            "#737373" if "FF " in name else "#d4d4d4"
    lw = 1.5 if is_key else 0.8
    dash = "dot" if "FF " in name else None

    fig.add_trace(go.Scatter(
        x=dates, y=cum.values,
        mode="lines", name=name,
        line=dict(color=color, width=lw, dash=dash),
        legendgroup=name, opacity=0.6 if not is_key else 1.0,
    ), row=1, col=1)

# Plot our portfolios on top (thick colored lines)
for name in our_names:
    r = oos[name].dropna()
    if len(r) < 6:
        continue
    cum = (1 + r).cumprod()
    dates = r.index.to_timestamp()
    color = _get_our_color(name)
    is_bl = "+ BL" in name
    dash = "dash" if is_bl else "solid"

    fig.add_trace(go.Scatter(
        x=dates, y=cum.values,
        mode="lines", name=name,
        line=dict(color=color, width=2.5, dash=dash),
        legendgroup=name,
    ), row=1, col=1)

    # Drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    fig.add_trace(go.Scatter(
        x=dates, y=dd.values,
        mode="lines", name=name,
        line=dict(color=color, width=1, dash=dash),
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

# ── Chart 2: All strategies bar chart (Sharpe ratios, color-coded by category) ──
comp = pd.read_csv(tables / "s3_benchmark_comparison.csv", index_col=0)
comp_sorted = comp.sort_values("Sharpe", ascending=True)

# Color-code: blue = our portfolios, orange = mutual funds, green = smart beta,
# red = HF indices, gray = FF factors, light gray = S&P 500
def _bar_color(name):
    if name.startswith("Our:"):
        return "#2563eb"
    if "Mutual Fund" in name or name in ("FSMEX", "FLPSX", "FCNTX", "FSPHX", "FBALX"):
        return "#f59e0b"
    if "Smart Beta" in name or name in ("PSL", "MTUM", "USMV", "QUAL", "VLUE", "SIZE"):
        return "#84cc16"
    if "Hedge Fund" in name or "HFRI" in name:
        return "#ef4444"
    if "FF " in name:
        return "#a3a3a3"
    if "S&P 500" in name:
        return "#78716c"
    # Individual funds — try to categorize
    return "#d4d4d4"

colors_bar = [_bar_color(name) for name in comp_sorted.index]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y=comp_sorted.index,
    x=comp_sorted["Sharpe"],
    orientation="h",
    marker_color=colors_bar,
    text=[f"{v:.3f}" for v in comp_sorted["Sharpe"]],
    textposition="outside",
))

# Add zero line
fig2.add_vline(x=0, line_color="black", line_width=0.5)

fig2.update_layout(
    title="Out-of-Sample Sharpe Ratio Comparison (All Portfolios & Benchmarks)",
    xaxis_title="Annualized Sharpe Ratio",
    template="plotly_white",
    height=max(600, len(comp_sorted) * 22),
    width=1000,
    font=dict(size=12),
    margin=dict(l=280),
)

fig2.write_html(str(fig_path / "sharpe_comparison_bar.html"), include_plotlyjs="cdn")
try:
    fig2.write_image(str(fig_path / "sharpe_comparison_bar.png"),
                     width=1000, height=max(600, len(comp_sorted) * 22), scale=2)
except Exception as e:
    print(f"  PNG export failed (kaleido): {e}")

print(f"Saved to {fig_path / 'sharpe_comparison_bar.html'}")
print("Done!")
