"""Plot individual mutual fund and hedge fund performance vs our portfolio."""
import faulthandler
faulthandler.enable()

import os
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

def _flush(msg=""):
    if msg:
        print(msg)
    sys.stdout.flush()

config_start = "2013-01"
run_id = os.environ.get("PIPELINE_RUN_ID", "")
if run_id:
    run_base = PROJECT_ROOT / "outputs" / run_id
else:
    run_base = PROJECT_ROOT / "outputs"
tables = run_base / "tables"
fig_path = run_base / "figures" / "stage_3"
fig_path.mkdir(parents=True, exist_ok=True)

# ── Load our OOS returns ──
_flush("Loading OOS returns...")
oos = pd.read_csv(tables / "oos_returns_all.csv", index_col=0)
oos.index = pd.PeriodIndex(oos.index, freq="M")
our_ret = oos["Our: IC-Weighted"]
sp500_ret = oos["S&P 500"]
_flush(f"  OOS shape: {oos.shape}")

# ── Parse Excel sheets ──
_flush("Reading Excel...")
xl = pd.ExcelFile(PROJECT_ROOT / "data" / "raw" / "fama_french_factors.xlsx")


def parse_sheet(name):
    df = pd.read_excel(xl, name)
    df["Date"] = df["Date"].astype(str)
    df["Date"] = pd.PeriodIndex(
        df["Date"].str[:4] + "-" + df["Date"].str[4:6], freq="M"
    )
    df = df.set_index("Date")
    numeric = df.select_dtypes(include=[np.number])
    # auto-detect if returns are in pct points (>1 means pct)
    if numeric.abs().mean().mean() > 1:
        numeric = numeric / 100
    return numeric.loc[config_start:]


# ── Save intermediate CSVs for each sheet ──
for sheet_name in ["Mutual Fund", "Hedge Fund Index"]:
    _flush(f"\nParsing {sheet_name}...")
    df = parse_sheet(sheet_name)
    _flush(f"  Funds: {list(df.columns)}, shape: {df.shape}")

    # Build cumulative returns table for saving
    cum_data = {}

    # Our portfolio
    our = our_ret.loc[config_start:]
    cum_data["Our: IC-Weighted"] = (1 + our).cumprod()

    # S&P 500
    sp = sp500_ret.loc[config_start:]
    cum_data["S&P 500"] = (1 + sp).cumprod()

    # Each fund
    sharpe_dict = {}
    for col in df.columns:
        r = df[col].dropna()
        if len(r) < 12:
            _flush(f"  Skipping {col} (only {len(r)} months)")
            continue
        cum_data[col] = (1 + r).cumprod()
        sharpe_dict[col] = r.mean() / r.std() * np.sqrt(12)
        _flush(f"  {col}: Sharpe={sharpe_dict[col]:.3f}, cumulative={cum_data[col].iloc[-1]:.3f}")

    # Save CSV first (crash-safe)
    fname = sheet_name.lower().replace(" ", "_")
    cum_df = pd.DataFrame(cum_data)
    cum_df.to_csv(tables / f"{fname}_cumulative.csv")
    _flush(f"  Saved CSV: {fname}_cumulative.csv")

    # Save Sharpe summary
    sharpe_series = pd.Series(sharpe_dict, name="Sharpe")
    sharpe_series["Our: IC-Weighted"] = our.mean() / our.std() * np.sqrt(12)
    sharpe_series["S&P 500"] = sp.mean() / sp.std() * np.sqrt(12)
    sharpe_series.to_csv(tables / f"{fname}_sharpes.csv")
    _flush(f"  Saved CSV: {fname}_sharpes.csv")

gc.collect()

# ── Now plot from saved CSVs (separate step to survive crashes) ──
_flush("\n--- Generating plots ---")
import plotly.graph_objects as go

COLORS_FUND = [
    "#16a34a", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899",
    "#84cc16", "#f97316", "#6366f1", "#14b8a6", "#e11d48",
]

for sheet_name in ["Mutual Fund", "Hedge Fund Index"]:
    fname = sheet_name.lower().replace(" ", "_")
    _flush(f"\nPlotting {sheet_name}...")

    cum_df = pd.read_csv(tables / f"{fname}_cumulative.csv", index_col=0)
    cum_df.index = pd.PeriodIndex(cum_df.index, freq="M")
    sharpes = pd.read_csv(tables / f"{fname}_sharpes.csv", index_col=0, header=None)
    sharpes.columns = ["Sharpe"]

    fig = go.Figure()

    # Our portfolio (bold blue)
    dates = cum_df.index.to_timestamp()
    fig.add_trace(go.Scatter(
        x=dates, y=cum_df["Our: IC-Weighted"].values,
        mode="lines", name="Our: IC-Weighted",
        line=dict(color="#2563eb", width=3),
    ))

    # S&P 500 (dashed red)
    fig.add_trace(go.Scatter(
        x=dates, y=cum_df["S&P 500"].values,
        mode="lines", name="S&P 500",
        line=dict(color="#dc2626", width=2, dash="dash"),
    ))

    # Individual funds
    fund_cols = [c for c in cum_df.columns if c not in ("Our: IC-Weighted", "S&P 500")]
    for i, col in enumerate(fund_cols):
        sh = float(sharpes.loc[col, "Sharpe"]) if col in sharpes.index else 0.0
        fig.add_trace(go.Scatter(
            x=dates, y=cum_df[col].values,
            mode="lines", name=f"{col} (Sharpe:{sh:.2f})",
            line=dict(color=COLORS_FUND[i % len(COLORS_FUND)], width=1.5),
        ))

    fig.update_layout(
        title=f"{sheet_name}: Individual Fund Performance vs Our Portfolio (OOS 2013-2019)",
        yaxis_title="Cumulative Return (Growth of $1)",
        template="plotly_white",
        height=700, width=1200,
        font=dict(size=13),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
    )

    fig.write_html(str(fig_path / f"{fname}_individual.html"), include_plotlyjs="cdn")
    _flush(f"  Saved HTML: {fname}_individual.html")

    try:
        fig.write_image(str(fig_path / f"{fname}_individual.png"), width=1200, height=700, scale=2)
        _flush(f"  Saved PNG: {fname}_individual.png")
    except Exception as e:
        _flush(f"  PNG export failed: {e}")

    del fig
    gc.collect()

_flush("\nDone!")
