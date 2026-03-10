import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# NBER recession dates (start, end) for shading
NBER_RECESSIONS = [
    ("1970-01", "1970-11"),
    ("1973-12", "1975-03"),
    ("1980-02", "1980-07"),
    ("1981-08", "1982-11"),
    ("1990-08", "1991-03"),
    ("2001-04", "2001-11"),
    ("2008-01", "2009-06"),
    ("2020-03", "2020-04"),
]

# Color palette
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "long": "#2ca02c",
    "short": "#d62728",
    "benchmark": "#1f77b4",
    "portfolio": "#ff7f0e",
}

QUINTILE_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]


def set_style():
    """Apply consistent publication-quality style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def add_recession_shading(ax, start_date=None, end_date=None):
    """Add NBER recession shading to a matplotlib axis."""
    import pandas as pd

    for rec_start, rec_end in NBER_RECESSIONS:
        rs = pd.Period(rec_start, freq="M")
        re = pd.Period(rec_end, freq="M")

        if start_date and rs < pd.Period(start_date, freq="M"):
            rs = pd.Period(start_date, freq="M")
        if end_date and re > pd.Period(end_date, freq="M"):
            re = pd.Period(end_date, freq="M")

        ax.axvspan(rs, re, alpha=0.15, color="gray", label=None)
