import numpy as np
import pandas as pd


def compute_ic_series(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    is_sp500: pd.DataFrame,
) -> pd.Series:
    """Compute monthly Information Coefficient (rank correlation) for a factor.

    IC_t = Spearman correlation between factor exposures at t and returns at t+1,
    computed cross-sectionally across S&P 500 members.

    Avoids pandas Index.intersection() which segfaults on Windows + Python 3.12.
    """
    ic_values = {}

    for t in factor.index:
        t_plus_1 = t + 1
        if t_plus_1 not in returns.index:
            continue

        # Filter to S&P 500
        if t not in is_sp500.columns:
            continue
        sp500_members = is_sp500[is_sp500[t] == 1].index

        f_vals = factor.loc[t].reindex(sp500_members).dropna()
        r_vals = returns.loc[t_plus_1].reindex(sp500_members).dropna()

        # Use numpy set intersection to avoid pandas Index.intersection segfault
        common = np.intersect1d(f_vals.index.values, r_vals.index.values)
        if len(common) < 20:
            continue

        f = f_vals.reindex(common).replace([np.inf, -np.inf], np.nan).dropna()
        r = r_vals.reindex(f.index)

        # Drop any remaining NaN
        valid = f.notna() & r.notna()
        f = f[valid]
        r = r[valid]

        if len(f) < 20:
            continue

        # Manual Spearman: rank then Pearson (avoids scipy.stats.spearmanr segfault)
        f_ranks = f.values.argsort().argsort().astype(float)
        r_ranks = r.values.argsort().argsort().astype(float)
        corr = np.corrcoef(f_ranks, r_ranks)[0, 1]
        ic_values[t_plus_1] = corr

    return pd.Series(ic_values, dtype=float)


def ic_summary(ic_series: pd.Series) -> dict:
    """Compute IC summary statistics."""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {}

    return {
        "Mean IC": ic.mean(),
        "Std IC": ic.std(),
        "IC IR": ic.mean() / ic.std() if ic.std() > 0 else 0.0,
        "Hit Rate": (ic > 0).mean(),
        "Max IC": ic.max(),
        "Min IC": ic.min(),
        "N Months": len(ic),
    }


def ic_decay_analysis(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    is_sp500: pd.DataFrame,
    max_lag: int = 12,
) -> pd.DataFrame:
    """Compute IC at different holding horizons to measure signal persistence."""
    decay = {}

    for lag in range(1, max_lag + 1):
        shifted_returns = returns.shift(-lag)
        ic_series = compute_ic_series(factor, shifted_returns, is_sp500)
        summary = ic_summary(ic_series)
        if summary:
            decay[lag] = {
                "Mean IC": summary["Mean IC"],
                "IC IR": summary["IC IR"],
            }

    return pd.DataFrame(decay).T
