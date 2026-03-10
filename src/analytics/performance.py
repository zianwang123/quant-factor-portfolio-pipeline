import numpy as np
import pandas as pd


def compute_descriptive_stats(returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    """Compute full descriptive statistics for factor/portfolio returns.

    Args:
        returns: DataFrame where each column is a return series (in percentage or decimal).
        annualize: If True, annualize mean and std by sqrt(12).
    """
    stats = {}

    for col in returns.columns:
        ret = returns[col].dropna()
        if len(ret) == 0:
            continue

        mean = ret.mean()
        std = ret.std()
        sr = mean / std if std > 0 else 0.0

        if annualize:
            mean *= 12
            std *= np.sqrt(12)
            sr *= np.sqrt(12)

        stats[col] = {
            "Mean": mean,
            "Std": std,
            "Sharpe Ratio": sr,
            "Skewness": pd.Series(ret).skew(),
            "Kurtosis": pd.Series(ret).kurtosis(),
            "Max": ret.max(),
            "Min": ret.min(),
            "Max Drawdown": max_drawdown(ret),
            "ACF(1)": ret.autocorr(lag=1) if len(ret) > 1 else np.nan,
            "ACF(12)": ret.autocorr(lag=12) if len(ret) > 12 else np.nan,
            "ACF(24)": ret.autocorr(lag=24) if len(ret) > 24 else np.nan,
        }

    return pd.DataFrame(stats)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return series.

    Auto-detects whether returns are in percentage points or decimals.
    If mean absolute return > 1, assumes percentage points and divides by 100.
    """
    r = returns.copy()
    if r.abs().mean() > 1:
        r = r / 100.0
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns from a simple return series."""
    return (1 + returns).cumprod()


def rolling_sharpe(returns: pd.Series, window: int = 36) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(12)


def calmar_ratio(returns: pd.Series) -> float:
    """Annualized return / abs(max drawdown)."""
    ann_ret = returns.mean() * 12
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else np.nan


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """Sortino ratio using downside deviation."""
    excess = returns - target
    downside = returns[returns < target]
    downside_std = downside.std()
    return excess.mean() / downside_std if downside_std > 0 else np.nan


def t_test_factor_spreads(qspreads: pd.DataFrame) -> pd.DataFrame:
    """Test whether each factor's mean QSpread is significantly different from zero.

    Uses one-sample t-test with normal approximation.
    """
    results = {}

    for col in qspreads.columns:
        ret = qspreads[col].dropna()
        n = len(ret)
        if n < 2:
            continue

        mean = ret.mean()
        std = ret.std()
        t_stat = abs(mean / (std / np.sqrt(n)))
        # Normal CDF via math.erfc (avoids scipy import)
        import math
        p_value = math.erfc(t_stat / math.sqrt(2))  # Two-sided test

        results[col] = {
            "Mean Spread": mean,
            "t-statistic": t_stat,
            "p-value": p_value,
            "Significant (5%)": p_value < 0.05,
            "N": n,
        }

    return pd.DataFrame(results).T


def compute_turnover_stats(turnover_results: dict[str, pd.DataFrame]) -> pd.Series:
    """Compute average turnover for each factor."""
    avg = {}
    for name, df in turnover_results.items():
        if not df.empty:
            avg[name] = df.mean().mean()
    return pd.Series(avg, name="Avg Turnover")
