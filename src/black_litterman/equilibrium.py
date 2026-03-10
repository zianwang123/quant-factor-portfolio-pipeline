import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    delta: float,
    sigma: np.ndarray,
    w_mkt: np.ndarray,
) -> np.ndarray:
    """Compute implied equilibrium returns via reverse optimization.

    pi = delta * Sigma * w_mkt

    These are the returns that make the market portfolio optimal under
    mean-variance with risk aversion delta.

    Args:
        delta: Risk aversion coefficient.
        sigma: Covariance matrix (N x N).
        w_mkt: Market capitalization weights (N,).

    Returns:
        Equilibrium expected returns vector (N,).
    """
    return delta * sigma @ w_mkt


def market_cap_weights(market_caps: pd.Series) -> pd.Series:
    """Compute market-capitalization weights from market caps."""
    total = market_caps.sum()
    if total <= 0:
        raise ValueError("Total market cap must be positive.")
    return market_caps / total
