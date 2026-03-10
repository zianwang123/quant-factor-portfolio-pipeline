import numpy as np
import pandas as pd
from typing import Optional


def build_factor_view(
    factor_qspread_history: pd.Series,
    winner_stocks: list,
    loser_stocks: list,
    all_stocks: list,
) -> tuple[np.ndarray, float, float]:
    """Construct BL view matrices from a factor's long-short portfolio.

    P_row: +1/n_long for winners, -1/n_short for losers, 0 for others.
    Q: Average historical QSpread (expected return of the view).
    omega: Variance of historical QSpread (uncertainty of the view).

    Returns:
        (P_row, Q, omega) where P_row is (1, N), Q is scalar, omega is scalar.
    """
    n_stocks = len(all_stocks)
    P = np.zeros(n_stocks)

    stock_to_idx = {s: i for i, s in enumerate(all_stocks)}
    n_long = len([s for s in winner_stocks if s in stock_to_idx])
    n_short = len([s for s in loser_stocks if s in stock_to_idx])

    for s in winner_stocks:
        if s in stock_to_idx:
            P[stock_to_idx[s]] = 1.0 / n_long

    for s in loser_stocks:
        if s in stock_to_idx:
            P[stock_to_idx[s]] = -1.0 / n_short

    Q = np.nanmean(factor_qspread_history.values)
    omega = np.nanvar(factor_qspread_history.values)

    return P, Q, omega


def build_multi_view(
    views: list[tuple[np.ndarray, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine multiple factor views into BL matrices.

    Args:
        views: List of (P_row, Q, omega) tuples from build_factor_view.

    Returns:
        P: (K, N) pick matrix.
        Q: (K,) expected returns vector.
        Omega: (K, K) diagonal uncertainty matrix.
    """
    K = len(views)
    N = views[0][0].shape[0]

    P = np.zeros((K, N))
    Q_vec = np.zeros(K)
    omega_diag = np.zeros(K)

    for k, (p_row, q, omega) in enumerate(views):
        P[k, :] = p_row
        Q_vec[k] = q
        omega_diag[k] = omega

    Omega = np.diag(omega_diag)

    return P, Q_vec, Omega


def build_ic_weighted_view(
    P_row: np.ndarray,
    Q: float,
    omega: float,
    ic: float,
    ic_scale: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Adjust view confidence based on Information Coefficient.

    Higher IC -> lower omega (more confident).
    omega_adjusted = omega / (ic_scale * |IC|)
    """
    ic_abs = abs(ic)
    if ic_abs < 0.01:
        adjusted_omega = omega * 100  # Very uncertain
    else:
        adjusted_omega = omega / (ic_scale * ic_abs)

    return P_row, Q, adjusted_omega
