import numpy as np
import pandas as pd


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


def build_views_with_prior_scaling(
    views: list[tuple[np.ndarray, float]],
    sigma: np.ndarray,
    tau: float,
    ic_values: dict[str, float] = None,
    view_names: list[str] = None,
    base_confidence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build BL view matrices with Omega scaled from the prior (He & Litterman 1999).

    Instead of using raw QSpread variance for Omega, scale each view's
    uncertainty proportional to tau * P_k @ Sigma @ P_k'. This ensures
    views operate on the same scale as the prior and have meaningful impact.

    Confidence adjustment:
        omega_k = (tau * P_k @ Sigma @ P_k') / confidence_k
        confidence_k = base_confidence * (1 + scale * |IC_k|)

    Higher IC -> higher confidence -> lower omega -> view has more impact.

    Args:
        views: List of (P_row, Q) tuples — P_row is (N,), Q is scalar.
        sigma: (N, N) covariance matrix.
        tau: BL prior uncertainty scalar.
        ic_values: Dict of factor_name -> IC for confidence weighting.
        view_names: Names corresponding to each view (for IC lookup).
        base_confidence: Base confidence multiplier (>1 means more confident).

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

    for k, (p_row, q) in enumerate(views):
        P[k, :] = p_row
        Q_vec[k] = q

        # View portfolio variance under the prior
        prior_var = tau * p_row @ sigma @ p_row

        # IC-based confidence scaling
        confidence = base_confidence
        if ic_values and view_names and view_names[k] in ic_values:
            ic = abs(ic_values[view_names[k]])
            # Scale: IC of 0.05 gives 2x confidence, IC of 0.10 gives 3x
            confidence *= (1.0 + 20.0 * ic)

        omega_diag[k] = prior_var / confidence

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
