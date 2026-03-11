import numpy as np
import pandas as pd
from numpy.linalg import inv
from src.black_litterman.equilibrium import implied_equilibrium_returns


def black_litterman_posterior(
    delta: float,
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> dict:
    """Compute Black-Litterman posterior distribution and optimal weights.

    The BL model combines equilibrium returns (CAPM prior) with investor views:

    mu_bar = inv(inv(tau*Sigma) + P'*inv(Omega)*P) *
             (inv(tau*Sigma)*pi + P'*inv(Omega)*Q)

    Sigma_bar = Sigma + inv(inv(tau*Sigma) + P'*inv(Omega)*P)

    w* = (1/delta) * inv(Sigma_bar) * mu_bar

    Args:
        delta: Risk aversion parameter.
        sigma: (N, N) covariance matrix of returns.
        w_mkt: (N,) market capitalization weights.
        tau: Scalar uncertainty of CAPM prior.
        P: (K, N) pick matrix identifying view portfolios.
        Q: (K,) expected returns for each view.
        Omega: (K, K) diagonal uncertainty matrix for views.

    Returns:
        dict with: mu_posterior, sigma_posterior, weights, pi (prior returns).
    """
    # Prior equilibrium returns
    pi = implied_equilibrium_returns(delta, sigma, w_mkt)

    # Precision matrices
    tau_sigma_inv = inv(tau * sigma)
    omega_inv = inv(Omega)

    # Posterior precision
    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
    posterior_cov = inv(posterior_precision)

    # Posterior mean
    mu_bar = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    # Combined covariance: Σ̄ = Σ + M⁻¹ (He & Litterman 1999, Eq. 9)
    # Asset covariance + estimation uncertainty from posterior
    sigma_bar = sigma + posterior_cov

    # Optimal weights
    w_star = (1.0 / delta) * inv(sigma_bar) @ mu_bar
    w_star = w_star / w_star.sum()  # Normalize to fully invested

    return {
        "mu_posterior": mu_bar,
        "sigma_posterior": sigma_bar,
        "weights": w_star,
        "pi": pi,
        "posterior_cov": posterior_cov,
    }


def black_litterman_no_views(
    delta: float,
    sigma: np.ndarray,
    w_mkt: np.ndarray,
) -> dict:
    """BL model with no views (returns equilibrium portfolio)."""
    pi = implied_equilibrium_returns(delta, sigma, w_mkt)
    w_star = w_mkt.copy()

    return {
        "mu_posterior": pi,
        "sigma_posterior": sigma,
        "weights": w_star,
        "pi": pi,
    }
