import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional


def mean_variance_optimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 10.0,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    gross_leverage: Optional[float] = None,
    target_factor_exposures: Optional[dict] = None,
    factor_loadings: Optional[np.ndarray] = None,
    tracking_error_limit: Optional[float] = None,
    benchmark_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Flexible mean-variance portfolio optimizer using CVXPY.

    Solves: max  w'mu - (delta/2) * w'Sigma*w
            s.t. sum(w) = 1
                 + optional constraints

    Args:
        mu: Expected returns vector (N,).
        sigma: Covariance matrix (N, N).
        risk_aversion: Lambda parameter for risk-return tradeoff.
        long_only: If True, w >= 0.
        max_weight: Upper bound on individual weights.
        min_weight: Lower bound on individual weights (ignored if long_only).
        gross_leverage: Max sum(|w|), e.g., 1.5 for 150% gross.
        target_factor_exposures: dict of {factor_name: target_value} for factor neutrality.
        factor_loadings: (N, K) matrix of factor betas (required if target_factor_exposures).
        tracking_error_limit: Max annualized tracking error vs benchmark.
        benchmark_weights: Benchmark weights for tracking error constraint.

    Returns:
        Optimal weight vector (N,).
    """
    n = len(mu)
    w = cp.Variable(n)

    # Objective: maximize utility = w'mu - (delta/2) * w'Sigma*w
    ret = mu @ w
    risk = cp.quad_form(w, sigma)
    objective = cp.Maximize(ret - (risk_aversion / 2) * risk)

    # Constraints
    constraints = [cp.sum(w) == 1]

    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= max_weight)
    else:
        constraints.append(w >= min_weight)
        constraints.append(w <= max_weight)

    if gross_leverage is not None:
        constraints.append(cp.norm(w, 1) <= gross_leverage)

    if target_factor_exposures is not None and factor_loadings is not None:
        for k, (fname, target) in enumerate(target_factor_exposures.items()):
            constraints.append(factor_loadings[:, k] @ w == target)

    if tracking_error_limit is not None and benchmark_weights is not None:
        te_monthly = tracking_error_limit / np.sqrt(12)
        active = w - benchmark_weights
        constraints.append(cp.quad_form(active, sigma) <= te_monthly**2)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError(f"Optimization failed: {prob.status}")

    return w.value


def global_minimum_variance(
    sigma: np.ndarray,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Find the Global Minimum Variance portfolio.

    Solves: min  w'Sigma*w
            s.t. sum(w) = 1
    """
    n = sigma.shape[0]
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = [cp.sum(w) == 1]

    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= max_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError(f"GMV optimization failed: {prob.status}")

    return w.value


def max_sharpe_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Find the Maximum Sharpe Ratio (tangency) portfolio.

    Uses the reformulation: min w'Sigma*w  s.t. (mu-rf)'w = 1, w >= 0
    Then normalize weights to sum to 1.
    """
    n = len(mu)
    w = cp.Variable(n)
    excess = mu - rf

    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = [excess @ w == 1]

    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError(f"Max Sharpe optimization failed: {prob.status}")

    weights = w.value
    return weights / weights.sum()


def risk_parity(sigma: np.ndarray, budget: Optional[np.ndarray] = None) -> np.ndarray:
    """Risk Parity / Equal Risk Contribution portfolio.

    Each asset contributes equally to total portfolio risk.
    Uses Spinu (2013) convex reformulation.
    """
    n = sigma.shape[0]
    if budget is None:
        budget = np.ones(n) / n

    y = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(y, sigma) - budget @ cp.log(y))
    constraints = [y >= 1e-6]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    weights = y.value
    return weights / weights.sum()
