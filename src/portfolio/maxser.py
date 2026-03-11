"""MAXSER: Maximum Sharpe-ratio Estimated and Sparse Regression.

Implements Ao, Li, Zheng (Review of Financial Studies, 2019).
Reformulates mean-variance optimization as an unconstrained regression,
then applies LASSO/Ridge with cross-validation to control estimation risk.

Supports two scenarios:
  - Scenario 1: Individual assets only
  - Scenario 2: Factor investing allowed (factor + stock decomposition)
"""
import numpy as np
from numpy.linalg import inv, norm
from scipy.special import betainc, beta as beta_func
from scipy.optimize import brentq
from sklearn.linear_model import LassoLars, Ridge
import statsmodels.api as sm


def _theta_sample(mu_hat: np.ndarray, sigma_hat: np.ndarray) -> float:
    """Compute sample squared max Sharpe: theta_s = mu' Sigma^{-1} mu."""
    return mu_hat @ inv(sigma_hat) @ mu_hat


def _theta_adjusted(theta_s: float, N: int, T: int) -> float:
    """Kan-Zhou (2007) bias-corrected theta estimator (Eq 1.32 in paper).

    The naive estimator (T-N-2)*theta_s - N)/T can go negative.
    This adjusted version adds a correction term using the incomplete beta function.
    """
    a = N / 2
    b = (T - N) / 2
    if b <= 0 or a <= 0:
        # Fallback to simple estimator when T <= N
        return max(((T - N - 2) * theta_s - N) / T, 1e-6)

    simple = ((T - N - 2) * theta_s - N) / T

    # Beta function correction term
    x = theta_s / (1 + theta_s)
    try:
        Bx = betainc(a, b, x) * beta_func(a, b)
        if Bx > 0:
            correction = 2 * theta_s**a * (1 + theta_s)**(-(T - 2) / 2) / (T * Bx)
        else:
            correction = 0
    except (ValueError, ZeroDivisionError, OverflowError):
        correction = 0

    return max(simple + correction, 1e-6)


def _compute_response(theta_adj: float, sigma_target: float) -> float:
    """Compute the regression response r_c = sigma * (1 + theta) / sqrt(theta)."""
    return sigma_target * (1 + theta_adj) / np.sqrt(theta_adj)


def _lasso_solution_path(X: np.ndarray, y: np.ndarray):
    """Compute full LASSO solution path using LARS algorithm.

    Returns:
        coef_path: (N, n_steps) array of coefficients along the path
        zeta_path: (n_steps,) array of l1-norm ratios ||w||_1 / ||w_ols||_1
    """
    reg = LassoLars(alpha=0, fit_intercept=False, fit_path=True)
    reg.fit(X, y)
    coef_path = reg.coef_path_  # (N, n_steps)

    # Compute zeta = ||w||_1 / ||w_ols||_1
    ols_norm = norm(coef_path[:, -1], 1)
    if ols_norm < 1e-12:
        ols_norm = 1e-12
    zeta_path = norm(coef_path, 1, axis=0) / ols_norm

    return coef_path, zeta_path


def _select_by_risk(coef_path: np.ndarray, sigma_cov: np.ndarray,
                    sigma_target: float) -> int:
    """Select the portfolio on the path closest to the risk constraint."""
    risks = np.array([
        np.sqrt(coef_path[:, i] @ sigma_cov @ coef_path[:, i])
        for i in range(coef_path.shape[1])
    ])
    return np.argmin(np.abs(risks - sigma_target))


def _interpolate_weights(coef_path: np.ndarray, zeta_path: np.ndarray,
                         target_zeta: float) -> np.ndarray:
    """Interpolate weights on the LARS path at a given zeta value."""
    if target_zeta >= zeta_path[-1]:
        return coef_path[:, -1]
    if target_zeta <= zeta_path[0]:
        return coef_path[:, 0]

    # Find the first index where zeta > target_zeta
    idx = np.searchsorted(zeta_path, target_zeta)
    if idx == 0:
        return coef_path[:, 0]

    # Linear interpolation
    q = (target_zeta - zeta_path[idx - 1]) / (zeta_path[idx] - zeta_path[idx - 1] + 1e-12)
    return (1 - q) * coef_path[:, idx - 1] + q * coef_path[:, idx]


def maxser_lasso(
    returns: np.ndarray,
    sigma_target: float,
    n_folds: int = 10,
    population_cov: np.ndarray = None,
) -> np.ndarray:
    """MAXSER with LASSO (Scenario 1: individual assets only).

    Args:
        returns: (T, N) matrix of asset excess returns.
        sigma_target: Target monthly portfolio risk (standard deviation).
        n_folds: Number of CV folds.
        population_cov: If provided, use this for risk evaluation (simulation).
                       If None, use validation set sample covariance.

    Returns:
        Optimal weight vector (N,).
    """
    T, N = returns.shape
    fold_size = T // n_folds

    # Step 1: K-fold CV to find optimal zeta
    best_zetas = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        val_set = returns[start:end]
        train_set = np.vstack([returns[:start], returns[end:]])

        T_train = train_set.shape[0]

        # Estimate theta on training set
        mu_train = train_set.mean(axis=0)
        cov_train = np.cov(train_set, rowvar=False, ddof=0)
        theta_s = _theta_sample(mu_train, cov_train)
        theta_adj = _theta_adjusted(theta_s, N, T_train)

        # Compute response
        rc = _compute_response(theta_adj, sigma_target)
        y_train = rc * np.ones(T_train)

        # Get solution path
        try:
            coef_path, zeta_path = _lasso_solution_path(train_set, y_train)
        except Exception:
            continue

        # Evaluate on validation set
        val_cov = np.cov(val_set, rowvar=False, ddof=0)
        best_idx = _select_by_risk(coef_path, val_cov, sigma_target)
        best_zetas.append(zeta_path[best_idx])

    if not best_zetas:
        # Fallback to OLS if CV fails
        mu_hat = returns.mean(axis=0)
        cov_hat = np.cov(returns, rowvar=False, ddof=0)
        theta_s = _theta_sample(mu_hat, cov_hat)
        return (sigma_target / np.sqrt(theta_s)) * inv(cov_hat) @ mu_hat

    avg_zeta = np.mean(best_zetas)

    # Step 2: Fit on full sample with average zeta
    mu_hat = returns.mean(axis=0)
    cov_hat = np.cov(returns, rowvar=False, ddof=0)
    theta_s = _theta_sample(mu_hat, cov_hat)
    theta_adj = _theta_adjusted(theta_s, N, T)

    rc = _compute_response(theta_adj, sigma_target)
    y_full = rc * np.ones(T)

    coef_path, zeta_path = _lasso_solution_path(returns, y_full)
    weights = _interpolate_weights(coef_path, zeta_path, avg_zeta)

    return weights


def maxser_ridge(
    returns: np.ndarray,
    sigma_target: float,
    n_folds: int = 10,
) -> np.ndarray:
    """MAXSER with Ridge regression (Scenario 1).

    Uses Brent's method to find the Ridge alpha that makes OOS risk = sigma_target.

    Args:
        returns: (T, N) matrix of asset excess returns.
        sigma_target: Target monthly portfolio risk.
        n_folds: Number of CV folds.

    Returns:
        Optimal weight vector (N,).
    """
    T, N = returns.shape
    fold_size = T // n_folds

    best_alphas = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        val_set = returns[start:end]
        train_set = np.vstack([returns[:start], returns[end:]])

        T_train = train_set.shape[0]
        mu_train = train_set.mean(axis=0)
        cov_train = np.cov(train_set, rowvar=False, ddof=0)
        theta_s = _theta_sample(mu_train, cov_train)
        theta_adj = _theta_adjusted(theta_s, N, T_train)

        rc = _compute_response(theta_adj, sigma_target)
        y_train = rc * np.ones(T_train)

        val_cov = np.cov(val_set, rowvar=False, ddof=0)

        # Check if OLS already satisfies risk constraint
        try:
            ols_w = sm.OLS(y_train, train_set).fit().params
            ols_risk = np.sqrt(ols_w @ val_cov @ ols_w)
        except Exception:
            best_alphas.append(0)
            continue

        if ols_risk <= sigma_target:
            best_alphas.append(0)
            continue

        # Brent's method: find alpha where OOS risk = sigma_target
        def risk_residual(alpha):
            reg = Ridge(alpha=alpha, fit_intercept=False)
            reg.fit(train_set, y_train)
            w = reg.coef_
            return np.sqrt(w @ val_cov @ w) - sigma_target

        try:
            optimal_alpha = brentq(risk_residual, 0, 1000, xtol=1e-6)
            best_alphas.append(optimal_alpha)
        except (ValueError, RuntimeError):
            best_alphas.append(0)

    avg_alpha = np.mean(best_alphas) if best_alphas else 0

    # Fit on full sample
    mu_hat = returns.mean(axis=0)
    cov_hat = np.cov(returns, rowvar=False, ddof=0)
    theta_s = _theta_sample(mu_hat, cov_hat)
    theta_adj = _theta_adjusted(theta_s, N, T)

    rc = _compute_response(theta_adj, sigma_target)
    y_full = rc * np.ones(T)

    # Ridge closed-form: w = (X'X + alpha*I)^{-1} X'y
    weights = inv(returns.T @ returns + avg_alpha * np.eye(N)) @ (returns.T @ y_full)

    return weights


def maxser_scenario2(
    stock_returns: np.ndarray,
    factor_returns: np.ndarray,
    sigma_target: float,
    n_folds: int = 10,
    method: str = "lasso",
    subpool_size: int = None,
    n_subpools: int = 1000,
) -> dict:
    """MAXSER Scenario 2: Factor + stock investing.

    Decomposes the optimal portfolio into factor and idiosyncratic components
    per Proposition 3 of Ao, Li, Zheng (2019).

    Args:
        stock_returns: (T, N) matrix of individual stock excess returns.
        factor_returns: (T, K) matrix of factor excess returns.
        sigma_target: Target monthly portfolio risk.
        n_folds: Number of CV folds for LASSO/Ridge on idiosyncratic component.
        method: "lasso" or "ridge" for the idiosyncratic component.
        subpool_size: If set, select a subpool of stocks (Section 1.5.3).
        n_subpools: Number of random subpools to evaluate.

    Returns:
        dict with keys:
            "w_factors": (K,) factor portfolio weights
            "w_stocks": (N,) stock portfolio weights (sparse for LASSO)
            "w_factors_raw": (K,) raw factor weights before beta adjustment
            "beta": (N, K) estimated factor loadings
            "n_nonzero_stocks": number of stocks with nonzero weight
            "theta_f", "theta_u", "theta_all": squared Sharpe ratios
    """
    T, N = stock_returns.shape
    K = factor_returns.shape[1]

    # Step 0: Subpool selection (if needed)
    selected_idx = np.arange(N)
    if subpool_size is not None and subpool_size < N:
        selected_idx = _select_subpool(
            stock_returns, factor_returns, sigma_target,
            subpool_size, n_subpools, T
        )
        stock_returns = stock_returns[:, selected_idx]
        N = stock_returns.shape[1]

    # Step 1: Regress stocks on factors to get betas and idiosyncratic returns
    # r_i = alpha_i + sum(beta_ij * f_j) + e_i
    beta_hat = np.zeros((N, K))
    for i in range(N):
        valid = ~np.isnan(stock_returns[:, i])
        if valid.sum() < K + 2:
            continue
        X = factor_returns[valid]
        y = stock_returns[valid, i]
        try:
            reg = sm.OLS(y, sm.add_constant(X)).fit()
            beta_hat[i] = reg.params[1:]  # skip intercept
        except Exception:
            pass

    # Idiosyncratic returns: U = R - F * beta'
    U_hat = stock_returns - factor_returns @ beta_hat.T

    # Step 2: Estimate theta_f (plug-in, K is small)
    mu_f = factor_returns.mean(axis=0)
    cov_f = np.cov(factor_returns, rowvar=False, ddof=0)
    theta_f = mu_f @ inv(cov_f) @ mu_f

    # Step 3: Estimate theta_all with bias correction
    all_returns = np.hstack([factor_returns, stock_returns])
    mu_all = all_returns.mean(axis=0)
    cov_all = np.cov(all_returns, rowvar=False, ddof=0)
    theta_s_all = mu_all @ inv(cov_all) @ mu_all
    N_all = K + N
    theta_all = _theta_adjusted(theta_s_all, N_all, T)

    # Step 4: theta_u = theta_all - theta_f
    theta_u = max(theta_all - theta_f, 1e-6)

    # Step 5: Compute response for idiosyncratic component
    rc_u = sigma_target * (1 + theta_u) / np.sqrt(theta_u)

    # Step 6: Apply MAXSER (LASSO or Ridge) to idiosyncratic returns
    if method == "lasso":
        w_u = _maxser_idiosyncratic_lasso(U_hat, rc_u, sigma_target, n_folds)
    else:
        w_u = _maxser_idiosyncratic_ridge(U_hat, rc_u, sigma_target, n_folds)

    # Step 7: Factor portfolio (plug-in, K is small)
    w_f_raw = (1 / np.sqrt(theta_f)) * inv(cov_f) @ mu_f

    # Step 8: Combine per Proposition 3 (Eq 1.30)
    # w_u from MAXSER regression is already sigma-scaled (rc includes sigma_target),
    # so divide by sigma_target to get the unit tangency direction w_u*
    w_u_unit = w_u / sigma_target if sigma_target > 1e-12 else w_u

    scale_f = np.sqrt(theta_f / theta_all)
    scale_u = np.sqrt(theta_u / theta_all)

    w_factors = sigma_target * (scale_f * w_f_raw - scale_u * beta_hat.T @ w_u_unit)
    w_stocks = sigma_target * scale_u * w_u_unit

    # Note: w_stocks has length = subpool size (N after line 304).
    # The caller maps back to the full universe using selected_stock_idx.

    return {
        "w_factors": w_factors,
        "w_stocks": w_stocks,
        "w_factors_raw": w_f_raw,
        "beta": beta_hat,
        "n_nonzero_stocks": np.sum(np.abs(w_stocks) > 1e-8),
        "theta_f": theta_f,
        "theta_u": theta_u,
        "theta_all": theta_all,
        "selected_stock_idx": selected_idx,
    }


def _maxser_idiosyncratic_lasso(U: np.ndarray, rc: float, sigma_target: float,
                                 n_folds: int) -> np.ndarray:
    """Apply MAXSER-LASSO to idiosyncratic returns."""
    T, N = U.shape
    fold_size = T // n_folds

    best_zetas = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        val_set = U[start:end]
        train_set = np.vstack([U[:start], U[end:]])
        T_train = train_set.shape[0]

        # Re-estimate theta_u on training set
        mu_train = train_set.mean(axis=0)
        cov_train = np.cov(train_set, rowvar=False, ddof=0)
        theta_s = _theta_sample(mu_train, cov_train)
        theta_adj = _theta_adjusted(theta_s, N, T_train)
        rc_train = sigma_target * (1 + theta_adj) / np.sqrt(theta_adj)

        y_train = rc_train * np.ones(T_train)

        try:
            coef_path, zeta_path = _lasso_solution_path(train_set, y_train)
        except Exception:
            continue

        val_cov = np.cov(val_set, rowvar=False, ddof=0)
        best_idx = _select_by_risk(coef_path, val_cov, sigma_target)
        best_zetas.append(zeta_path[best_idx])

    avg_zeta = np.mean(best_zetas) if best_zetas else 0.2

    # Fit on full sample
    y_full = rc * np.ones(T)
    coef_path, zeta_path = _lasso_solution_path(U, y_full)
    return _interpolate_weights(coef_path, zeta_path, avg_zeta)


def _maxser_idiosyncratic_ridge(U: np.ndarray, rc: float, sigma_target: float,
                                 n_folds: int) -> np.ndarray:
    """Apply MAXSER-Ridge to idiosyncratic returns."""
    T, N = U.shape
    fold_size = T // n_folds

    best_alphas = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        val_set = U[start:end]
        train_set = np.vstack([U[:start], U[end:]])
        T_train = train_set.shape[0]

        mu_train = train_set.mean(axis=0)
        cov_train = np.cov(train_set, rowvar=False, ddof=0)
        theta_s = _theta_sample(mu_train, cov_train)
        theta_adj = _theta_adjusted(theta_s, N, T_train)
        rc_train = sigma_target * (1 + theta_adj) / np.sqrt(theta_adj)

        y_train = rc_train * np.ones(T_train)
        val_cov = np.cov(val_set, rowvar=False, ddof=0)

        try:
            ols_w = sm.OLS(y_train, train_set).fit().params
            ols_risk = np.sqrt(ols_w @ val_cov @ ols_w)
        except Exception:
            best_alphas.append(0)
            continue

        if ols_risk <= sigma_target:
            best_alphas.append(0)
            continue

        def risk_residual(alpha):
            reg = Ridge(alpha=alpha, fit_intercept=False)
            reg.fit(train_set, y_train)
            w = reg.coef_
            return np.sqrt(w @ val_cov @ w) - sigma_target

        try:
            optimal_alpha = brentq(risk_residual, 0, 1000, xtol=1e-6)
            best_alphas.append(optimal_alpha)
        except (ValueError, RuntimeError):
            best_alphas.append(0)

    avg_alpha = np.mean(best_alphas) if best_alphas else 0

    y_full = rc * np.ones(T)
    return inv(U.T @ U + avg_alpha * np.eye(N)) @ (U.T @ y_full)


def _select_subpool(stock_returns: np.ndarray, factor_returns: np.ndarray,
                    sigma_target: float, subpool_size: int,
                    n_subpools: int = None, T: int = None) -> np.ndarray:
    """Select best subpool of stocks based on idiosyncratic information ratio.

    Deterministic, optimization-driven selection:
    1. Regress each stock on factors to extract idiosyncratic returns.
    2. Rank stocks by |idiosyncratic IR| = |mean(u_i)| / std(u_i).
    3. Select top `subpool_size` stocks — these have the strongest
       idiosyncratic alpha signals, which is exactly what MAXSER exploits
       in its Proposition 3 decomposition.
    """
    _T, N = stock_returns.shape
    K = factor_returns.shape[1]

    # Regress each stock on factors to get idiosyncratic returns
    X = sm.add_constant(factor_returns)
    idio_ir = np.zeros(N)

    for i in range(N):
        y = stock_returns[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < K + 5:
            idio_ir[i] = 0.0
            continue
        try:
            resid = sm.OLS(y[valid], X[valid]).fit().resid
            std_r = resid.std()
            if std_r > 1e-10:
                idio_ir[i] = abs(resid.mean()) / std_r
            else:
                idio_ir[i] = 0.0
        except Exception:
            idio_ir[i] = 0.0

    # Select top stocks by idiosyncratic IR
    selected = np.argsort(idio_ir)[::-1][:subpool_size]
    return np.sort(selected)
