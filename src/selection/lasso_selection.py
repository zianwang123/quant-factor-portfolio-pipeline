import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def lasso_factor_selection(
    returns: pd.DataFrame,
    factor_exposures: dict[str, pd.DataFrame],
    is_sp500: pd.DataFrame,
    n_folds: int = 5,
    n_alphas: int = 100,
    test_start: str = "2010-01",
) -> dict:
    """Use LASSO to select factors with predictive power.

    Pools cross-sectional data across time (with proper time-series CV:
    only uses data before test_start for fitting).

    Returns:
        dict with selected factors, coefficients, and cross-validation results.
    """
    factor_names = list(factor_exposures.keys())

    # Build pooled dataset (in-sample only)
    X_rows = []
    y_rows = []

    for t in returns.index[:-1]:
        if str(t) >= test_start:
            break

        t_plus_1 = t + 1
        if t_plus_1 not in returns.index:
            continue

        if t not in is_sp500.columns:
            continue

        sp500_members = is_sp500[is_sp500[t] == 1].index
        r = returns.loc[t_plus_1].reindex(sp500_members).dropna()

        X_t = {}
        for fname in factor_names:
            f_df = factor_exposures[fname]
            if t in f_df.index:
                X_t[fname] = f_df.loc[t].reindex(sp500_members)

        if len(X_t) < len(factor_names):
            continue

        X_df = pd.DataFrame(X_t).reindex(r.index)
        # Drop rows with any NaN (avoid .dropna() which has a pandas/Python 3.13 bug)
        valid_mask = X_df.notna().all(axis=1) & r.notna()
        X_df = X_df[valid_mask]
        r = r[valid_mask]

        if len(r) < 50:
            continue

        X_rows.append(X_df)
        y_rows.append(r)

    if not X_rows:
        return {"selected_factors": [], "coefficients": {}}

    X_pooled = pd.concat(X_rows)
    y_pooled = pd.concat(y_rows)

    # Remove any remaining NaN/inf
    mask = np.isfinite(X_pooled).all(axis=1) & np.isfinite(y_pooled)
    X_pooled = X_pooled[mask]
    y_pooled = y_pooled[mask]

    # Subsample if too large to avoid memory issues
    max_obs = 100_000
    if len(y_pooled) > max_obs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_pooled), size=max_obs, replace=False)
        X_pooled = X_pooled.iloc[idx]
        y_pooled = y_pooled.iloc[idx]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pooled)

    # LASSO with cross-validation
    lasso = LassoCV(n_alphas=n_alphas, cv=n_folds, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y_pooled)

    coefficients = dict(zip(factor_names, lasso.coef_))
    selected = [f for f, c in coefficients.items() if abs(c) > 1e-8]

    return {
        "selected_factors": selected,
        "coefficients": coefficients,
        "alpha": lasso.alpha_,
        "r_squared": lasso.score(X_scaled, y_pooled),
        "n_observations": len(y_pooled),
    }
