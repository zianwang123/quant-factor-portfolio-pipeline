import gc
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fama_macbeth_regression(
    returns: pd.DataFrame,
    factor_exposures: dict[str, pd.DataFrame],
    is_sp500: pd.DataFrame,
    nw_lags: int = 6,
    end_date: str = None,
) -> pd.DataFrame:
    """Two-pass Fama-MacBeth cross-sectional regression (univariate per factor).

    For each factor independently:
      Pass 1: For each month t, run cross-sectional regression:
          R_{i,t+1} = gamma_0,t + gamma_1,t * F_{i,t} + epsilon_{i,t}
      Pass 2: Time-series average of gamma_1,t with Newey-West standard errors.

    Running univariate regressions avoids multicollinearity and memory issues
    when dealing with 20+ factors.

    Args:
        end_date: If provided, only use factor dates up to this period (inclusive)
                  to avoid look-ahead bias. The last return used is end_date itself
                  (predicted by factor at end_date - 1 month).
    """
    results = {}

    for fname, f_df in factor_exposures.items():
        gamma_series = []
        common_dates = returns.index[:-1].intersection(f_df.index)
        if end_date is not None:
            common_dates = common_dates[common_dates <= pd.Period(end_date, "M")]

        for t in common_dates:
            t_plus_1 = t + 1
            if t_plus_1 not in returns.index:
                continue

            if t not in is_sp500.columns:
                continue

            sp500_members = is_sp500[is_sp500[t] == 1].index
            y = returns.loc[t_plus_1].reindex(sp500_members).dropna()

            if t not in f_df.index:
                continue

            x = f_df.loc[t].reindex(y.index).dropna()
            y = y.reindex(x.index).dropna()
            x = x.reindex(y.index)

            if len(y) < 20:
                continue

            X_const = sm.add_constant(x)
            try:
                model = sm.OLS(y.values, X_const.values).fit()
                gamma_series.append(model.params[1])  # slope coefficient
            except Exception:
                continue

        if len(gamma_series) < 10:
            continue

        # Pass 2: test average risk premium with Newey-West SE
        y_ts = np.array(gamma_series)
        X_ts = np.ones((len(y_ts), 1))

        try:
            ols = sm.OLS(y_ts, X_ts).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        except Exception:
            continue

        results[fname] = {
            "Risk Premium": ols.params[0],
            "t-stat (NW)": ols.tvalues[0],
            "p-value": ols.pvalues[0],
            "Significant (5%)": ols.pvalues[0] < 0.05,
            "N Months": len(gamma_series),
        }

        gc.collect()

    return pd.DataFrame(results).T
