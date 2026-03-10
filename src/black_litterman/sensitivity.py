import numpy as np
import pandas as pd
from src.black_litterman.model import black_litterman_posterior


def tau_delta_grid(
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau_values: list[float] = None,
    delta_values: list[float] = None,
    stock_names: list = None,
) -> dict:
    """Run BL model across a grid of tau and delta values.

    Returns dict of (tau, delta) -> BL results.
    """
    if tau_values is None:
        tau_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    if delta_values is None:
        delta_values = [1, 2.5, 5, 10, 25, 50]

    results = {}
    weights_table = {}
    returns_table = {}

    for tau in tau_values:
        for delta in delta_values:
            try:
                bl = black_litterman_posterior(delta, sigma, w_mkt, tau, P, Q, Omega)
                key = f"tau={tau}, delta={delta}"
                results[key] = bl

                if stock_names:
                    weights_table[key] = pd.Series(bl["weights"], index=stock_names)
                    returns_table[key] = pd.Series(bl["mu_posterior"], index=stock_names)
            except Exception as e:
                print(f"  Failed for tau={tau}, delta={delta}: {e}")

    return {
        "results": results,
        "weights": pd.DataFrame(weights_table) if weights_table else pd.DataFrame(),
        "returns": pd.DataFrame(returns_table) if returns_table else pd.DataFrame(),
    }


def view_impact_analysis(
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    tau: float,
    delta: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    stock_names: list = None,
    n_top: int = 5,
) -> dict:
    """Analyze the impact of views on portfolio weights and returns.

    Returns:
        dict with bullish, bearish, and neutral stock tables.
    """
    bl = black_litterman_posterior(delta, sigma, w_mkt, tau, P, Q, Omega)

    weight_change = bl["weights"] - w_mkt
    return_change = bl["mu_posterior"] - bl["pi"]

    if stock_names is None:
        stock_names = list(range(len(w_mkt)))

    # Sort by weight change
    wc = pd.Series(weight_change, index=stock_names).sort_values(ascending=False)

    # P vector for each stock
    if P.ndim == 1:
        p_series = pd.Series(P, index=stock_names)
    else:
        p_series = pd.Series(P.sum(axis=0), index=stock_names)

    bullish = wc.iloc[:n_top].index.tolist()
    bearish = wc.iloc[-n_top:].index.tolist()
    mid = len(wc) // 2
    neutral = wc.iloc[mid - n_top // 2 : mid + n_top // 2 + 1].index.tolist()

    def _make_table(stocks):
        return pd.DataFrame({
            "Prior Weight": pd.Series(w_mkt, index=stock_names).loc[stocks],
            "View (P)": p_series.loc[stocks],
            "Posterior Weight": pd.Series(bl["weights"], index=stock_names).loc[stocks],
            "Weight Change": pd.Series(weight_change, index=stock_names).loc[stocks],
            "Prior Return": pd.Series(bl["pi"], index=stock_names).loc[stocks],
            "Posterior Return": pd.Series(bl["mu_posterior"], index=stock_names).loc[stocks],
        })

    return {
        "bullish": _make_table(bullish),
        "bearish": _make_table(bearish),
        "neutral": _make_table(neutral),
        "bl_result": bl,
        "weight_change": wc,
    }
