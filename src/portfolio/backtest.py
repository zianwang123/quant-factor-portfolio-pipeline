import numpy as np
import pandas as pd
from typing import Callable, Optional
from src.portfolio.covariance import ledoit_wolf_shrinkage, sample_covariance


class RollingBacktest:
    """Rolling-window portfolio backtest engine.

    Rebalances at specified frequency, tracks weights, computes out-of-sample returns,
    and accounts for transaction costs.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        optimizer_func: Callable,
        lookback_months: int = 60,
        rebalance_freq: str = "quarterly",
        transaction_cost_bps: float = 10.0,
        shrinkage: str = "ledoit_wolf",
    ):
        self.returns = returns
        self.optimizer_func = optimizer_func
        self.lookback = lookback_months
        self.rebalance_freq = rebalance_freq
        self.tc_bps = transaction_cost_bps / 10000.0
        self.shrinkage = shrinkage

    def _get_rebalance_dates(self) -> list:
        """Get rebalance dates based on frequency."""
        dates = self.returns.index.tolist()
        if self.rebalance_freq == "monthly":
            return dates[self.lookback:]
        elif self.rebalance_freq == "quarterly":
            return [d for d in dates[self.lookback:] if d.month in [3, 6, 9, 12]]
        elif self.rebalance_freq == "annual":
            return [d for d in dates[self.lookback:] if d.month == 12]
        return dates[self.lookback:]

    def _estimate_covariance(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        if self.shrinkage == "ledoit_wolf":
            return ledoit_wolf_shrinkage(returns_window)
        return sample_covariance(returns_window)

    def run(self, **optimizer_kwargs) -> dict:
        """Execute rolling backtest.

        Returns:
            dict with portfolio_returns, weights_history, turnover, costs.
        """
        rebalance_dates = self._get_rebalance_dates()
        if not rebalance_dates:
            return {}

        all_dates = self.returns.index.tolist()
        asset_names = self.returns.columns.tolist()
        n_assets = len(asset_names)

        portfolio_returns = pd.Series(dtype=float)
        weights_history = {}
        turnover_series = pd.Series(dtype=float)
        cost_series = pd.Series(dtype=float)

        current_weights = np.zeros(n_assets)
        last_rebalance_weights = np.zeros(n_assets)

        for date in all_dates:
            if date < all_dates[self.lookback]:
                continue

            # Check if this is a rebalance date
            if date in rebalance_dates:
                # Estimate parameters from lookback window
                idx = all_dates.index(date)
                window = self.returns.iloc[idx - self.lookback : idx].dropna(axis=1, how="any")

                if window.shape[1] < 3 or window.shape[0] < 20:
                    continue

                try:
                    cov = self._estimate_covariance(window)
                    mu = window.mean().values

                    # Run optimizer
                    new_weights = self.optimizer_func(
                        mu=mu,
                        sigma=cov.values,
                        **optimizer_kwargs,
                    )

                    # Turnover
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    turnover_series[date] = turnover

                    # Transaction cost
                    cost = turnover * self.tc_bps
                    cost_series[date] = cost

                    current_weights = new_weights
                    last_rebalance_weights = new_weights.copy()
                    weights_history[date] = pd.Series(new_weights, index=window.columns)

                except Exception as e:
                    print(f"  Optimization failed at {date}: {e}")
                    continue

            # Compute portfolio return
            if np.sum(np.abs(current_weights)) > 0:
                period_returns = self.returns.loc[date].reindex(asset_names).fillna(0).values
                port_ret = np.dot(current_weights, period_returns)

                # Deduct transaction cost on rebalance dates
                if date in cost_series.index:
                    port_ret -= cost_series[date]

                portfolio_returns[date] = port_ret

                # Update weights for drift
                drifted = current_weights * (1 + period_returns)
                total = drifted.sum()
                if total > 0:
                    current_weights = drifted / total

        return {
            "portfolio_returns": portfolio_returns,
            "weights_history": pd.DataFrame(weights_history).T,
            "turnover": turnover_series,
            "transaction_costs": cost_series,
        }
