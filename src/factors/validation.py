import numpy as np
import pandas as pd
from src.data.cleaner import remove_infinities


class QuintileSorter:
    """Engine for constructing long-short quintile portfolios from factor exposures.

    For each month:
    1. Filter to S&P 500 universe
    2. Sort stocks by factor value (descending)
    3. Form top 20% (winners) and bottom 20% (losers) equal-weighted portfolios
    4. QSpread = winner return(t+1) - loser return(t+1)

    Uses dicts for accumulation instead of Series.loc assignment to avoid
    pandas segfaults on Windows + Python 3.12.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def sort_single_factor(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        is_sp500: pd.DataFrame,
    ) -> dict:
        """Run quintile sort for a single factor."""
        factor_t = factor.T.dropna(axis=1, how="all")

        # Use plain dicts to accumulate values (avoids Series.loc segfault)
        qspread_d = {}
        long_d = {}
        short_d = {}
        quintile_d = {q: {} for q in range(1, self.n_bins + 1)}
        winner_turn_d = {}
        loser_turn_d = {}

        prev_winners = []
        prev_losers = []

        for t in factor_t.columns:
            cross_section = factor_t.loc[:, t].copy()

            # Filter to S&P 500 members
            if t in is_sp500.columns:
                sp500_mask = is_sp500.loc[:, t] == 1
                sp500_members = sp500_mask[sp500_mask].index
                cross_section = cross_section.reindex(sp500_members)

            # Clean: drop NaN and infinities
            cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna()

            if len(cross_section) < self.n_bins:
                continue

            # Sort descending
            cross_section = cross_section.sort_values(ascending=False)

            n = len(cross_section)
            quintile_size = n // self.n_bins

            next_t = t + 1
            if next_t not in returns.index:
                continue

            # Assign quintile groups
            for q in range(1, self.n_bins + 1):
                if q < self.n_bins:
                    q_stocks = cross_section.iloc[(q - 1) * quintile_size : q * quintile_size]
                else:
                    q_stocks = cross_section.iloc[(q - 1) * quintile_size :]

                q_ret = returns.loc[next_t].reindex(q_stocks.index).mean()
                quintile_d[q][next_t] = q_ret

            # Top and bottom 20%
            top = cross_section.iloc[: int(n * 0.2)]
            bottom = cross_section.iloc[int(n * 0.8) :]

            r_winner = returns.loc[next_t].reindex(top.index).mean()
            r_loser = returns.loc[next_t].reindex(bottom.index).mean()

            long_d[next_t] = r_winner
            short_d[next_t] = r_loser
            qspread_d[next_t] = r_winner - r_loser

            # Turnover
            if prev_winners:
                overlap_w = len(set(prev_winners) & set(top.index))
                winner_turn_d[next_t] = 1 - overlap_w / len(prev_winners)
            if prev_losers:
                overlap_l = len(set(prev_losers) & set(bottom.index))
                loser_turn_d[next_t] = 1 - overlap_l / len(prev_losers)

            prev_winners = list(top.index)
            prev_losers = list(bottom.index)

        # Convert dicts to Series at the end (single allocation, no segfault)
        return {
            "qspread": pd.Series(qspread_d, dtype=float),
            "long_return": pd.Series(long_d, dtype=float),
            "short_return": pd.Series(short_d, dtype=float),
            "turnover": pd.DataFrame({
                "Winner": pd.Series(winner_turn_d, dtype=float),
                "Loser": pd.Series(loser_turn_d, dtype=float),
            }),
            "quintile_returns": {q: pd.Series(v, dtype=float) for q, v in quintile_d.items()},
        }

    def sort_all_factors(
        self,
        factors: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        is_sp500: pd.DataFrame,
    ) -> dict[str, dict]:
        """Run quintile sort for all factors."""
        results = {}
        for name, factor_df in factors.items():
            print(f"  Sorting {name}...")
            results[name] = self.sort_single_factor(factor_df, returns, is_sp500)
        return results
