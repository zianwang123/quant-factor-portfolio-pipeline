import gc
import numpy as np
import pandas as pd


class QuintileSorter:
    """Engine for constructing long-short quintile portfolios from factor exposures.

    For each month:
    1. Filter to S&P 500 universe
    2. Sort stocks by factor value (descending)
    3. Form top 20% (winners) and bottom 20% (losers) equal-weighted portfolios
    4. QSpread = winner return(t+1) - loser return(t+1)

    Uses numpy arrays in the inner loop to avoid pandas segfaults on
    Windows + Python 3.12.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def sort_single_factor(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        is_sp500: pd.DataFrame,
    ) -> dict:
        """Run quintile sort for a single factor using numpy internals."""
        # Pre-convert to numpy for speed and crash safety
        factor_vals = factor.values          # (dates, stocks)
        factor_dates = factor.index          # PeriodIndex
        stock_ids = factor.columns.values    # gvkey array

        returns_vals = returns.values
        returns_dates = returns.index
        ret_date_set = set(returns_dates)

        # Build fast date->row lookup for returns
        ret_date_to_row = {}
        for i, d in enumerate(returns_dates):
            ret_date_to_row[d] = i

        # Pre-convert is_sp500 to a dict of sets for fast lookup (pure numpy)
        sp500_vals = is_sp500.values  # (stocks, dates) or (dates, stocks)
        sp500_idx = is_sp500.index.values
        sp500_cols = is_sp500.columns.values
        sp500_sets = {}
        for j in range(len(sp500_cols)):
            d = sp500_cols[j]
            col_vals = sp500_vals[:, j]
            members = set()
            for k in range(len(sp500_idx)):
                if col_vals[k] == 1:
                    members.add(sp500_idx[k])
            sp500_sets[d] = members

        # Build stock_id -> column index in returns (first occurrence only)
        ret_stock_ids = returns.columns.values
        ret_stock_to_col = {}
        for i in range(len(ret_stock_ids)):
            sid = ret_stock_ids[i]
            if sid not in ret_stock_to_col:
                ret_stock_to_col[sid] = i

        # Accumulators
        qspread_d = {}
        long_d = {}
        short_d = {}
        quintile_d = {q: {} for q in range(1, self.n_bins + 1)}
        winner_turn_d = {}
        loser_turn_d = {}
        prev_winners = []
        prev_losers = []

        for row_i, t in enumerate(factor_dates):
            # Get factor values for this date
            fvals = factor_vals[row_i]

            # Filter to S&P 500 members
            if t in sp500_sets:
                sp_members = sp500_sets[t]
            else:
                continue

            # Build arrays of (stock_id, factor_value) for valid stocks
            valid_ids = []
            valid_fv = []
            for col_j in range(len(stock_ids)):
                sid = stock_ids[col_j]
                if sid not in sp_members:
                    continue
                v = fvals[col_j]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isnan(v) or np.isinf(v):
                    continue
                valid_ids.append(sid)
                valid_fv.append(v)

            if len(valid_ids) < self.n_bins:
                continue

            next_t = t + 1
            if next_t not in ret_date_set:
                continue
            ret_row = ret_date_to_row[next_t]

            # Sort descending by factor value
            valid_fv_arr = np.array(valid_fv, dtype=float)
            sort_idx = np.argsort(-valid_fv_arr)
            sorted_ids = [valid_ids[int(i)] for i in sort_idx]

            n = len(sorted_ids)
            quintile_size = n // self.n_bins

            # Quintile returns
            for q in range(1, self.n_bins + 1):
                start = (q - 1) * quintile_size
                end = q * quintile_size if q < self.n_bins else n
                q_rets = []
                for sid in sorted_ids[start:end]:
                    if sid in ret_stock_to_col:
                        r = float(returns_vals[ret_row, ret_stock_to_col[sid]])
                        if not np.isnan(r):
                            q_rets.append(r)
                quintile_d[q][next_t] = np.mean(q_rets) if q_rets else np.nan

            # Top 20% and bottom 20%
            top_n = max(1, int(n * 0.2))
            bot_start = int(n * 0.8)

            top_ids = list(sorted_ids[:top_n])
            bot_ids = list(sorted_ids[bot_start:])

            top_rets = []
            for sid in top_ids:
                if sid in ret_stock_to_col:
                    r = float(returns_vals[ret_row, ret_stock_to_col[sid]])
                    if not np.isnan(r):
                        top_rets.append(r)

            bot_rets = []
            for sid in bot_ids:
                if sid in ret_stock_to_col:
                    r = float(returns_vals[ret_row, ret_stock_to_col[sid]])
                    if not np.isnan(r):
                        bot_rets.append(r)

            r_winner = np.mean(top_rets) if top_rets else np.nan
            r_loser = np.mean(bot_rets) if bot_rets else np.nan

            long_d[next_t] = r_winner
            short_d[next_t] = r_loser
            qspread_d[next_t] = r_winner - r_loser

            # Turnover
            if prev_winners:
                overlap_w = len(set(prev_winners) & set(top_ids))
                winner_turn_d[next_t] = 1 - overlap_w / len(prev_winners)
            if prev_losers:
                overlap_l = len(set(prev_losers) & set(bot_ids))
                loser_turn_d[next_t] = 1 - overlap_l / len(prev_losers)

            prev_winners = top_ids
            prev_losers = bot_ids

        # Convert dicts to Series at the end
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
            print(f"  Sorting {name}...", flush=True)
            results[name] = self.sort_single_factor(factor_df, returns, is_sp500)
            gc.collect()
        return results
