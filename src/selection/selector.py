import numpy as np
import pandas as pd
from src.selection.fama_macbeth import fama_macbeth_regression
from src.selection.information_coefficient import compute_ic_series, ic_summary
from src.selection.lasso_selection import lasso_factor_selection
from src.config import PipelineConfig


# Factor category mapping for diversification
FACTOR_CATEGORIES = {
    # Core
    "HL1M": "volatility",
    "LTGC": "analyst",
    "MOM": "momentum",
    "BP": "value",
    "Beta": "risk",
    "LogMktCap": "size",
    "AnnVol12M": "volatility",
    # Earnings Quality
    "AccrualRatio": "quality",
    "NetProfitMargin": "quality",
    "GrossProfit": "quality",
    "CFTP": "value",
    # Valuation
    "EarningsYield": "value",
    "DivYield": "value",
    "ROE": "quality",
    # Growth
    "SalesGrowth": "growth",
    "AssetGrowth": "growth",
    "SustGrowth": "growth",
    # Sentiment
    "SUE": "sentiment",
    "EarningsRevision": "sentiment",
    "STReversal": "momentum",
    # Leverage
    "Leverage": "leverage",
}


class FactorSelector:
    """Orchestrates multiple factor selection methods and produces a consensus ranking.

    Uses a greedy forward-selection algorithm that balances signal strength
    (consensus score) with pairwise decorrelation and category diversification.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run_all(
        self,
        returns: pd.DataFrame,
        factor_exposures: dict[str, pd.DataFrame],
        is_sp500: pd.DataFrame,
        qspreads: dict[str, pd.Series] | None = None,
    ) -> dict:
        """Run all selection methods and return combined results."""
        factor_names = list(factor_exposures.keys())

        # 1. Fama-MacBeth
        print("Running Fama-MacBeth regressions...")
        fm_results = fama_macbeth_regression(
            returns, factor_exposures, is_sp500,
            nw_lags=self.config.selection.fama_macbeth_nw_lags,
        )

        # 2. IC Analysis
        print("Computing Information Coefficients...")
        ic_results = {}
        for name in factor_names:
            ic_series = compute_ic_series(factor_exposures[name], returns, is_sp500)
            ic_results[name] = ic_summary(ic_series)
            ic_results[name]["ic_series"] = ic_series

        ic_table = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "ic_series"}
                                  for k, v in ic_results.items()}).T

        # 3. LASSO
        print("Running LASSO factor selection...")
        lasso_results = lasso_factor_selection(
            returns, factor_exposures, is_sp500,
            n_folds=self.config.selection.lasso_cv_folds,
            n_alphas=self.config.selection.lasso_alphas,
        )

        # 4. Consensus scoring
        scores = {}
        for name in factor_names:
            score = 0.0

            # FM: significant factor gets +1
            if name in fm_results.index and fm_results.loc[name, "Significant (5%)"]:
                score += 1.0

            # IC: above threshold gets +1, high IC IR gets +0.5
            if name in ic_table.index:
                if abs(ic_table.loc[name, "Mean IC"]) >= self.config.selection.min_ic:
                    score += 1.0
                if abs(ic_table.loc[name, "IC IR"]) >= self.config.selection.min_ic_ir:
                    score += 0.5

            # LASSO: selected gets +1
            if name in lasso_results.get("selected_factors", []):
                score += 1.0

            scores[name] = score

        consensus = pd.Series(scores).sort_values(ascending=False)

        # 5. Greedy forward selection for optimal combination
        selected_combo = self._greedy_select(
            consensus, qspreads, factor_names,
        )

        return {
            "fama_macbeth": fm_results,
            "ic_analysis": ic_table,
            "ic_series": {k: v["ic_series"] for k, v in ic_results.items()},
            "lasso": lasso_results,
            "consensus_scores": consensus,
            "selected_factors": selected_combo,
        }

    def _greedy_select(
        self,
        consensus: pd.Series,
        qspreads: dict[str, pd.Series] | None,
        all_factors: list[str],
    ) -> list[str]:
        """Greedy forward selection: pick factors that are strong AND decorrelated.

        Algorithm:
        1. Rank all factors by consensus score (descending).
        2. Start with the highest-scoring factor.
        3. For each candidate (in score order), add it only if:
           a) Its pairwise |correlation| with ALL already-selected factors < threshold.
           b) It adds category diversification (soft preference via bonus).
        4. Stop when we reach max_factors or run out of candidates above min score.
        """
        cfg = self.config.selection
        max_factors = cfg.max_factors
        min_factors = cfg.min_factors
        max_corr = cfg.max_pairwise_corr

        # Build QSpread correlation matrix if available
        corr_matrix = None
        if qspreads:
            valid_spreads = {k: v.dropna() for k, v in qspreads.items() if len(v.dropna()) > 10}
            if len(valid_spreads) > 1:
                spread_df = pd.DataFrame(valid_spreads)
                corr_matrix = spread_df.corr()

        # Candidates sorted by score
        candidates = consensus.sort_values(ascending=False)
        # Filter to factors with score > 0
        candidates = candidates[candidates > 0]

        selected = []
        selected_categories = set()

        for name, score in candidates.items():
            if len(selected) >= max_factors:
                break

            # Check pairwise correlation constraint
            if corr_matrix is not None and len(selected) > 0:
                too_correlated = False
                for existing in selected:
                    if name in corr_matrix.columns and existing in corr_matrix.columns:
                        pair_corr = abs(corr_matrix.loc[name, existing])
                        if pair_corr > max_corr:
                            too_correlated = True
                            break
                if too_correlated:
                    # If we haven't met minimum, relax constraint slightly
                    if len(selected) >= min_factors:
                        continue
                    # Below minimum: allow if corr < 0.8 (relaxed threshold)
                    if pair_corr > 0.8:
                        continue

            # Category diversification: prefer new categories
            cat = FACTOR_CATEGORIES.get(name, "other")
            if cat in selected_categories and len(selected) >= min_factors:
                # Already have this category and we've met minimum — skip unless score is very high
                if score < 2.0:
                    continue

            selected.append(name)
            selected_categories.add(cat)

        # If we still have fewer than min_factors, add top remaining by score
        if len(selected) < min_factors:
            for name, score in candidates.items():
                if name not in selected:
                    selected.append(name)
                    if len(selected) >= min_factors:
                        break

        print(f"\n  Greedy selection: {selected}")
        print(f"  Categories: {[FACTOR_CATEGORIES.get(f, 'other') for f in selected]}")
        return selected
