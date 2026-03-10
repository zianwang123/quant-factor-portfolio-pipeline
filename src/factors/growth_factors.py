import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class SalesGrowthFactor(Factor):
    """Quarterly Sales Growth.

    SalesGrowth = SALEQ / lag(SALEQ, 4) - 1

    Year-over-year quarterly sales growth. Uses 4-quarter lag
    to avoid seasonality. High growth can be positive (momentum in
    fundamentals) or negative (overvaluation risk).
    """

    name = "SalesGrowth"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        saleq = panel.pivot("saleq")
        # 4-quarter lag for YoY comparison, but data is monthly
        # Use 12-month lag as approximation
        growth = saleq / saleq.shift(12) - 1
        return remove_infinities(growth)


class AssetGrowthFactor(Factor):
    """Asset Growth — Investment Factor.

    AssetGrowth = ATQ / lag(ATQ, 12) - 1

    Cooper, Gulen, Schill (2008): Firms that grow assets aggressively
    subsequently earn lower returns. One of the most robust anomalies.
    Related to Fama-French CMA (conservative minus aggressive) factor.
    """

    name = "AssetGrowth"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        atq = panel.pivot("atq")
        growth = atq / atq.shift(12) - 1
        return remove_infinities(growth)


class SustainableGrowthFactor(Factor):
    """Sustainable Growth Rate.

    SGR = ROE * (1 - Payout Ratio)
        = (IBQ / CEQQ) * (1 - DVPSXQ * CSHOQ / IBQ)

    Maximum growth rate achievable without external financing.
    Combines profitability with reinvestment discipline.
    """

    name = "SustGrowth"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        ceqq = panel.pivot("ceqq")
        dvpsxq = panel.pivot("dvpsxq")
        cshoq = panel.pivot("cshoq")

        roe = ibq / ceqq
        total_dividends = dvpsxq * cshoq
        # Payout ratio, capped at 1 to avoid negative retention
        payout = (total_dividends / ibq).clip(upper=1.0)
        payout = payout.where(ibq > 0, 0)  # If earnings negative, no meaningful payout ratio

        sgr = roe * (1 - payout)
        return remove_infinities(sgr)
