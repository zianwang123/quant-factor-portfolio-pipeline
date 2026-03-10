import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class AccrualRatioFactor(Factor):
    """Accrual Ratio — Earnings Quality.

    AccrualRatio = (IBQ - OANCFY) / ATQ

    High accruals (earnings far exceeding cash flow) predict lower future returns.
    Sloan (1996) accruals anomaly. Firms with cash-backed earnings outperform.
    """

    name = "AccrualRatio"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        oancfy = panel.pivot("oancfy")
        atq = panel.pivot("atq")

        accruals = (ibq - oancfy) / atq
        return remove_infinities(accruals)


class NetProfitMarginFactor(Factor):
    """Net Profit Margin — Profitability.

    NPM = IBQ / SALEQ

    Higher margin firms tend to be higher quality and outperform.
    """

    name = "NetProfitMargin"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        saleq = panel.pivot("saleq")

        npm = ibq / saleq
        return remove_infinities(npm)


class GrossProfitabilityFactor(Factor):
    """Gross Profitability — Quality Factor.

    GP = (SALEQ - COGSQ) / ATQ

    Novy-Marx (2013): Gross profitability is the "cleanest" accounting measure
    of economic profitability. Strongly predicts cross-section of returns.
    Negatively correlated with value — provides independent alpha.
    """

    name = "GrossProfit"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        saleq = panel.pivot("saleq")
        cogsq = panel.pivot("cogsq")
        atq = panel.pivot("atq")

        gp = (saleq - cogsq) / atq
        return remove_infinities(gp)


class CashFlowToPriceFactor(Factor):
    """Cash Flow to Price — Valuation.

    CFTP = OANCFY / (PRCCM * CSHOQ)

    Cash-flow-based value measure. More robust than earnings-based ratios
    because operating cash flow is harder to manipulate than net income.
    """

    name = "CFTP"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        oancfy = panel.pivot("oancfy")
        prccm = panel.pivot("prccm")
        cshoq = panel.pivot("cshoq")

        cftp = oancfy / (prccm * cshoq)
        return remove_infinities(cftp)
