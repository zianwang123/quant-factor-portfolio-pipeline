import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class EarningsYieldFactor(Factor):
    """Earnings Yield — Valuation.

    EY = EPSFXQ / PRCCM

    Inverse of P/E ratio. High earnings yield = cheap stock.
    More intuitive than P/E for cross-sectional sorts (avoids
    negative P/E issues).
    """

    name = "EarningsYield"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        epsfxq = panel.pivot("epsfxq")
        prccm = panel.pivot("prccm")

        ey = epsfxq / prccm
        return remove_infinities(ey)


class DividendYieldFactor(Factor):
    """Dividend Yield — Income/Valuation.

    DY = DVPSXQ / PRCCM

    High dividend yield stocks have historically outperformed,
    though the effect has weakened in recent decades.
    """

    name = "DivYield"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        dvpsxq = panel.pivot("dvpsxq")
        prccm = panel.pivot("prccm")

        dy = dvpsxq / prccm
        return remove_infinities(dy)


class ROEFactor(Factor):
    """Return on Equity — Profitability.

    ROE = IBQ / CEQQ

    Measures how efficiently a firm generates profits from shareholders' equity.
    High ROE firms are higher quality. Fama-French RMW factor is related.
    """

    name = "ROE"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        ibq = panel.pivot("ibq")
        ceqq = panel.pivot("ceqq")

        roe = ibq / ceqq
        return remove_infinities(roe)
