import math
import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class HL1MFactor(Factor):
    """1-Month Price High-Low Ratio.

    HL1M = (HighM - CloseM) / (CloseM - LowM)

    Captures intra-month price dynamics. High values indicate the stock
    closed near its monthly low (bearish), low values near its high (bullish).
    """

    name = "HL1M"
    benchmark_col = "HL1M"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        prccm = panel.pivot("prccm")
        prchm = panel.pivot("prchm")
        prclm = panel.pivot("prclm")

        hl1m = (prchm - prccm) / (prccm - prclm)
        return remove_infinities(hl1m)


class AnnVol12MFactor(Factor):
    """12-Month Annualized Realized Volatility.

    AnnVol12M = sqrt(12) * sqrt( (1/11) * sum_{n=0}^{12} r_{t-n}^2 )

    where r_t = ln(P_t / P_{t-1}).

    Uses 13-month window of squared log returns, divided by 11 (degrees of freedom),
    then annualized by sqrt(12).
    """

    name = "AnnVol12M"
    benchmark_col = "AnnVol12M"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        prccm = panel.pivot("prccm")
        log_ret = np.log(prccm / prccm.shift(1))
        r_sq = log_ret ** 2

        # Avoid rolling.apply() which segfaults on Python 3.13/Windows
        # AnnVol = sqrt(12) * sqrt( sum(r^2) / 11 )  over 13-month window
        rolling_sum = r_sq.rolling(window=13, min_periods=13).sum()
        return np.sqrt(rolling_sum / 11) * np.sqrt(12)
