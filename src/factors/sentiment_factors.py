import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class SUEFactor(Factor):
    """Standardized Unexpected Earnings (SUE).

    SUE = surpmean / surpstdev

    Measures the magnitude of earnings surprise relative to its
    historical dispersion. Post-earnings-announcement drift (PEAD)
    is one of the most well-documented anomalies — stocks with
    positive SUE continue to drift upward for ~60 days.
    """

    name = "SUE"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        surpmean = panel.pivot("surpmean")
        surpstdev = panel.pivot("surpstdev")
        sue = surpmean / surpstdev
        return remove_infinities(sue)


class EarningsRevisionFactor(Factor):
    """Analyst Earnings Revision Ratio.

    RevRatio = (NUMUP - NUMDOWN) / NUMEST

    Measures net direction of analyst estimate revisions.
    +1 = all analysts revising up, -1 = all revising down.
    Strong predictor of near-term returns (analyst herding + slow
    information incorporation).
    """

    name = "EarningsRevision"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        numup = panel.pivot("NUMUP")
        numdown = panel.pivot("NUMDOWN")
        numest = panel.pivot("NUMEST")
        rev_ratio = (numup - numdown) / numest
        return remove_infinities(rev_ratio)


class ShortTermReversalFactor(Factor):
    """1-Month Short-Term Reversal.

    STR = -trt1m(t)

    Jegadeesh (1990): Last month's losers outperform next month's winners.
    Negative sign because we want high values = expected high future return.
    Driven by liquidity provision and overreaction.
    """

    name = "STReversal"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        trt1m = panel.pivot("trt1m")
        return -trt1m  # Negate: last month losers → high signal → expected winners
