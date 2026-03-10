import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class BookToPriceFactor(Factor):
    """Book-to-Price Ratio.

    BP = CEQQ / (CSHOQ * CloseM)

    Classic value factor. High BP stocks are "cheap" relative to book value.
    """

    name = "BP"
    benchmark_col = "BP"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        ceqq = panel.pivot("ceqq")
        cshoq = panel.pivot("cshoq")
        prccm = panel.pivot("prccm")

        bp = ceqq / (cshoq * prccm)
        return remove_infinities(bp)


class LogMarketCapFactor(Factor):
    """Log Market Capitalization.

    LogMktCap = log(CloseM * CSHOM)

    Size factor. Negative QSpread expected (small caps outperform large caps).
    """

    name = "LogMktCap"
    benchmark_col = "LogMktCap"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        prccm = panel.pivot("prccm")
        cshom = panel.pivot("cshom")

        mktcap = prccm * cshom
        # Replace zero/negative with NaN before log
        mktcap = mktcap.where(mktcap > 0)
        return np.log(mktcap)
