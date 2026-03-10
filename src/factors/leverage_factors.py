import numpy as np
import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.data.cleaner import remove_infinities


class LeverageFactor(Factor):
    """Financial Leverage.

    Leverage = (DLCQ + DLTTQ) / ATQ

    Total debt to total assets. High leverage firms are riskier.
    Bhandari (1988) showed leverage is priced in the cross-section.
    """

    name = "Leverage"
    benchmark_col = ""

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        dlcq = panel.pivot("dlcq")
        dlttq = panel.pivot("dlttq")
        atq = panel.pivot("atq")

        # fillna(0) for debt: if a firm has no short-term or long-term debt field,
        # it's reasonable to treat it as zero debt (not missing)
        leverage = (dlcq.fillna(0) + dlttq.fillna(0)) / atq
        return remove_infinities(leverage)
