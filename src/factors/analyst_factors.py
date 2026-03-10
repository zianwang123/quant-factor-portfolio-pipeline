import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel


class ExpectedLTGFactor(Factor):
    """Expected Long-Term Growth.

    LTGC = mean IBES consensus long-term (5-year) EPS growth estimate.

    Direct from analyst consensus. Positive LTG stocks are expected to
    grow earnings faster, commanding growth premium.
    """

    name = "LTGC"
    benchmark_col = "LTGC"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        return panel.pivot("LTG")
