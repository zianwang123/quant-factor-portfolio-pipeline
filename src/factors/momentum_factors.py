import pandas as pd
from src.factors.base import Factor
from src.data.loader import DataPanel


class MomentumFactor(Factor):
    """12-2 Month Long-Term Momentum.

    MOM = average of monthly returns from t-12 to t-2.

    Skip the most recent month (short-term reversal) and average
    the prior 11 months. Classic Jegadeesh & Titman (1993) signal.
    """

    name = "MOM"
    benchmark_col = "MOM"

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        trt1m = panel.pivot("trt1m")
        return trt1m.shift(2).rolling(window=11).mean()
