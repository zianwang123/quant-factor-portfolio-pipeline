import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm
from src.factors.base import Factor
from src.data.loader import DataPanel
from src.config import PipelineConfig


class CAPMBetaFactor(Factor):
    """CAPM Beta via Rolling OLS.

    Beta_i = slope from regressing stock excess returns on market excess returns
    using a 48-month rolling window.

    High-beta stocks are more sensitive to market movements.
    """

    name = "Beta"
    benchmark_col = "Beta"

    def __init__(self, config: PipelineConfig = None):
        self.window = config.factors.beta_window if config else 48

    def compute(self, panel: DataPanel) -> pd.DataFrame:
        trt1m = panel.pivot("trt1m")
        rf = panel.get_risk_free()
        mkt_excess = panel.get_market_excess()

        # Stock excess returns (trt1m is in percentage points, rf is in decimal)
        stock_excess = trt1m.sub(rf, axis=0) / 100.0
        stock_excess = stock_excess.dropna(axis=1, how="all")

        x = sm.add_constant(mkt_excess)
        betas = {}

        for col in tqdm(stock_excess.columns, desc="Computing CAPM Betas", leave=False):
            y = stock_excess[col]
            try:
                rols = RollingOLS(y, x, window=self.window, min_nobs=self.window)
                rres = rols.fit()
                betas[col] = rres.params["excess_return"]
            except Exception:
                continue

        beta_df = pd.DataFrame(betas)
        beta_df.index.name = "date"
        return beta_df
