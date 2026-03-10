import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PipelineConfig


def load_compustat(config: PipelineConfig) -> pd.DataFrame:
    """Load Compustat/CRSP monthly panel data."""
    filepath = config.raw_path(config.data.compustat_file)

    # Only force string for identifier columns; let pandas auto-infer numeric types
    # This avoids pd.to_numeric segfaults on large datasets (Windows/pandas issue)
    str_cols = {"gvkey", "iid", "tic", "cusip", "TICKER", "lpermno"}
    dtype_map = {col: str for col in str_cols}
    df = pd.read_csv(filepath, dtype=dtype_map, low_memory=False)

    # Parse date to monthly period
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.to_period("M")

    df["sp500"] = df["sp500"].fillna(0).astype(int)
    df["gvkey"] = df["gvkey"].astype(str).str.strip()

    return df


def load_sp500_returns(config: PipelineConfig) -> pd.DataFrame:
    """Load S&P 500 index returns and risk-free rate."""
    filepath = config.raw_path(config.data.sp500_file)
    df = pd.read_csv(filepath, dtype=str)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m").dt.to_period("M")
    df = df.set_index("Date")
    df["ret_sp500"] = df["ret_sp500"].astype(float)
    df["rf"] = df["rf"].astype(float)
    df["excess_return"] = df["ret_sp500"] - df["rf"]
    return df


def load_capital_iq(config: PipelineConfig) -> pd.DataFrame:
    """Load Capital IQ benchmark factor returns."""
    filepath = config.raw_path(config.data.capital_iq_file)
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d").dt.to_period("M")
    df = df.set_index("Date")
    return df


def load_fama_french(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load Fama-French factors and fund return data from Excel."""
    filepath = config.raw_path(config.data.fama_french_file)
    sheets = pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
    return sheets


class DataPanel:
    """Central data panel providing pivoted views of the Compustat data."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._raw = None
        self._sp500 = None
        self._capital_iq = None

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            self._raw = load_compustat(self.config)
        return self._raw

    @property
    def sp500(self) -> pd.DataFrame:
        if self._sp500 is None:
            self._sp500 = load_sp500_returns(self.config)
        return self._sp500

    @property
    def capital_iq(self) -> pd.DataFrame:
        if self._capital_iq is None:
            self._capital_iq = load_capital_iq(self.config)
        return self._capital_iq

    # Quarterly Compustat fields — reported once per quarter, need forward-fill
    # for monthly factor construction (carry last known value forward up to 3 months)
    QUARTERLY_FIELDS = {
        "epsfxq", "ceqq", "cshoq", "atq", "ltq", "dlcq", "dlttq",
        "saleq", "cogsq", "ibq", "dvpsxq", "oancfy", "cheq", "rectq",
        "invtq", "acoq", "lcoq", "txditcq",
    }

    def pivot(self, field: str) -> pd.DataFrame:
        """Pivot raw data into (date x gvkey) matrix for a given field.

        Quarterly fields are forward-filled up to 3 months so that stocks
        aren't dropped from monthly quintile sorts between reporting dates.
        This uses only past information (no look-ahead bias).
        """
        start, end = self.config.dates.start, self.config.dates.end
        pivoted = self.raw.pivot(index="date", columns="gvkey", values=field)
        pivoted = pivoted.loc[start:end]

        if field in self.QUARTERLY_FIELDS:
            pivoted = pivoted.ffill(limit=3)

        return pivoted

    def get_returns(self, extend_end: str = "2020-01") -> pd.DataFrame:
        """Get total returns matrix. Extends one month past end for t+1 lookups."""
        pivoted = self.raw.pivot(index="date", columns="gvkey", values="trt1m")
        return pivoted.loc[self.config.dates.start:extend_end] / 100.0

    def get_sp500_membership(self) -> pd.DataFrame:
        """Get S&P 500 membership flags as (gvkey x date) boolean matrix."""
        start, end = self.config.dates.start, self.config.dates.end
        is_sp500 = self.raw.pivot(index="date", columns="gvkey", values="sp500")
        return is_sp500.loc[start:end].T

    def get_risk_free(self) -> pd.Series:
        """Get monthly risk-free rate series."""
        start, end = self.config.dates.start, self.config.dates.end
        return self.sp500["rf"].loc[start:end]

    def get_market_excess(self) -> pd.Series:
        """Get monthly market excess return series."""
        start, end = self.config.dates.start, self.config.dates.end
        return self.sp500["excess_return"].loc[start:end]
