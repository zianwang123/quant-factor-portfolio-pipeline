from src.factors.base import Factor
from src.factors.price_factors import HL1MFactor, AnnVol12MFactor
from src.factors.fundamental_factors import BookToPriceFactor, LogMarketCapFactor
from src.factors.momentum_factors import MomentumFactor
from src.factors.analyst_factors import ExpectedLTGFactor
from src.factors.risk_factors import CAPMBetaFactor
from src.factors.earnings_quality_factors import (
    AccrualRatioFactor, NetProfitMarginFactor, GrossProfitabilityFactor, CashFlowToPriceFactor,
)
from src.factors.valuation_factors import EarningsYieldFactor, DividendYieldFactor, ROEFactor
from src.factors.growth_factors import SalesGrowthFactor, AssetGrowthFactor, SustainableGrowthFactor
from src.factors.sentiment_factors import SUEFactor, EarningsRevisionFactor, ShortTermReversalFactor
from src.factors.leverage_factors import LeverageFactor
from src.data.loader import DataPanel
from src.config import PipelineConfig
import pandas as pd
from tqdm import tqdm


# Original 7 factors from RSM6308 homework (validated against Capital IQ)
CORE_FACTORS: dict[str, type[Factor]] = {
    "HL1M": HL1MFactor,
    "LTGC": ExpectedLTGFactor,
    "MOM": MomentumFactor,
    "BP": BookToPriceFactor,
    "Beta": CAPMBetaFactor,
    "LogMktCap": LogMarketCapFactor,
    "AnnVol12M": AnnVol12MFactor,
}

# Extended factor zoo — built from full Compustat/IBES dataset
EXTENDED_FACTORS: dict[str, type[Factor]] = {
    # Earnings Quality
    "AccrualRatio": AccrualRatioFactor,
    "NetProfitMargin": NetProfitMarginFactor,
    "GrossProfit": GrossProfitabilityFactor,
    "CFTP": CashFlowToPriceFactor,
    # Valuation
    "EarningsYield": EarningsYieldFactor,
    "DivYield": DividendYieldFactor,
    "ROE": ROEFactor,
    # Growth / Investment
    "SalesGrowth": SalesGrowthFactor,
    "AssetGrowth": AssetGrowthFactor,
    "SustGrowth": SustainableGrowthFactor,
    # Analyst Sentiment
    "SUE": SUEFactor,
    "EarningsRevision": EarningsRevisionFactor,
    # Short-Term
    "STReversal": ShortTermReversalFactor,
    # Leverage
    "Leverage": LeverageFactor,
}

# Full registry: core + extended
FACTOR_REGISTRY: dict[str, type[Factor]] = {**CORE_FACTORS, **EXTENDED_FACTORS}


def build_all_factors(
    panel: DataPanel,
    config: PipelineConfig,
    include_extended: bool = True,
    exclude: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute all registered factors and return as dict of DataFrames.

    Args:
        panel: DataPanel with loaded data.
        config: Pipeline configuration.
        include_extended: If True, also build the extended factor zoo.
        exclude: List of factor names to skip (e.g., ["Beta"] to avoid RollingOLS crash).
    """
    registry = FACTOR_REGISTRY if include_extended else CORE_FACTORS
    if exclude:
        registry = {k: v for k, v in registry.items() if k not in exclude}
    factors = {}

    for name, factor_cls in tqdm(registry.items(), desc="Building factors"):
        factor = factor_cls(config) if name == "Beta" else factor_cls()

        try:
            factors[name] = factor.compute(panel)
            non_null = factors[name].count().sum()
            print(f"  {name}: shape={factors[name].shape}, non-null={non_null}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    return factors


def validate_all_factors(
    factors: dict[str, pd.DataFrame],
    qspreads: dict[str, pd.Series],
    capital_iq: pd.DataFrame,
    validation_start: str = "1987-01",
    validation_end: str = "2019-12",
) -> pd.DataFrame:
    """Validate factor QSpreads against Capital IQ benchmarks.

    Returns a DataFrame of correlations. Only applies to core factors
    that have Capital IQ benchmarks.
    """
    results = {}

    for name, qspread in qspreads.items():
        if name not in capital_iq.columns:
            continue

        bench = capital_iq[name].loc[validation_start:validation_end].dropna()
        comp = qspread.loc[validation_start:validation_end].dropna()
        common = comp.index.intersection(bench.index)

        if len(common) > 10:
            results[name] = abs(comp.loc[common].corr(bench.loc[common]))
        else:
            results[name] = float("nan")

    return pd.DataFrame.from_dict(results, orient="index", columns=["Correlation"])
