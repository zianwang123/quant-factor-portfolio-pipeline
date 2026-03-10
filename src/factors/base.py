from abc import ABC, abstractmethod
import pandas as pd
from src.data.loader import DataPanel


class Factor(ABC):
    """Abstract base class for all equity factors."""

    name: str = ""
    benchmark_col: str = ""

    @abstractmethod
    def compute(self, panel: DataPanel) -> pd.DataFrame:
        """Compute factor exposures. Returns (date x gvkey) DataFrame."""
        ...

    def validate(self, computed: pd.DataFrame, benchmark: pd.DataFrame) -> float:
        """Compute correlation with Capital IQ benchmark."""
        if self.benchmark_col not in benchmark.columns:
            return float("nan")

        bench = benchmark[self.benchmark_col].dropna()
        comp = computed.reindex(bench.index).dropna()
        common = comp.index.intersection(bench.index)
        if len(common) < 10:
            return float("nan")
        return comp.loc[common].corr(bench.loc[common])
