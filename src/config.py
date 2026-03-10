import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DateConfig:
    start: str = "1970-01"
    end: str = "2019-12"
    validation_start: str = "1987-01"
    in_sample_end: str = "2012-12"
    out_of_sample_start: str = "2013-01"


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    compustat_file: str = "compustat_crsp.csv"
    sp500_file: str = "sp500_returns.csv"
    capital_iq_file: str = "capital_iq_benchmark.csv"
    fama_french_file: str = "fama_french_factors.xlsx"


@dataclass
class FactorConfig:
    beta_window: int = 48
    momentum_skip: int = 2
    momentum_window: int = 11
    vol_window: int = 13
    quintile_bins: int = 5
    winsorize_pct: float = 0.01


@dataclass
class SelectionConfig:
    fama_macbeth_nw_lags: int = 6
    min_ic: float = 0.02
    min_ic_ir: float = 0.5
    lasso_cv_folds: int = 5
    lasso_alphas: int = 100
    max_factors: int = 5
    min_factors: int = 3
    max_pairwise_corr: float = 0.6


@dataclass
class OptimizationConfig:
    risk_aversion: float = 10.0
    max_leverage: float = 1.5
    rebalance_freq: str = "quarterly"
    transaction_cost_bps: float = 10.0
    shrinkage: str = "ledoit_wolf"


@dataclass
class BlackLittermanConfig:
    tau: float = 0.5
    delta: float = 10.0
    view_confidence: str = "ic_based"


@dataclass
class OutputConfig:
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    reports_dir: str = "outputs/reports"


@dataclass
class PipelineConfig:
    dates: DateConfig = field(default_factory=DateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    black_litterman: BlackLittermanConfig = field(default_factory=BlackLittermanConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    project_root: Path = field(default_factory=lambda: Path("."))

    def raw_path(self, filename: str) -> Path:
        return self.project_root / self.data.raw_dir / filename

    def processed_path(self, filename: str) -> Path:
        path = self.project_root / self.data.processed_dir
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def figures_path(self, stage: str) -> Path:
        path = self.project_root / self.output.figures_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tables_path(self) -> Path:
        path = self.project_root / self.output.tables_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(config_path: Optional[str] = None, project_root: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        return PipelineConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = PipelineConfig(
        dates=DateConfig(**raw.get("dates", {})),
        data=DataConfig(**raw.get("data", {})),
        factors=FactorConfig(**raw.get("factors", {})),
        selection=SelectionConfig(
            fama_macbeth_nw_lags=raw.get("selection", {}).get("fama_macbeth", {}).get("newey_west_lags", 6),
            min_ic=raw.get("selection", {}).get("ic", {}).get("min_ic", 0.02),
            min_ic_ir=raw.get("selection", {}).get("ic", {}).get("min_ic_ir", 0.5),
            lasso_cv_folds=raw.get("selection", {}).get("lasso", {}).get("cv_folds", 5),
            lasso_alphas=raw.get("selection", {}).get("lasso", {}).get("alphas", 100),
        ),
        optimization=OptimizationConfig(**raw.get("optimization", {})),
        black_litterman=BlackLittermanConfig(**raw.get("black_litterman", {})),
        output=OutputConfig(**raw.get("output", {})),
    )

    if project_root:
        config.project_root = Path(project_root)

    return config
