import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DateConfig:
    start: str = "1970-01"
    end: str = "2019-12"
    validation_start: str = "1987-01"
    in_sample_end: str = "2014-12"
    out_of_sample_start: str = "2015-01"


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
    run_id: str = ""  # set to timestamp like "run_20260310_153600"


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

    def _run_base(self) -> Path:
        """Base output directory: outputs/run_TIMESTAMP/ or outputs/."""
        if self.output.run_id:
            return self.project_root / "outputs" / self.output.run_id
        return self.project_root / "outputs"

    def figures_path(self, stage: str) -> Path:
        path = self._run_base() / "figures" / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tables_path(self) -> Path:
        path = self._run_base() / "tables"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reports_path(self) -> Path:
        path = self._run_base() / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def find_prior_output(self, filename: str, subdir: str = "tables") -> Path | None:
        """Find a file from a prior stage, searching current run then recent runs.

        When stages are run individually (separate run folders), downstream stages
        need to find upstream outputs. Searches the current run first, then falls
        back to the most recent run folder containing the file.
        """
        # Check current run folder first
        current = self._run_base() / subdir / filename
        if current.exists():
            return current

        # Search recent run folders (newest first)
        outputs_dir = self.project_root / "outputs"
        if not outputs_dir.exists():
            return None
        for run_dir in sorted(outputs_dir.glob("run_*"), reverse=True):
            candidate = run_dir / subdir / filename
            if candidate.exists():
                return candidate

        return None


def load_config(config_path: Optional[str] = None, project_root: Optional[str] = None) -> PipelineConfig:
    if config_path is None:
        from datetime import datetime
        config = PipelineConfig()
        run_id = os.environ.get("PIPELINE_RUN_ID", "")
        if not run_id:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config.output.run_id = run_id
        return config

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

    # Pick up timestamped run ID from environment (set by run_all_stages.py),
    # or create a new timestamped run folder for this invocation
    from datetime import datetime
    run_id = os.environ.get("PIPELINE_RUN_ID", "")
    if not run_id:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.output.run_id = run_id

    return config
