# Quant Factor Portfolio Pipeline — Project Context

## What This Project Is
- End-to-end quantitative investment pipeline for grad school applications (Princeton, Stanford, Baruch, MIT, CMU)
- Enhanced from 3 RSM6308 group projects (factor investment, portfolio optimization, Black-Litterman)
- GitHub repo: https://github.com/zianwang123/quant-factor-portfolio-pipeline (private)

## Tech Stack
- Python 3.12 (pinned via `.python-version`) — do NOT upgrade to 3.13 (segfaults)
- Package manager: `uv` (not pip, not conda)
- Build backend: `hatchling`
- Pinned versions: `numpy==2.0.2`, `pandas==2.2.3`, `scipy==1.14.1` — do NOT upgrade (newer versions segfault on Windows)
- Plotting: `plotly` (interactive HTML charts)

## Project Structure
```
src/                          # Python package
  data/                       # Data loading utilities
  factors/                    # Factor construction + validation (quintile sorts)
  selection/                  # Factor selection (FM, IC, LASSO, greedy forward)
  portfolio/                  # Portfolio optimization (MVO, Max Sharpe, Risk Parity, etc.)
  black_litterman/            # Black-Litterman model
  analytics/                  # Performance metrics
  visualization/              # Plotting utilities
scripts/
  run_all_stages.py           # Orchestrator — runs stages 1-3 + plots, with retries
  stage_1_factors.py          # Factor construction (21 factors, quintile sorts, benchmark validation)
  stage_2_selection.py        # Factor selection (FM, IC, LASSO, greedy forward)
  stage_3_optimize.py         # Portfolio optimization (5 variants) + benchmark comparison
  stage_4_black_litterman.py  # Black-Litterman (NOT yet tested end-to-end)
  plot_comparison.py          # Cumulative returns + drawdowns + Sharpe bar chart
  plot_individual_funds.py    # Individual mutual fund / hedge fund performance
config/
  pipeline.yaml               # Pipeline configuration
data/
  raw/                        # Raw data (compustat_crsp.csv 173MB, sp500_returns.csv, etc.)
  processed/                  # Pre-converted CSVs (from Excel), cached .pkl files
  processed/cache/            # Cached factor .pkl files from Stage 1
outputs/
  run_YYYYMMDD_HHMMSS/        # Timestamped run folders
    tables/                   # CSV results (oos_returns_all.csv, benchmark_comparison.csv, etc.)
    figures/                  # HTML/PNG charts organized by stage
    reports/                  # (future) LaTeX reports
```

## Pipeline Stages — Current Status

### Stage 1: Factor Construction — WORKING
- Builds 21 factors (7 core + 14 extended) from Compustat/CRSP data
- Runs quintile sorts (long-short Q1 vs Q5) for each factor
- Validates against Capital IQ benchmarks (correlations 0.91-0.998)
- Caches factors as .pkl files in `data/processed/cache/`
- Saves: factor_qspreads.csv, factor_benchmark_correlations.csv, factor_descriptive_stats.csv, factor_t_tests.csv

### Stage 2: Factor Selection — WORKING
- Loads cached factors from Stage 1 .pkl files (does NOT rebuild)
- Runs Fama-MacBeth (univariate), IC analysis, LASSO, greedy forward selection
- Selected 5 factors: STReversal (momentum), EarningsYield (value), CFTP (value), AccrualRatio (quality), AssetGrowth (growth)
- All pairwise QSpread correlations < 0.41

### Stage 3: Portfolio Optimization — WORKING
- Builds 5 portfolio variants from selected factors:
  - Equal Weight
  - IC-Weighted (best Sharpe: 1.049)
  - MVO (hedge fund style: min_weight=-2.0, max_weight=2.0, gross_leverage=3.0)
  - Max Sharpe (long_only=False, NO leverage cap — user explicitly requested this)
  - Risk Parity
- Compares against benchmarks: S&P 500, Hedge Fund Index EW, Mutual Fund EW, Smart Beta EW, Fama-French factors
- Benchmark data loaded from pre-converted CSVs in `data/processed/` (NOT from Excel — openpyxl segfaults)
- Saves: oos_returns_all.csv, benchmark_comparison.csv, factor_allocation_weights.csv, etc.

### Stage 4: Black-Litterman — NOT TESTED
- Code exists in `scripts/stage_4_black_litterman.py` and `src/black_litterman/`
- Has NOT been run end-to-end yet

## Latest Run Results (run_20260310_163209)

### OOS Sharpe Ratios
| Portfolio | Sharpe | Ann. Return | Max DD |
|-----------|--------|-------------|--------|
| IC-Weighted | 1.049 | 3.4% | -2.8% |
| Equal Weight | 0.964 | 2.8% | -3.2% |
| MVO | 0.885 | 6.6% | -10.3% |
| Risk Parity | 0.853 | 2.4% | -3.0% |
| Max Sharpe | 0.681 | 46.5% | -80.6% |
| S&P 500 | 1.299 | 14.4% | -13.6% |
| Hedge Fund Index (EW) | 1.032 | 3.6% | -6.2% |

### Key Observations
- Our portfolios are long-short (market-neutral), so low absolute returns are expected vs S&P 500
- IC-Weighted is the best risk-adjusted variant
- Max Sharpe overfits with unconstrained leverage (-80% drawdown)
- "Market + Factor Alpha" (S&P 500 + IC-Weighted) shows what a long-only investor gets by overlaying our signals

## Known Issues & Workarounds
1. **Windows Python segfaults**: Random access violations in pandas/numpy/scipy C extensions. NOT a code bug. Mitigated with:
   - Retry mechanism (5 attempts per stage on segfault exit codes 3221225477, 139, -11)
   - Separate subprocess per stage (avoids memory accumulation)
   - Numpy-based quintile sorting instead of pandas inner loops
   - Cached .pkl files between stages
   - `tqdm.tqdm.monitor_interval = 0` to avoid GIL contention
   - Pre-converted Excel to CSV to avoid openpyxl crashes
2. **Beta excluded from Stage 2**: RollingOLS segfaults; Beta has weak signal anyway (IC=-0.009)
3. **CSV loading**: Must use `low_memory=False` and auto-infer types (not `dtype=str`)
4. **Returns in raw data**: In percentage points — divide by 100 for decimal

## How to Run
```bash
# Install dependencies
uv sync

# Run full pipeline (stages 1-3 + plots)
uv run python scripts/run_all_stages.py

# Run a single stage
uv run python scripts/stage_1_factors.py
uv run python scripts/stage_2_selection.py
uv run python scripts/stage_3_optimize.py
```

## What's Left To Do

### 1. Stage 4: Black-Litterman — test end-to-end
- Code: `scripts/stage_4_black_litterman.py`, `src/black_litterman/model.py`, `equilibrium.py`, `sensitivity.py`
- Builds factor-based views (P, Q, Omega) from quintile sort results
- Computes BL posterior returns and optimal weights
- Runs OOS evaluation: BL weights vs market-cap weights
- Includes Diebold-Mariano test and tau/delta sensitivity grid
- Needs: integrate into `run_all_stages.py`, test for segfault resilience

### 2. Rolling Backtest with Transaction Costs — untested
- Code: `src/portfolio/backtest.py` (`RollingBacktest` class)
- Rolling-window rebalancing (monthly/quarterly/annual) with lookback
- Tracks weight drift between rebalances
- Accounts for transaction costs (configurable bps)
- Uses Ledoit-Wolf shrinkage for covariance estimation
- Needs: apply to our 5 portfolio variants, compare net-of-cost performance

### 3. Polished Visualization Suite — unused
- Code: `src/visualization/portfolio_plots.py`
- `plot_efficient_frontier()` — mean-variance frontier with GMV, Max Sharpe, Risk Parity points
- `plot_rolling_sharpe_comparison()` — rolling 36M Sharpe for multiple portfolios
- `plot_weight_evolution()` — stacked area chart of portfolio weights over time
- All save as interactive HTML + static PNG
- Needs: wire into Stage 3 or a dedicated plotting stage

### 4. Statistical Tests — partially used
- Code: `src/analytics/statistical_tests.py`
- `diebold_mariano_test()` — compares two strategies' forecast accuracy (used in Stage 4 only)
- `sharpe_ratio_test()` — Ledoit-Wolf test for equality of Sharpe ratios (Jobson-Korkie with Memmel correction)
- Needs: run Sharpe ratio tests comparing our portfolios vs S&P 500, Hedge Fund Index, etc.

### 5. Risk Analytics — partially used
- Code: `src/analytics/risk.py`
- `cornish_fisher_var()` — VaR adjusted for skewness and kurtosis (not used yet)
- `drawdown_stats()` — max DD, avg DD, max duration (partially used)
- Needs: add Cornish-Fisher VaR to benchmark comparison table

### 6. Documentation / LaTeX Report — not started
- To be done at the end
- Should cover methodology, results, and interpretation

## User Preferences
- Uses `uv` for Python package management
- Prefers industry project structure over Jupyter notebooks
- Wants documentation/LaTeX done at the end
- Do NOT add leverage cap to Max Sharpe — user explicitly rejected this
