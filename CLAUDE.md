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
  stage_1_factors.py          # Factor construction (20 factors, quintile sorts, benchmark validation)
  stage_2_selection.py        # Factor selection (FM, IC, LASSO, greedy forward)
  stage_3_optimize.py         # Portfolio optimization (5 variants) + benchmark comparison
  stage_4_black_litterman.py  # Black-Litterman (NOT yet tested end-to-end)
  plot_comparison.py          # Cumulative returns + drawdowns + Sharpe bar chart
  plot_individual_funds.py    # Individual mutual fund / hedge fund performance
  plot_factors.py             # Per-factor plots (QSpread, long/short legs, quintiles, IC, correlation heatmap)
config/
  pipeline.yaml               # Pipeline configuration
data/
  raw/                        # Raw data (compustat_crsp.csv 173MB, sp500_returns.csv, etc.)
  processed/                  # Pre-converted CSVs (from Excel), cached .pkl files
  processed/cache/            # Cached factor .pkl, sort_results.pkl, ic_series.pkl
outputs/
  run_YYYYMMDD_HHMMSS/        # Timestamped run folders
    tables/                   # CSV results (oos_returns_all.csv, benchmark_comparison.csv, etc.)
    figures/                  # HTML/PNG charts organized by stage
    reports/                  # (future) LaTeX reports
```

## Pipeline Stages — Current Status

### Stage 1: Factor Construction — WORKING
- Builds 20 factors (7 core + 13 extended, Beta excluded) from Compustat/CRSP data
- Runs quintile sorts (long-short Q1 vs Q5) for each factor
- Validates against Capital IQ benchmarks (correlations 0.87-0.99)
- Caches factors as .pkl files in `data/processed/cache/`
- Caches `sort_results.pkl` (for factor plots) and `returns.pkl` / `is_sp500.pkl`
- Saves: factor_qspreads.csv, factor_benchmark_correlations.csv, factor_descriptive_stats.csv, factor_t_tests.csv

### Stage 2: Factor Selection — WORKING
- Loads cached factors from Stage 1 .pkl files (does NOT rebuild)
- Runs Fama-MacBeth (univariate), IC analysis, LASSO, greedy forward selection
- Caches `ic_series.pkl` for factor plots
- Selected 5 factors: AccrualRatio (quality), CFTP (value), STReversal (momentum), AssetGrowth (growth), ROE (quality)
- All pairwise QSpread correlations < 0.56

### Stage 3: Portfolio Optimization — WORKING
- Builds 5 portfolio variants from selected factors:
  - Equal Weight
  - IC-Weighted (best Sharpe: 1.311)
  - MVO (hedge fund style: min_weight=-2.0, max_weight=2.0, gross_leverage=3.0)
  - Max Sharpe (long_only=False, gross_leverage=3.0)
  - Risk Parity
- Compares against benchmarks: S&P 500, Hedge Fund Index EW/Best, Mutual Fund EW/Best, Smart Beta EW/Best, Fama-French factors
- Benchmark data loaded from pre-converted CSVs in `data/processed/` (converted from fama_french_factors.xlsx tabs)
- FSPHX data anomaly (split/NAV errors) auto-fixed via `_fix_anomalies()` in both stage_3 and plot scripts
- All Sharpe ratios use excess returns (rf subtracted for long-only benchmarks)
- Saves: oos_returns_all.csv, benchmark_comparison.csv, factor_allocation_weights.csv, etc.

### Plotting Stages — WORKING
- `plot_comparison.py`: Cumulative returns (all "Our:" + EW/Best benchmarks), drawdowns, Sharpe bar chart
- `plot_individual_funds.py`: Per-fund cumulative vs our portfolio, with anomaly correction and excess-return Sharpe
- `plot_factors.py`: Per-factor plots (QSpread vs Capital IQ, long/short legs, quintile monotonicity, cumulative vs market, IC time series, correlation heatmap)
- All integrated into `run_all_stages.py` as non-blocking plot stages (warn on failure, don't abort)

### Stage 4: Black-Litterman — NOT TESTED
- Code exists in `scripts/stage_4_black_litterman.py` and `src/black_litterman/`
- Has NOT been run end-to-end yet

## Latest Run Results (run_20260310_221037)

### OOS Sharpe Ratios (IS: 1987-2014, OOS: 2015-2019)
| Portfolio | Sharpe | Ann. Return | Ann. Vol | Max DD |
|-----------|--------|-------------|----------|--------|
| HFRIMAI (Best HF) | 1.358 | 3.2% | 2.4% | -2.4% |
| Our: IC-Weighted | 1.311 | 4.3% | 3.3% | -1.9% |
| Our: Equal Weight | 1.201 | 3.7% | 3.1% | -3.2% |
| Our: MVO | 1.164 | 8.5% | 7.3% | -7.5% |
| Our: Max Sharpe | 1.082 | 10.0% | 9.3% | -10.4% |
| Our: Risk Parity | 1.005 | 3.0% | 3.0% | -3.0% |
| Mutual Fund (EW) | 0.946 | 16.4% | 17.4% | -21.1% |
| FSMEX (Best MF) | 0.919 | 19.0% | 20.7% | -17.1% |
| S&P 500 | 0.907 | 10.8% | 12.0% | -14.1% |
| FF Market (Mkt-RF) | 0.857 | 10.7% | 12.4% | -15.1% |
| Smart Beta Best (PSL) | 0.818 | 8.0% | 9.8% | -10.6% |
| Smart Beta (EW) | 0.744 | 8.9% | 11.9% | -16.4% |
| Hedge Fund Index (EW) | 0.562 | 2.0% | 3.6% | -7.8% |

### Key Observations
- Our portfolios are long-short (market-neutral), so low absolute returns are expected vs S&P 500
- IC-Weighted is the best risk-adjusted variant — beats S&P 500 on Sharpe (1.311 vs 0.907)
- All 5 variants have Sharpe > 1.0 with the extended training set (IS through 2014)
- Max Sharpe constrained to 3x gross leverage — well-behaved weights and drawdown
- "Market + Factor Alpha" (S&P 500 + IC-Weighted) shows what a long-only investor gets by overlaying our signals

## Known Issues & Workarounds
1. **Windows Python segfaults**: Random access violations in pandas/numpy/scipy C extensions. NOT a code bug. Mitigated with:
   - Retry mechanism (5 attempts per stage on segfault exit codes 3221225477, 139, -11)
   - Separate subprocess per stage (avoids memory accumulation)
   - Numpy-based quintile sorting instead of pandas inner loops
   - Cached .pkl files between stages
   - Pre-converted Excel to CSV to avoid openpyxl crashes
2. **Beta excluded from Stage 2**: RollingOLS segfaults; Beta has weak signal anyway (IC=-0.009)
3. **CSV loading**: Must use `low_memory=False` and auto-infer types (not `dtype=str`)
4. **Returns in raw data**: In percentage points — divide by 100 for decimal
5. **FSPHX data anomaly**: Spike+reversal pairs (+827%/-91%, +902%/-89%) from split/NAV errors. Auto-fixed by `_fix_anomalies()` which detects >500% followed by >-80% and replaces with net monthly equivalent.

## How to Run
```bash
# Install dependencies
uv sync

# Run full pipeline (stages 1-3 + all plots)
uv run python scripts/run_all_stages.py

# Run a single stage
uv run python scripts/stage_1_factors.py
uv run python scripts/stage_2_selection.py
uv run python scripts/stage_3_optimize.py

# Run plots separately
uv run python scripts/plot_comparison.py
uv run python scripts/plot_individual_funds.py
uv run python scripts/plot_factors.py
```

## What's Left To Do

### Priority 1: Black-Litterman (Stage 4) — code exists, needs testing
- Code: `scripts/stage_4_black_litterman.py`, `src/black_litterman/model.py`, `equilibrium.py`, `sensitivity.py`
- Builds factor-based views (P, Q, Omega) from quintile sort results
- Computes BL posterior returns and optimal weights
- Runs OOS evaluation: BL weights vs market-cap weights
- Includes Diebold-Mariano test and tau/delta sensitivity grid
- Needs: integrate into `run_all_stages.py`, test for segfault resilience
- **Why it matters**: Shows Bayesian thinking, combining prior (market equilibrium) with views (our factor signals)

### Priority 2: Rolling Backtest with Transaction Costs — code exists, needs wiring
- Code: `src/portfolio/backtest.py` (`RollingBacktest` class)
- Rolling-window rebalancing (monthly/quarterly/annual) with lookback
- Tracks weight drift between rebalances
- Accounts for transaction costs (configurable bps, e.g. 5-10bps)
- Uses Ledoit-Wolf shrinkage for covariance estimation
- Needs: apply to our 5 portfolio variants, compare net-of-cost Sharpe
- **Why it matters**: Shows practical awareness — backtests without costs are unrealistic, every quant firm cares about implementation shortfall

### Priority 3: Polished Visualization Suite — code exists, needs wiring
- Code: `src/visualization/portfolio_plots.py`
- `plot_efficient_frontier()` — mean-variance frontier with GMV, Max Sharpe, Risk Parity points marked
- `plot_rolling_sharpe_comparison()` — rolling 36M Sharpe for multiple portfolios over time
- `plot_weight_evolution()` — stacked area chart of portfolio weights over time
- All save as interactive HTML + static PNG
- Needs: wire into Stage 3 or a dedicated plotting stage
- **Why it matters**: Classic portfolio theory visuals that interviewers expect; rolling Sharpe shows regime behavior

### Priority 4: Bootstrap Confidence Intervals for Sharpe Ratios — new code needed
- Block bootstrap (preserving autocorrelation) to generate 95% CI on Sharpe ratios
- Instead of just reporting Sharpe = 1.31, show CI [0.8, 1.8] to address "is this luck?"
- Apply to all our variants and benchmarks
- **Why it matters**: Shows statistical rigor, directly addresses overfitting concerns

### Priority 5: Factor Timing / Regime Analysis — new code needed
- Split OOS into bull vs bear periods (e.g., using market drawdowns or VIX threshold)
- Show which factors work in which regime (e.g., momentum crashes in reversals, value works in recoveries)
- Conditional Sharpe ratios by regime
- **Why it matters**: Quant funds care about factor crowding and regime shifts; shows awareness of when strategies fail

### Priority 6: Turnover-Constrained Optimization — new code needed
- Add turnover penalty to MVO/Max Sharpe objective: `min w'Σw - λ·μ'w + κ·||w - w_prev||₁`
- Optimizer explicitly trades off Sharpe vs rebalancing cost
- Show efficient frontier of Sharpe vs turnover
- **Why it matters**: This is what real quant PMs do — net alpha after costs is all that matters

### Priority 7: Factor Risk Attribution — new code needed
- Decompose portfolio returns into contributions from each factor over time
- Show time-varying factor exposures (rolling regression on factor returns)
- Answer "where did the return come from?" for each month
- **Why it matters**: Core skill for portfolio managers, shows you can explain P&L

### Priority 8: Stress Testing — new code needed
- Apply historical scenarios (2008 GFC, 2020 COVID, 2022 rate hikes) to portfolio
- Use full-sample factor returns even if OOS doesn't cover those periods
- Show expected drawdown under each scenario
- **Why it matters**: Standard risk management practice at every quant fund

### Priority 9: OOS Factor Decay Analysis — partially exists
- IC decay already computed in-sample (`src/selection/information_coefficient.py`)
- Run it OOS too — show how quickly each factor's signal decays (alpha half-life)
- Compare in-sample vs OOS decay rates
- **Why it matters**: Directly measures signal persistence, core to any quant strategy's capacity analysis

### Priority 10: Cornish-Fisher VaR — code exists, needs wiring
- Code: `src/analytics/risk.py` has `cornish_fisher_var()`
- Adjusts VaR for skewness and kurtosis (non-normal returns)
- Add to benchmark comparison table alongside historical VaR
- **Why it matters**: Shows you know real return distributions are fat-tailed, not Gaussian

### Future: Documentation / LaTeX Report — not started
- To be done at the end after all features are complete
- Should cover methodology, results, and interpretation
- Target: professional-quality research paper format

## Existing Code Inventory (already written, may need fixes)

### Analytics (`src/analytics/`)
- `performance.py`: Sharpe (annualized), Sortino (annualized × √12), Calmar, max drawdown, descriptive stats, t-tests, turnover stats
- `risk.py`: historical VaR, CVaR, Cornish-Fisher VaR, drawdown stats
- `statistical_tests.py`: Diebold-Mariano test, Sharpe ratio equality test (Jobson-Korkie with Memmel correction)

### Portfolio (`src/portfolio/`)
- `optimization.py`: MVO, Max Sharpe (with gross_leverage constraint), Risk Parity, Global Minimum Variance
- `covariance.py`: Ledoit-Wolf shrinkage
- `backtest.py`: RollingBacktest class (untested)

### Visualization (`src/visualization/`)
- `portfolio_plots.py`: efficient frontier, rolling Sharpe, weight evolution, cumulative comparison
- `factor_plots.py`: QSpread vs benchmark, long/short legs, quintile monotonicity, cumulative vs market, correlation heatmap, IC time series

## User Preferences
- Uses `uv` for Python package management
- Prefers industry project structure over Jupyter notebooks
- Wants documentation/LaTeX done at the end
- Max Sharpe uses gross_leverage=3.0 constraint (same as MVO)
- Do NOT add leverage cap changes without explicit user request
