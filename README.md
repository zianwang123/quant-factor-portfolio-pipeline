# Quant Factor Portfolio Pipeline

End-to-end quantitative factor investment pipeline: from alpha discovery through portfolio optimization to out-of-sample backtesting with transaction costs.

> **Note**: This is an enhanced independent research project originally developed from group coursework in RSM6308 (factor investing, portfolio optimization, and Black-Litterman allocation). The original course projects covered basic factor construction, simple portfolio optimization, and introductory Black-Litterman — this repo significantly extends and rewrites those foundations with a full rolling backtest framework, transaction cost modeling, adaptive factor re-selection, BL as a return estimator across multiple optimizers, Cornish-Fisher VaR, and comprehensive visualization. **This project may contain errors** — it is intended for educational and research purposes only. See [Disclaimer](#disclaimer) below.

## Key Results (OOS: 2015–2019)

| Portfolio | Sharpe | Ann. Return | Ann. Vol | Max DD |
|-----------|--------|-------------|----------|--------|
| Our: IC-Weighted | **1.311** | 4.3% | 3.3% | -1.9% |
| Our: MVO + BL | **1.242** | 5.7% | 4.6% | -4.5% |
| Our: Equal Weight | **1.201** | 3.7% | 3.1% | -3.2% |
| S&P 500 | 0.907 | 10.8% | 12.0% | -14.1% |
| Hedge Fund Index (EW) | 0.562 | 2.0% | 3.6% | -7.8% |

Our portfolios are **market-neutral long-short** (factor QSpreads), so lower absolute returns but superior risk-adjusted performance vs the S&P 500 and hedge fund benchmarks.

## Pipeline Overview

```
Stage 1: Factor Construction     → 20 factors from Compustat/CRSP, quintile sorts, benchmark validation
Stage 2: Factor Selection        → Fama-MacBeth, IC analysis, LASSO, greedy forward selection → 5 factors
Stage 3: Portfolio Optimization  → 7 variants (EW, IC-Wt, MVO, Max Sharpe, Risk Parity, MVO+BL, MaxSharpe+BL)
Stage 4: Black-Litterman         → Factor-level & stock-level BL allocation, sensitivity analysis
Stage 5: Rolling Backtest        → Monthly rebalancing, transaction costs, BL comparison, adaptive re-selection
```

### Stage 1: Factor Construction

Constructs 20 cross-sectional factors from Compustat/CRSP merged data (1970–2019):

| Category | Factors |
|----------|---------|
| Value | Book-to-Price, Cash Flow to Price, Earnings Yield, Dividend Yield |
| Size | Log Market Cap |
| Momentum | 12-1 Month Momentum, Short-Term Reversal, High-Low 1M, Long-Term Growth |
| Quality | Accrual Ratio, ROE, Net Profit Margin, Gross Profit |
| Growth | Sales Growth, Asset Growth, Sustainable Growth |
| Analyst | Earnings Revision, Standardized Unexpected Earnings |
| Risk | Annualized 12M Volatility |
| Leverage | Debt-to-Equity |

Each factor is validated via quintile sorts (long Q1 vs short Q5) and benchmarked against Capital IQ factor indices with correlations **0.87–0.99**.

### Stage 2: Factor Selection

Four selection methods vote on which factors to include:

1. **Fama-MacBeth regression** — univariate cross-sectional regressions with Newey-West standard errors
2. **Information Coefficient** — rank IC and IC-IR filtering (min IC > 0.02, IC-IR > 0.5)
3. **LASSO** — L1-penalized regression with 5-fold cross-validation
4. **Greedy forward selection** — maximize |QSpread Sharpe| subject to pairwise correlation < 0.6

**Selected factors**: AccrualRatio (quality), CFTP (value), STReversal (momentum), AssetGrowth (growth), ROE (quality)

### Stage 3: Portfolio Optimization

Seven portfolio variants constructed from selected factor QSpreads:

| Variant | Method | Constraints |
|---------|--------|-------------|
| Equal Weight | 1/N allocation | — |
| IC-Weighted | Weights ∝ in-sample IC | Long-only |
| MVO | Mean-variance optimization | Leverage ≤ 3×, weights ∈ [-2, 2] |
| Max Sharpe | Maximum Sharpe ratio | Leverage ≤ 3× |
| Risk Parity | Equal risk contribution | Long-only |
| MVO + BL | MVO with BL posterior returns | Leverage ≤ 2×, weights ∈ [-0.5, 1] |
| Max Sharpe + BL | Max Sharpe with BL posterior returns | Leverage ≤ 3× |

Black-Litterman variants use BL as a **return estimation method**: compute posterior expected returns from equilibrium + views (in-sample means), then feed to the optimizer.

Compared against 14 benchmarks: S&P 500, Fama-French factors (Mkt-RF, SMB, HML, UMD), mutual funds (FSMEX, FLPSX, etc.), hedge fund indices (HFRI), and smart beta ETFs.

Includes efficient frontier visualization and Sharpe ratio equality tests (Jobson-Korkie with Memmel correction).

### Stage 4: Black-Litterman

- **Part A (Factor-Level)**: BL applied to factor allocation weights. Prior = equal-weight; views from IC and FM analysis. He & Litterman (1999) Omega scaling. Sensitivity grid over τ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} × δ ∈ {1, 5, 10, 50, 100}.
- **Part B (Stock-Level)**: 274 S&P 500 stocks — shown for academic completeness. Views too diffuse to beat market-cap weighting.

### Stage 5: Rolling Backtest

Monthly rolling backtest (36-month lookback, quarterly rebalance) with:

- **Transaction costs**: 10 bps per unit turnover
- **Weight drift tracking**: Weights drift with returns between rebalances
- **Ledoit-Wolf shrinkage**: Covariance estimated with shrinkage at each rebalance
- **8 strategies**: 5 base + 3 BL variants (IC-Weighted+BL, MVO+BL, MaxSharpe+BL)

| Strategy | Net Sharpe | Avg Turnover | Total Cost (bps) |
|----------|-----------|-------------|-----------------|
| Equal Weight | 1.184 | 4.9% | 55 |
| Risk Parity | 1.143 | 5.5% | 60 |
| Max Sharpe + BL | 0.913 | 38.1% | 423 |
| MVO + BL | 0.892 | 27.7% | 307 |

**BL improves MVO and Max Sharpe** (Sharpe +0.20 and +0.10 respectively) while reducing turnover. BL shrinks extreme weights toward equilibrium, acting as a natural regularizer.

**Adaptive factor re-selection** (annual re-selection from full 20-factor universe) underperforms fixed factors across all strategies, demonstrating the cost of factor chasing and outer turnover.

## Project Structure

```
src/
  data/               Data loading (Compustat/CRSP)
  factors/            20 factor implementations + quintile sort validation
  selection/          FM, IC, LASSO, greedy forward selection
  portfolio/          MVO, Max Sharpe, Risk Parity, GMV, RollingBacktest
  black_litterman/    BL posterior, views, equilibrium, sensitivity
  analytics/          Sharpe, Sortino, Calmar, VaR, CVaR, Cornish-Fisher VaR
  visualization/      Plotly charts (efficient frontier, rolling Sharpe, weight evolution)
scripts/
  run_all_stages.py   Orchestrator — runs all stages with retry on segfaults
  stage_1_factors.py  Factor construction + benchmark validation
  stage_2_selection.py Factor selection (4 methods)
  stage_3_optimize.py  Portfolio optimization (7 variants) + benchmarks
  stage_4_black_litterman.py  BL allocation (factor + stock level)
  stage_5_backtest.py  Rolling backtest + BL comparison + adaptive re-selection
  plot_*.py           Comparison, individual fund, and factor-level plots
config/
  pipeline.yaml       All pipeline parameters
outputs/
  run_YYYYMMDD_HHMMSS/  Timestamped run folders with tables/ and figures/
```

## Setup & Usage

**Requirements**: Python 3.12, [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone and install
git clone https://github.com/zianwang123/quant-factor-portfolio-pipeline.git
cd quant-factor-portfolio-pipeline
uv sync

# Run full pipeline (all stages + plots)
uv run python scripts/run_all_stages.py

# Run individual stages
uv run python scripts/stage_1_factors.py
uv run python scripts/stage_2_selection.py
uv run python scripts/stage_3_optimize.py
uv run python scripts/stage_4_black_litterman.py
uv run python scripts/stage_5_backtest.py
```

Each stage caches intermediate results as `.pkl` files in `data/processed/cache/`, so stages can be run independently after Stage 1 completes.

## Data

- **Compustat/CRSP merged** (~173 MB): Monthly stock-level fundamentals and returns, 1970–2019
- **S&P 500 constituents**: Monthly index membership for universe filtering
- **Capital IQ factor indices**: Benchmark factor returns for validation
- **Fama-French factors**: Mkt-RF, SMB, HML, UMD monthly returns
- **Benchmark funds**: Mutual funds (FSMEX, FLPSX, FCNTX, etc.), hedge fund indices (HFRI), smart beta ETFs

Raw data files are not included in the repository.

## Methodology Notes

- **QSpreads**: Long-short returns from quintile sorts (Q1 − Q5 or Q5 − Q1 depending on factor direction). Market-neutral by construction.
- **Sharpe ratios**: All computed on excess returns. Long-short QSpreads need no risk-free subtraction; long-only benchmarks subtract monthly rf.
- **In-sample / Out-of-sample**: IS = 1987–2014 (factor construction + optimization), OOS = 2015–2019 (evaluation only).
- **Cornish-Fisher VaR**: Adjusts Gaussian VaR for skewness and excess kurtosis of empirical return distributions.
- **BL posterior**: μ̄ = (Σ̃⁻¹ + P'Ω⁻¹P)⁻¹(Σ̃⁻¹π + P'Ω⁻¹Q), where Σ̃ = τΣ, π = δΣw_eq. Omega diagonal scaled per He & Litterman (1999).

## Tech Stack

- Python 3.12 with `uv` package manager
- NumPy, Pandas, SciPy, statsmodels, scikit-learn
- CVXPY for convex optimization
- Plotly for interactive HTML charts + Kaleido for static PNG export
- PyYAML for configuration

## Disclaimer

**This project is for educational and research purposes only. It does not constitute financial advice, investment recommendation, or solicitation to buy or sell any securities.**

- This codebase originated from university coursework (RSM6308) and has been independently extended by the author. It may contain errors, bugs, or methodological oversights.
- Past performance shown in backtests does not guarantee future results. Backtested results are hypothetical, do not reflect actual trading, and may overstate real-world performance.
- The raw data (Compustat/CRSP, Capital IQ benchmarks, HFRI indices) used in this project is proprietary and **not included** in this repository. Users must obtain their own data licenses from the respective providers (S&P Global, CRSP, Hedge Fund Research, etc.).
- The author makes no representations or warranties regarding the accuracy, completeness, or reliability of the code or results.
- Use at your own risk. The author is not liable for any financial losses or damages resulting from the use of this software.

## License

This project is licensed under the [MIT License](LICENSE).
