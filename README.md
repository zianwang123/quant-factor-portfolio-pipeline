# Quant Factor Portfolio Pipeline

End-to-end quantitative factor investment pipeline: from alpha discovery through portfolio optimization to out-of-sample backtesting with transaction costs.

> **Note**: This is an enhanced independent research project originally developed from group coursework in RSM6308 (factor investing, portfolio optimization, and Black-Litterman allocation). The original course projects covered basic factor construction, simple portfolio optimization, and introductory Black-Litterman — this repo significantly extends and rewrites those foundations with a full rolling backtest framework, transaction cost modeling, adaptive factor re-selection, BL as a return estimator across multiple optimizers, Cornish-Fisher VaR, and comprehensive visualization. **This project may contain errors** — it is intended for educational and research purposes only. See [Disclaimer](#disclaimer) below.

## Key Results (OOS: Jan 2015 – Dec 2019)

### Static OOS Performance (IS: 1987–2014, OOS: 2015–2019)

| Portfolio | Sharpe | Sortino | Ann. Return | Ann. Vol | Max DD | CF-VaR 95% |
|-----------|--------|---------|-------------|----------|--------|------------|
| HF Index Best (HFRIMAI) | 1.358 | 2.371 | 3.2% | 2.4% | -2.4% | -1.0% |
| **Our: Equal Weight** | **1.203** | 2.464 | 3.9% | 3.2% | -3.3% | -1.2% |
| **Our: MAXSER-Lasso (F+S)** | **1.129** | 2.049 | 14.7% | 13.0% | -13.8% | -4.7% |
| **Our: MAXSER-Ridge (F+S)** | **1.125** | 2.213 | 14.6% | 13.0% | -13.2% | -4.3% |
| **Our: IC-Weighted** | **1.122** | 2.298 | 3.9% | 3.5% | -2.2% | -1.2% |
| **Our: MVO + BL** | **1.114** | 2.062 | 5.5% | 4.9% | -5.1% | -1.7% |
| **Our: MVO** | **1.038** | 1.861 | 8.0% | 7.7% | -8.3% | -2.7% |
| **Our: Max Sharpe + BL** | **1.023** | 1.778 | 7.8% | 7.7% | -9.4% | -2.6% |
| **Our: Risk Parity** | **0.998** | 1.814 | 3.1% | 3.1% | -2.8% | -1.2% |
| Mutual Fund (EW) | 0.947 | 1.357 | 16.4% | 17.4% | -21.1% | -6.7% |
| **Our: Max Sharpe** | **0.938** | 1.661 | 10.0% | 10.6% | -12.8% | -3.7% |
| Mutual Fund Best (FSMEX) | 0.919 | 1.465 | 19.0% | 20.7% | -17.1% | -8.2% |
| S&P 500 | 0.907 | 1.219 | 10.8% | 11.9% | -14.1% | -5.2% |
| FF Mkt-RF | 0.857 | 1.103 | 10.7% | 12.4% | -15.1% | -5.5% |
| Smart Beta Best (PSL) | 0.818 | 1.059 | 8.0% | 9.8% | -10.6% | -4.6% |
| Smart Beta (EW) | 0.744 | 0.925 | 8.9% | 11.9% | -16.4% | -5.3% |
| HF Index (EW) | 0.562 | 0.707 | 2.0% | 3.6% | -7.8% | -1.7% |
| FF Momentum (UMD) | 0.201 | 0.351 | 2.6% | 13.2% | -21.3% | -5.9% |
| FF SMB | -0.257 | -0.547 | -2.2% | 8.6% | -17.2% | -4.0% |
| FF HML | -0.472 | -0.968 | -4.3% | 9.0% | -30.1% | -3.8% |

**Why are absolute returns low?** Our factor-only portfolios are **market-neutral long-short** (long Q1, short Q5 within S&P 500). They earn pure alpha with zero beta — the 3–8% returns come from cross-sectional dispersion, not market direction. The S&P 500's 10.8% is mostly equity risk premium, which our portfolios don't take. The right comparison is risk-adjusted: **8 of 9 variants beat the S&P 500 Sharpe (0.91)**, and all beat the HF Index EW (0.56).

**Why include MAXSER?** Standard plug-in optimizers (MVO, Max Sharpe) cannot be applied to the joint [factors + stocks] universe because stock returns contain factor exposures — optimizing over both double-counts factor risk and inflates effective leverage. MAXSER (Ao, Li, Zheng 2019) solves this by decomposing the portfolio into a factor leg and an idiosyncratic leg via sparse regression on the squared Sharpe ratio, with beta-adjustment to avoid double-counting. This is the only principled way to combine factor allocation with stock-level alpha in one portfolio. The higher absolute returns (~14.7%) and vol (~13%) reflect the added stock-specific exposure from 50 selected stocks.

### Rolling Backtest (Net of 10 bps Transaction Costs)

| Strategy | Net Sharpe | Gross Sharpe | Avg Turnover | Total Cost (bps) | Max DD |
|----------|-----------|-------------|-------------|-----------------|--------|
| Equal Weight | **1.211** | 1.215 | 5.0% | 55 | -3.2% |
| Risk Parity | **1.155** | 1.160 | 5.6% | 62 | -3.1% |
| MVO + BL | **1.001** | 1.032 | 27.0% | 300 | -6.4% |
| Max Sharpe + BL | 0.833 | 0.861 | 44.1% | 490 | -4.4% |
| IC-Weighted + BL | 0.815 | 0.839 | 18.2% | 202 | -7.8% |
| MVO | 0.760 | 0.795 | 56.4% | 626 | -13.3% |
| Max Sharpe | 0.719 | 0.756 | 56.1% | 622 | -7.4% |
| IC-Weighted | 0.686 | 0.714 | 19.3% | 214 | -9.0% |

**What happens after transaction costs?** Simple strategies dominate: Equal Weight (net Sharpe 1.21) barely loses anything to costs because turnover is only 5%. MVO loses 3.5 Sharpe points (0.795 → 0.760) from 56% turnover. This is the central tension in quant PM: more sophisticated optimization ≠ better net performance.

**Does Black-Litterman help?** Yes, for optimizers sensitive to return estimation error:
- **MVO**: BL improves net Sharpe from 0.760 to 1.001 (+0.24) and halves turnover (56% → 27%)
- **Max Sharpe**: BL improves net Sharpe from 0.719 to 0.833 (+0.11), reduces turnover (56% → 44%)
- **IC-Weighted**: BL improves net Sharpe from 0.686 to 0.815 (+0.13) — BL helps here by stabilizing the return estimates IC weights are applied to

BL shrinks posterior returns toward the equal-weight equilibrium, dampening the extreme positions that cause MVO instability. It acts as a **return regularizer**, not a standalone allocation method.

## Pipeline Overview

```
Stage 1: Factor Construction     → 20 factors from Compustat/CRSP, quintile sorts, benchmark validation
Stage 2: Factor Selection        → Fama-MacBeth, IC analysis, LASSO, greedy forward selection → 5 factors
Stage 3: Portfolio Optimization  → 9 variants (7 factor-only + 2 MAXSER factor+stock) vs 14 benchmarks
Stage 4: Black-Litterman         → Factor-level & stock-level BL allocation, sensitivity analysis
Stage 5: Rolling Backtest        → Monthly rebalancing, 10 bps costs, BL comparison, adaptive re-selection
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

Additionally, **factor+stock portfolios** combine factor QSpread allocation with stock-level idiosyncratic alpha using MAXSER (Ao, Li, Zheng 2019). MAXSER solves the weight double-counting problem that prevents standard optimizers from being applied to the joint [factors + stocks] space: it decomposes the portfolio into a factor leg (tangency portfolio over K factors) and an idiosyncratic leg (sparse regression on stock residuals), with beta-adjustment to ensure factor exposure is not counted twice.

#### Portfolio Variant Combo Tree

The full combinatorial space of portfolio variants follows this decision tree:

```
Asset Universe        Optimizer        Return Estimation    Estimation Method
──────────────        ─────────        ─────────────────    ─────────────────
Factor-only       ─┬─ EW           ─┬─ Raw sample mu       (MAXSER N/A:
                   ├─ IC-Weighted   │                        only 5 assets)
                   ├─ MVO           ├─ BL posterior mu
                   ├─ Max Sharpe    │
                   └─ Risk Parity  ─┘

Factor+Stocks     ─── MAXSER       ─── Raw sample mu    ─┬─ Lasso
                                                         └─ Ridge
```

This yields 5 x 2 = 10 factor-only + 5 x 2 x 2 = 20 factor+stocks = **30 theoretical combos**. We implement **9** and exclude 21 for principled reasons:

| Excluded Combo | Reason |
|---|---|
| **EW + BL** (factor-only) | EW assigns 1/K regardless of returns -- BL posterior mu has no effect |
| **IC-Weighted + BL** (factor-only & factor+stocks) | IC weights derive from rank correlation, not return estimates -- BL adjusts mu but IC-Weighted doesn't use mu |
| **Risk Parity + BL** (factor-only & factor+stocks) | Risk Parity uses only the covariance matrix, never expected returns -- BL's posterior mu is irrelevant |
| **EW (factor+stocks)** | 1/K across ~279 assets gives each stock ~0.4% weight, drowning factor signal in stock noise |
| **Risk Parity (factor+stocks)** | Same 1/K-risk issue: hundreds of stocks dominate the risk budget over 5 factors |
| **Plug-in F+S (MVO, MaxSharpe, IC-Weighted on [factors+stocks])** | Factors are portfolios of stocks -- optimizing over both double-counts exposure and inflates leverage. MAXSER avoids this via its own sparse regression framework |
| **MAXSER + BL** | MAXSER has its own shrinkage/regularization (Lasso/Ridge on squared Sharpe) -- layering BL would double-regularize |
| **MAXSER + specific optimizer** | MAXSER *is* its own optimizer (maximizes squared Sharpe via sparse regression) -- it replaces MVO/MaxSharpe, not composes with them |

Compared against 14 benchmarks: S&P 500, Fama-French factors (Mkt-RF, SMB, HML, UMD), mutual funds (FSMEX, FLPSX, etc.), hedge fund indices (HFRI), and smart beta ETFs.

Includes efficient frontier visualization and Sharpe ratio equality tests (Jobson-Korkie with Memmel correction).

### Stage 4: Black-Litterman Robustness Analysis

- **Part A (BL Robustness)**: Tests whether Stage 3's BL improvement is robust or fragile to hyperparameter choice. Runs a τ × δ sensitivity grid (30 combinations) through the *actual* Stage 3 optimizers (MVO, Max Sharpe with matching constraints). Baseline τ=0.05, δ=10 chosen a priori from He & Litterman (1999) — the grid is a robustness check, **not** parameter optimization (using OOS to select τ/δ would be data snooping). Also runs Jobson-Korkie and Diebold-Mariano tests for formal statistical significance of BL improvement. Note: with He & Litterman's recommended Omega scaling (Ω ∝ τ), τ cancels out of the posterior mean — we fix Omega at baseline τ to break this cancellation and make τ sensitivity meaningful.
- **Part B (Stock-Level BL)**: 274 S&P 500 stocks — demonstrates why factor-level is the right granularity (stock-level BL Sharpe 0.32 vs market-cap 0.83, views too diffuse across hundreds of stocks).

### Stage 5: Rolling Backtest

Monthly rolling backtest (36-month lookback, quarterly rebalance) with:

- **Transaction costs**: 10 bps per unit turnover
- **Weight drift tracking**: Weights drift with returns between rebalances
- **Ledoit-Wolf shrinkage**: Covariance estimated with shrinkage at each rebalance
- **8 strategies**: 5 base + 3 BL variants (IC-Weighted+BL, MVO+BL, MaxSharpe+BL)
- **Adaptive factor re-selection**: Annual greedy forward selection from full 20-factor universe

**BL improves MVO** (net Sharpe 0.760 → 1.001, turnover halved) and **Max Sharpe** (0.719 → 0.833). BL shrinks extreme weights toward equilibrium, acting as a natural regularizer.

**Adaptive re-selection underperforms fixed factors** across all strategies (EW: 1.211 → 0.727), demonstrating the cost of factor chasing and outer turnover.

### Why Equal Weight Performs Well in the Rolling Backtest

Equal Weight achieves the best net Sharpe in the rolling backtest (1.21) even though it uses no optimization. Two mechanisms drive this:

1. **Quarterly rebalancing harvests mean-reversion.** By resetting to 1/K every quarter, the rolling backtest systematically sells factors that grew (overweighted by drift) and buys factors that shrank. This captures a weak but persistent mean-reversion signal across factor QSpreads — the same effect documented in DeMiguel et al. (2009).

2. **Low turnover preserves alpha.** EW's 5% average turnover costs only 55 bps total, while MVO's 56% turnover costs 626 bps. More sophisticated optimization does not compensate for the transaction cost drag at these rebalancing frequencies.

### Systematic Discipline vs Discretionary Override

A natural question: if the static strategy "works fine this month," should a practitioner stick with the rolling backtest framework or override it?

- **Industry practice**: Practitioners commit to the systematic framework ex ante and follow it. If the rolling backtest says to rebalance, you rebalance — cherry-picking which month to follow is data snooping in real time.
- **Exception**: If the framework itself is broken (data error, regime change like COVID), practitioners may override — but this requires a documented, pre-agreed process (e.g., "halt if drawdown > 15%"), not ad-hoc decisions.
- **Our evidence**: Adaptive factor re-selection — which is exactly "switch when it looks better" — underperforms fixed factors across all strategies (EW: 1.21 → 0.73). This directly demonstrates the cost of second-guessing the systematic process.
- **Key principle**: The backtest result is the expected value over many months. Any single month can deviate. Sticking with the systematic approach is what distinguishes quantitative from discretionary portfolio management.

### Future Work

1. **Bootstrap Confidence Intervals** — Block bootstrap (block size ~12 months) for 95% CI on Sharpe ratios. Answers "is Sharpe 1.20 significantly different from 1.0?"
2. **Factor Risk Attribution** — Decompose monthly portfolio returns into contributions from each factor. Stacked area chart of factor contributions over time.
3. **Turnover-Constrained Optimization** — Add turnover penalty to MVO objective and sweep to produce Sharpe-vs-turnover efficient frontier.
4. **Regime Analysis** — Classify months as bull/bear, compute conditional Sharpe by regime. Shows which factors crash in drawdowns.
5. **Stress Testing** — Apply historical scenarios (2008 GFC, 2020 COVID, 2022 rate hikes) to current portfolio weights using full-sample factor returns.
6. **OOS Factor Decay** — Run IC decay analysis on OOS data and compare with in-sample decay curves to measure alpha half-life.

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
