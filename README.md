# Markowitz Portfolio Analyzer

A production-ready single-page app for Modern Portfolio Theory analysis with Monte Carlo simulation.

## Live Demo
**[https://facubelini.github.io/markowitz-analyzer/](https://facubelini.github.io/markowitz-analyzer/)**

## Features

- **Efficient Frontier** — 5,000 Monte Carlo portfolios colored by Sharpe ratio
- **Optimal Portfolios** — Maximum Sharpe & Minimum Variance with weight breakdowns
- **Correlation Matrix** — Color-coded heatmap of asset correlations
- **Asset Statistics** — Annualized return, volatility, Sharpe, beta (sortable table)

## Usage

1. Enter 2–10 stock tickers (defaults: AAPL, MSFT, GOOGL, AMZN)
2. Click **Run Analysis**
3. Explore the efficient frontier and optimal allocations

## Methodology

| Parameter | Value |
|---|---|
| Data source | Yahoo Finance (adjusted close) |
| Lookback | 2 years daily prices |
| Returns | Daily log returns, annualized × 252 |
| Simulation | 5,000 Dirichlet-sampled random portfolios |
| Risk-free rate | 5% (U.S. Treasury approximation) |
| Sharpe Ratio | `(Return − Rf) / Volatility` |
| Beta | Covariance with first asset / variance of first asset |

## Tech Stack

- Vanilla HTML + CSS + JavaScript (no build step)
- [Chart.js 4.4](https://www.chartjs.org/) via CDN
- Yahoo Finance public API (CORS fallback via corsproxy.io)
- GitHub Pages hosting

## Known Limitations

- Yahoo Finance API may occasionally be unavailable or throttled
- Beta is computed relative to the first ticker, not a market index
- Monte Carlo uses random sampling — results vary slightly between runs
- Does not account for transaction costs or taxes
