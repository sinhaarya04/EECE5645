# Cross-Sectional Stock Return Prediction Using Parallel Machine Learning

**EECE 5645 — Parallel Processing for Data Analytics | Northeastern University**

## Overview

This project applies machine learning to predict one-month-ahead cross-sectional stock returns for S&P 500 constituents and constructs long-short equity portfolios. All models are implemented as distributed PySpark pipelines and benchmarked on the Northeastern Discovery Cluster.

**Key results:**
- Ridge regression achieves the highest predictive accuracy (IC = 0.062)
- Neural network (64-32) achieves the best risk-adjusted returns (Sharpe = 2.27, max drawdown -5.8%)
- All models generate statistically significant Fama-French alpha > 27% annualized (t > 6.5)
- Neural network achieves 4.4x speedup at 16 cores

## Dataset

- **Source:** Bloomberg Terminal
- **Universe:** 499 S&P 500 constituents
- **Period:** January 2010 — November 2025 (209 months)
- **Observations:** 93,237 stock-month records
- **Features:** 19 (valuation, size, profitability, leverage, risk, momentum)
- **Target:** Forward 1-month total return

## Models

| Model | Mean IC | Sharpe Ratio | Approach |
|-------|---------|-------------|----------|
| Fama-MacBeth OLS | 0.060 | 2.03 | Cross-sectional regression with time-series aggregation |
| Ridge (L2) | 0.062 | 2.08 | Regularized regression, all features retained |
| Lasso (L1) | 0.051 | 1.89 | Sparse regression, 6/19 features selected |
| Logistic | AUC 0.54 | — | Binary classification (top vs bottom quintile) |
| SGD | 0.048 | — | Mini-batch stochastic gradient descent |
| Neural Net (64-32) | 0.058 | 2.27 | Two-layer feedforward, ReLU, Adam optimizer |

## Project Structure

```
EECE5645/
├── scripts/                 # All pipeline scripts (01-20)
│   ├── 01_reshape_data.py         # Bloomberg CSV reshape
│   ├── 02_clean_data.py           # Null handling, target construction
│   ├── 03_preprocess_data.py      # Winsorization, z-scoring
│   ├── 04_feature_engineering.py  # Momentum, reversal, regime labels
│   ├── 05_ols_fama_macbeth.py     # Fama-MacBeth two-pass regression
│   ├── 06_ridge_lasso.py          # Ridge & Lasso with expanding window
│   ├── 07_logistic_regression.py  # Binary classification (top/bottom quintile)
│   ├── 08_sgd.py                  # Mini-batch SGD convergence analysis
│   ├── 09_neural_net.py           # MLPRegressor (3 architectures)
│   ├── 10_portfolio_backtest.py   # Long-short decile portfolio
│   ├── 11_regime_analysis.py      # Bull/Neutral/Crash performance
│   ├── 12_factor_decay.py         # Signal decay at 1-12 month horizons
│   ├── 13_spark_pipeline.py       # PySpark distributed ML pipeline
│   ├── 14_spark_speedup.py        # Speedup benchmarking (1-16 cores)
│   ├── 15_descriptive_stats.py    # Feature statistics & correlation
│   ├── 16_model_significance.py   # Paired t-tests, Diebold-Mariano
│   ├── 17_quintile_analysis.py    # Quintile spread monotonicity
│   ├── 18_turnover_costs.py       # Turnover & transaction cost analysis
│   ├── 19_shap_analysis.py        # Feature importance (permutation)
│   └── 20_ff_alpha.py             # Fama-French 5-factor + momentum alpha
├── paper/                   # LaTeX paper
│   ├── main.tex                   # Full paper source
│   └── figures/                   # All figures referenced in paper
├── data/                    # Data files
│   ├── raw/                       # Raw Bloomberg export
│   ├── processed/                 # Cleaned & feature-engineered panels
│   ├── predictions/               # Model prediction outputs
│   └── results/                   # Intermediate results
└── .gitignore
```

## Running the Pipeline

**Prerequisites:** Python 3.8+, PySpark (for scripts 13-14)

```bash
pip install pandas numpy scipy scikit-learn matplotlib pyspark
```

**Execute scripts sequentially:**
```bash
# Data preparation
python scripts/01_reshape_data.py
python scripts/02_clean_data.py
python scripts/03_preprocess_data.py
python scripts/04_feature_engineering.py

# Model training & evaluation
python scripts/05_ols_fama_macbeth.py
python scripts/06_ridge_lasso.py
python scripts/07_logistic_regression.py
python scripts/08_sgd.py
python scripts/09_neural_net.py

# Portfolio & analysis
python scripts/10_portfolio_backtest.py
python scripts/11_regime_analysis.py
python scripts/12_factor_decay.py

# Spark parallelism (requires PySpark)
python scripts/13_spark_pipeline.py --cores 16
python scripts/14_spark_speedup.py

# Additional analyses
python scripts/15_descriptive_stats.py
python scripts/16_model_significance.py
python scripts/17_quintile_analysis.py
python scripts/18_turnover_costs.py
python scripts/19_shap_analysis.py
python scripts/20_ff_alpha.py
```

All outputs are saved to `outputs/tables/` (CSV) and `outputs/figures/` (PNG).

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
```

## References

- Fama & MacBeth (1973). *Risk, Return, and Equilibrium: Empirical Tests*
- Hoerl & Kennard (1970). *Ridge Regression*
- Tibshirani (1996). *Regression Shrinkage and Selection via the Lasso*
- Fama & French (2015). *A Five-Factor Asset Pricing Model*
- Gu, Kelly & Xiu (2020). *Empirical Asset Pricing via Machine Learning*
