"""
10_portfolio_backtest.py
Constructs long-short portfolios from each model's predictions.
Long top decile, short bottom decile, monthly rebalanced.
Reports annualized return, Sharpe ratio, max drawdown.

Input:  data/processed/panel_features.csv
Output: outputs/tables/portfolio_results.csv
        outputs/figures/cumulative_returns.png
        outputs/figures/sharpe_comparison.png
        outputs/figures/drawdown.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

print("Loading data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])

FEATURES = [
    'PE_RATIO', 'PX_TO_SALES_RATIO', 'CURRENT_EV_TO_T12M_EBITDA',
    'PX_TO_FREE_CASH_FLOW', 'EQY_DVD_YLD_12M', 'CUR_MKT_CAP',
    'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'GROSS_MARGIN', 'OPER_MARGIN',
    'TOT_DEBT_TO_TOT_EQY', 'CUR_RATIO', 'BETA_RAW_OVERRIDABLE',
    'EARN_YLD', 'VOLATILITY_90D', 'VOLUME_AVG_30D',
    'MOMENTUM_12_1', 'REVERSAL_3M', 'HIGH_52W_RATIO'
]
TARGET = 'FWD_RETURN'

dates = sorted(df['Date'].unique())
MIN_TRAIN_MONTHS = 60

# ============================================================
# Model definitions
# ============================================================
MODELS = {
    'OLS_FamaMacBeth': lambda: Ridge(alpha=0.0001),  # near-zero regularization = OLS
    'Ridge': lambda: Ridge(alpha=0.01),
    'Lasso': lambda: Lasso(alpha=0.001, max_iter=10000),
    'NN_64_32': lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu', solver='adam', alpha=0.01,
        learning_rate='adaptive', learning_rate_init=0.001,
        max_iter=200, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10,
        random_state=42
    ),
}

# ============================================================
# Generate predictions for all models
# ============================================================
def generate_predictions(df, dates, features, target, model_fn, label):
    """Expanding window: train on past, predict current month."""
    all_preds = []
    start = time.time()
    
    for i in range(MIN_TRAIN_MONTHS, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]
        
        train = df[df['Date'].isin(train_dates)][features + [target]].dropna()
        test = df[df['Date'] == test_date].dropna(subset=features + [target])
        
        if len(train) < 100 or len(test) < 20:
            continue
        
        X_train, y_train = train[features].values, train[target].values
        X_test = test[features].values
        
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        preds = test[['Date', 'Ticker', TARGET]].copy()
        preds['PREDICTED'] = y_pred
        all_preds.append(preds)
    
    elapsed = time.time() - start
    print(f"  {label}: {elapsed:.1f}s")
    return pd.concat(all_preds, ignore_index=True)


print("=" * 70)
print("GENERATING PREDICTIONS FOR ALL MODELS")
print("=" * 70)

model_preds = {}
for name, model_fn in MODELS.items():
    model_preds[name] = generate_predictions(
        df, dates, FEATURES, TARGET, model_fn, name
    )

# Also add equal-weight benchmark (no prediction, just equal weight)
print("  Equal-weight benchmark")

# ============================================================
# Construct long-short portfolios
# ============================================================
def construct_portfolio(preds, n_decile=10):
    """
    Each month: long top decile, short bottom decile.
    Returns monthly portfolio returns.
    """
    monthly_returns = []
    
    for date, group in preds.groupby('Date'):
        if len(group) < n_decile * 2:
            continue
        
        group = group.sort_values('PREDICTED', ascending=False)
        n = len(group) // n_decile
        
        # Top decile (long) and bottom decile (short)
        long_stocks = group.head(n)
        short_stocks = group.tail(n)
        
        long_ret = long_stocks[TARGET].mean()
        short_ret = short_stocks[TARGET].mean()
        ls_ret = long_ret - short_ret  # long-short return
        
        # Equal weight benchmark
        eq_ret = group[TARGET].mean()
        
        # Long-only (top decile)
        long_only_ret = long_ret
        
        monthly_returns.append({
            'Date': date,
            'Long_Short': ls_ret,
            'Long_Only': long_only_ret,
            'Short_Only': -short_ret,
            'Equal_Weight': eq_ret
        })
    
    return pd.DataFrame(monthly_returns)


print("\n" + "=" * 70)
print("CONSTRUCTING LONG-SHORT PORTFOLIOS")
print("=" * 70)

portfolios = {}
for name, preds in model_preds.items():
    port = construct_portfolio(preds)
    portfolios[name] = port
    print(f"  {name}: {len(port)} months")

# ============================================================
# Portfolio metrics
# ============================================================
def compute_metrics(returns_series, label="Portfolio"):
    """Compute annualized return, volatility, Sharpe, max drawdown."""
    monthly_mean = returns_series.mean()
    monthly_std = returns_series.std()
    
    ann_return = monthly_mean * 12
    ann_vol = monthly_std * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (returns_series > 0).mean()
    
    return {
        'Model': label,
        'Ann_Return': ann_return,
        'Ann_Volatility': ann_vol,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd,
        'Win_Rate': win_rate,
        'Monthly_Mean': monthly_mean,
        'Monthly_Std': monthly_std,
        'Months': len(returns_series)
    }


print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE — LONG-SHORT")
print("=" * 70)

all_metrics = []
for name, port in portfolios.items():
    metrics = compute_metrics(port['Long_Short'], name)
    all_metrics.append(metrics)
    print(f"\n{name}:")
    print(f"  Annualized Return: {metrics['Ann_Return']:.2%}")
    print(f"  Annualized Vol:    {metrics['Ann_Volatility']:.2%}")
    print(f"  Sharpe Ratio:      {metrics['Sharpe_Ratio']:.3f}")
    print(f"  Max Drawdown:      {metrics['Max_Drawdown']:.2%}")
    print(f"  Win Rate:          {metrics['Win_Rate']:.2%}")

# Add equal-weight benchmark
eq_metrics = compute_metrics(portfolios['Ridge']['Equal_Weight'], 'Equal_Weight')
all_metrics.append(eq_metrics)
print(f"\nEqual Weight Benchmark:")
print(f"  Annualized Return: {eq_metrics['Ann_Return']:.2%}")
print(f"  Sharpe Ratio:      {eq_metrics['Sharpe_Ratio']:.3f}")

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join('outputs', 'tables', 'portfolio_results.csv'), index=False)

# ============================================================
# Plot 1: Cumulative returns
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

colors = {'OLS_FamaMacBeth': '#3498db', 'Ridge': '#2ecc71',
          'Lasso': '#e74c3c', 'NN_64_32': '#9b59b6'}

for name, port in portfolios.items():
    cumulative = (1 + port['Long_Short']).cumprod()
    ax.plot(port['Date'], cumulative, label=name, linewidth=1.5,
            color=colors.get(name, 'gray'))

# Equal weight
eq_cum = (1 + portfolios['Ridge']['Equal_Weight']).cumprod()
ax.plot(portfolios['Ridge']['Date'], eq_cum, label='Equal Weight',
        linewidth=1.5, linestyle='--', color='black')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return ($1 invested)')
ax.set_title('Long-Short Portfolio Cumulative Returns by Model')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'cumulative_returns.png'), dpi=150)
print("\nSaved cumulative_returns.png")
plt.close()

# ============================================================
# Plot 2: Sharpe ratio comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

model_names = metrics_df['Model'].values
sharpes = metrics_df['Sharpe_Ratio'].values
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6']

bars = ax.bar(model_names, sharpes, color=colors_bar[:len(model_names)], edgecolor='black')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Long-Short Portfolio Sharpe Ratios')
ax.axhline(y=0, color='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'sharpe_comparison.png'), dpi=150)
print("Saved sharpe_comparison.png")
plt.close()

# ============================================================
# Plot 3: Drawdown chart
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

for name, port in portfolios.items():
    cumulative = (1 + port['Long_Short']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax.plot(port['Date'], drawdown, label=name, linewidth=1,
            color=colors.get(name, 'gray'))

ax.set_xlabel('Date')
ax.set_ylabel('Drawdown')
ax.set_title('Long-Short Portfolio Drawdowns')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'drawdown.png'), dpi=150)
print("Saved drawdown.png")
plt.close()

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(metrics_df[['Model', 'Ann_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']].to_string(index=False))
print("\nDone!")