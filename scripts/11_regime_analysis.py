"""
11_regime_analysis.py
Split portfolio performance by market regime (BULL / NEUTRAL / CRASH).
Requires: scripts 05-10 already run (model predictions + portfolio construction).
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'data/processed/panel_features.csv'
TARGET = 'FWD_RETURN'
FEATURES = [
    'PE_RATIO', 'PX_TO_SALES_RATIO', 'CURRENT_EV_TO_T12M_EBITDA',
    'PX_TO_FREE_CASH_FLOW', 'EQY_DVD_YLD_12M', 'CUR_MKT_CAP',
    'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'GROSS_MARGIN', 'OPER_MARGIN',
    'TOT_DEBT_TO_TOT_EQY', 'CUR_RATIO', 'BETA_RAW_OVERRIDABLE',
    'EARN_YLD', 'VOLATILITY_90D', 'VOLUME_AVG_30D',
    'MOMENTUM_12_1', 'REVERSAL_3M', 'HIGH_52W_RATIO'
]
TRAIN_FRAC = 0.35  # Start test in ~2015 to capture 2020 crash + 2022 selloff

os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(FEATURES_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Recompute regime labels with relaxed thresholds
# -7% trailing 3m = CRASH, +10% trailing 6m = BULL, else NEUTRAL
monthly_mkt = df.groupby('Date')[TARGET].mean().sort_index()
trail_3m = monthly_mkt.rolling(3).sum()
trail_6m = monthly_mkt.rolling(6).sum()

regime_series = pd.Series('NEUTRAL', index=monthly_mkt.index)
regime_series[trail_3m < -0.07] = 'CRASH'
regime_series[(trail_6m > 0.10) & (regime_series != 'CRASH')] = 'BULL'

regime_map = regime_series.reset_index()
regime_map.columns = ['Date', 'REGIME']
print(f"Regime distribution (full dataset):")
print(regime_map['REGIME'].value_counts().to_string())

dates = sorted(df['Date'].unique())
split_idx = int(len(dates) * TRAIN_FRAC)
test_dates = dates[split_idx:]
df_test = df[df['Date'].isin(test_dates)].copy()

test_regimes = regime_map[regime_map['Date'].isin(test_dates)]
print(f"\nRegime distribution (test set only):")
print(test_regimes['REGIME'].value_counts().to_string())

# ============================================================
# REGENERATE PREDICTIONS (same logic as scripts 05-10)
# ============================================================
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS FOR ALL MODELS")
print("=" * 70)

def rolling_predict(df, dates, split_idx, model_fn, label):
    """Expanding-window predictions on test set."""
    import time
    t0 = time.time()
    all_preds = []
    train_dates = dates[:split_idx]

    for i, test_date in enumerate(dates[split_idx:]):
        train = df[df['Date'].isin(train_dates)].dropna(subset=FEATURES + [TARGET])
        test = df[df['Date'] == test_date].dropna(subset=FEATURES + [TARGET])
        if len(test) == 0:
            train_dates = list(train_dates) + [test_date]
            continue

        X_train, y_train = train[FEATURES].values, train[TARGET].values
        X_test, y_test = test[FEATURES].values, test[TARGET].values

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        out = test[['Date', 'Ticker', TARGET]].copy()
        out['PREDICTED'] = preds
        all_preds.append(out)
        train_dates = list(train_dates) + [test_date]

    elapsed = time.time() - t0
    print(f"  {label}: {elapsed:.1f}s")
    return pd.concat(all_preds, ignore_index=True)

model_preds = {}
model_preds['OLS_FamaMacBeth'] = rolling_predict(
    df, dates, split_idx, lambda: LinearRegression(), 'OLS_FamaMacBeth')
model_preds['Ridge'] = rolling_predict(
    df, dates, split_idx, lambda: Ridge(alpha=1.0), 'Ridge')
model_preds['Lasso'] = rolling_predict(
    df, dates, split_idx, lambda: Lasso(alpha=0.001, max_iter=5000), 'Lasso')
model_preds['NN_64_32'] = rolling_predict(
    df, dates, split_idx,
    lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         early_stopping=True, random_state=42),
    'NN_64_32')

# ============================================================
# CONSTRUCT PORTFOLIOS WITH REGIME LABELS
# ============================================================
print("\n" + "=" * 70)
print("CONSTRUCTING PORTFOLIOS & MERGING REGIMES")
print("=" * 70)

def construct_portfolio(preds, n_decile=10):
    monthly_returns = []
    for date, group in preds.groupby('Date'):
        if len(group) < n_decile * 2:
            continue
        group = group.sort_values('PREDICTED', ascending=False)
        n = len(group) // n_decile
        long_ret = group.head(n)[TARGET].mean()
        short_ret = group.tail(n)[TARGET].mean()
        monthly_returns.append({
            'Date': date,
            'Long_Short': long_ret - short_ret,
            'Long_Only': long_ret,
            'Equal_Weight': group[TARGET].mean()
        })
    return pd.DataFrame(monthly_returns)

portfolios = {}
for name, preds in model_preds.items():
    port = construct_portfolio(preds)
    port['Date'] = pd.to_datetime(port['Date'])
    port = port.merge(regime_map, on='Date', how='left')
    portfolios[name] = port
    print(f"  {name}: {len(port)} months")

# Save monthly returns with regimes
all_monthly = []
for name, port in portfolios.items():
    tmp = port[['Date', 'REGIME', 'Long_Short']].copy()
    tmp.rename(columns={'Long_Short': name}, inplace=True)
    all_monthly.append(tmp)

monthly_df = all_monthly[0]
for tmp in all_monthly[1:]:
    monthly_df = monthly_df.merge(tmp[['Date', tmp.columns[-1]]], on='Date')

monthly_df.to_csv('outputs/tables/monthly_returns_with_regime.csv', index=False)
print(f"\nSaved monthly_returns_with_regime.csv ({len(monthly_df)} months)")

# ============================================================
# REGIME PERFORMANCE
# ============================================================
print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE BY REGIME")
print("=" * 70)

def compute_metrics(returns_series, label="Portfolio"):
    if len(returns_series) < 2:
        monthly_mean = returns_series.mean() if len(returns_series) == 1 else 0
        return {
            'Model': label, 'Ann_Return': monthly_mean * 12,
            'Ann_Volatility': 0, 'Sharpe_Ratio': 0,
            'Max_Drawdown': 0, 'Win_Rate': (returns_series > 0).mean() if len(returns_series) > 0 else 0,
            'Months': len(returns_series)
        }
    monthly_mean = returns_series.mean()
    monthly_std = returns_series.std()
    ann_return = monthly_mean * 12
    ann_vol = monthly_std * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    win_rate = (returns_series > 0).mean()
    return {
        'Model': label, 'Ann_Return': ann_return, 'Ann_Volatility': ann_vol,
        'Sharpe_Ratio': sharpe, 'Max_Drawdown': max_dd,
        'Win_Rate': win_rate, 'Months': len(returns_series)
    }

regime_results = []
models = list(portfolios.keys())
regimes = ['BULL', 'NEUTRAL', 'CRASH']

for regime in regimes:
    print(f"\n--- {regime} ---")
    for name in models:
        port = portfolios[name]
        mask = port['REGIME'] == regime
        if mask.sum() == 0:
            print(f"  {name}: no months in {regime}")
            continue
        metrics = compute_metrics(port.loc[mask, 'Long_Short'], name)
        metrics['Regime'] = regime
        regime_results.append(metrics)
        print(f"  {name}: Sharpe={metrics['Sharpe_Ratio']:.3f}  "
              f"Return={metrics['Ann_Return']:.2%}  "
              f"MaxDD={metrics['Max_Drawdown']:.2%}  "
              f"WinRate={metrics['Win_Rate']:.2%}  "
              f"Months={metrics['Months']}")

regime_df = pd.DataFrame(regime_results)
regime_df.to_csv('outputs/tables/regime_performance.csv', index=False)
print(f"\nSaved regime_performance.csv")

# ============================================================
# PLOT 1: Sharpe by regime (grouped bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'OLS_FamaMacBeth': '#3498db', 'Ridge': '#2ecc71',
          'Lasso': '#e74c3c', 'NN_64_32': '#9b59b6'}
x = np.arange(len(regimes))
width = 0.18

for i, name in enumerate(models):
    sharpes = []
    for regime in regimes:
        row = regime_df[(regime_df['Model'] == name) & (regime_df['Regime'] == regime)]
        sharpes.append(row['Sharpe_Ratio'].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, sharpes, width, label=name, color=colors.get(name, '#95a5a6'))

ax.set_xlabel('Market Regime', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Long-Short Sharpe Ratio by Market Regime', fontsize=14)
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(regimes)
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/sharpe_by_regime.png', dpi=150)
print("Saved sharpe_by_regime.png")
plt.close()

# ============================================================
# PLOT 2: Win rate by regime
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
for i, name in enumerate(models):
    win_rates = []
    for regime in regimes:
        row = regime_df[(regime_df['Model'] == name) & (regime_df['Regime'] == regime)]
        win_rates.append(row['Win_Rate'].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, win_rates, width, label=name, color=colors.get(name, '#95a5a6'))

ax.set_xlabel('Market Regime', fontsize=12)
ax.set_ylabel('Win Rate', fontsize=12)
ax.set_title('Long-Short Win Rate by Market Regime', fontsize=14)
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(regimes)
ax.legend()
ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle='--', label='50%')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/winrate_by_regime.png', dpi=150)
print("Saved winrate_by_regime.png")
plt.close()

# ============================================================
# PLOT 3: Cumulative returns colored by regime
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
regime_colors = {'BULL': '#2ecc71', 'NEUTRAL': '#95a5a6', 'CRASH': '#e74c3c'}

for ax, name in zip(axes.flatten(), models):
    port = portfolios[name].sort_values('Date')
    cumulative = (1 + port['Long_Short']).cumprod()

    # Plot segments colored by regime
    for regime, color in regime_colors.items():
        mask = port['REGIME'] == regime
        ax.scatter(port.loc[mask, 'Date'], cumulative[mask],
                   c=color, s=8, label=regime, alpha=0.8)

    ax.plot(port['Date'], cumulative, color='black', linewidth=0.5, alpha=0.3)
    ax.set_title(name, fontsize=12)
    ax.set_ylabel('Cumulative Return')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle('Cumulative Long-Short Returns Colored by Regime', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/figures/cumulative_by_regime.png', dpi=150)
print("Saved cumulative_by_regime.png")
plt.close()

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("REGIME ANALYSIS SUMMARY")
print("=" * 70)
pivot = regime_df.pivot_table(
    values='Sharpe_Ratio', index='Model', columns='Regime'
)
# Only show regimes that exist
available_regimes = [r for r in regimes if r in pivot.columns]
pivot = pivot[available_regimes]
print("\nSharpe Ratios by Regime:")
print(pivot.to_string(float_format='{:.3f}'.format))

print(f"\n⚠️  Note: CRASH regime has only {regime_df[regime_df['Regime']=='CRASH']['Months'].sum()//len(models)} month(s) in test set.")
print("   Consider lowering TRAIN_FRAC to capture more crash periods (e.g., 0.4).")

print("\nDone!")