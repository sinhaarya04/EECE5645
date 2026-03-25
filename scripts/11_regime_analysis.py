"""
11_regime_analysis.py
Split portfolio performance by market regime (BULL / NEUTRAL / CRASH).
Loads pre-computed predictions from data/predictions/ instead of retraining.

Requires: scripts 05, 06, 09 already run.

Input:  data/predictions/*.csv, data/processed/panel_features.csv (for regime labels)
Output: outputs/tables/regime_performance.csv
        outputs/tables/monthly_returns_with_regime.csv
        outputs/figures/sharpe_by_regime.png
        outputs/figures/winrate_by_regime.png
        outputs/figures/cumulative_by_regime.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_PATH = 'data/processed/panel_features.csv'
PRED_DIR = os.path.join('data', 'predictions')
TARGET_COL = 'FWD_RETURN'

os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

MODEL_FILES = {
    'OLS_FamaMacBeth': 'OLS_FamaMacBeth_predictions.csv',
    'Ridge': 'Ridge_predictions.csv',
    'Lasso': 'Lasso_predictions.csv',
    'NN_64_32': 'NN_64_32_predictions.csv',
}

# ============================================================
# LOAD DATA
# ============================================================
print("Loading panel data for regime computation...")
df = pd.read_csv(FEATURES_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Regime classification: -7% trailing 3m = CRASH, +10% trailing 6m = BULL
monthly_mkt = df.groupby('Date')[TARGET_COL].mean().sort_index()
trail_3m = monthly_mkt.rolling(3).sum()
trail_6m = monthly_mkt.rolling(6).sum()

regime_series = pd.Series('NEUTRAL', index=monthly_mkt.index)
regime_series[trail_3m < -0.07] = 'CRASH'
regime_series[(trail_6m > 0.10) & (regime_series != 'CRASH')] = 'BULL'

regime_map = regime_series.reset_index()
regime_map.columns = ['Date', 'REGIME']
print(f"Regime distribution (full dataset):")
print(regime_map['REGIME'].value_counts().to_string())

# ============================================================
# LOAD PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL PREDICTIONS")
print("=" * 70)

model_preds = {}
for name, fname in MODEL_FILES.items():
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {name}")
        continue
    preds = pd.read_csv(path)
    preds['Date'] = pd.to_datetime(preds['Date'])
    model_preds[name] = preds
    print(f"  {name}: {len(preds)} rows, {preds['Date'].nunique()} months")

if not model_preds:
    raise FileNotFoundError("No prediction files found. Run scripts 05, 06, 09 first.")

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
        group = group.sort_values('predicted_return', ascending=False)
        n = len(group) // n_decile
        long_ret = group.head(n)['actual_return'].mean()
        short_ret = group.tail(n)['actual_return'].mean()
        monthly_returns.append({
            'Date': date,
            'Long_Short': long_ret - short_ret,
            'Long_Only': long_ret,
            'Equal_Weight': group['actual_return'].mean()
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
ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle='--')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/winrate_by_regime.png', dpi=150)
print("Saved winrate_by_regime.png")
plt.close()

# ============================================================
# PLOT 3: Cumulative returns colored by regime
# ============================================================
n_models = len(models)
ncols = 2
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
regime_colors = {'BULL': '#2ecc71', 'NEUTRAL': '#95a5a6', 'CRASH': '#e74c3c'}

for idx, name in enumerate(models):
    ax = axes.flatten()[idx] if n_models > 1 else axes
    port = portfolios[name].sort_values('Date')
    cumulative = (1 + port['Long_Short']).cumprod()

    for regime, color in regime_colors.items():
        mask = port['REGIME'] == regime
        ax.scatter(port.loc[mask, 'Date'], cumulative[mask],
                   c=color, s=8, label=regime, alpha=0.8)

    ax.plot(port['Date'], cumulative, color='black', linewidth=0.5, alpha=0.3)
    ax.set_title(name, fontsize=12)
    ax.set_ylabel('Cumulative Return')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Hide extra subplots if odd number of models
for idx in range(n_models, nrows * ncols):
    axes.flatten()[idx].set_visible(False)

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
available_regimes = [r for r in regimes if r in pivot.columns]
pivot = pivot[available_regimes]
print("\nSharpe Ratios by Regime:")
print(pivot.to_string(float_format='{:.3f}'.format))

print("\nDone!")
