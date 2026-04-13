"""
18_turnover_costs.py
Turnover analysis and transaction-cost-adjusted portfolio returns.
Measures how much each model's long-short portfolio trades each month,
then computes performance metrics at various cost levels.

Requires: scripts 05, 06, 09 already run (predictions saved to data/predictions/).

Input:  data/predictions/*.csv
Output: outputs/tables/turnover_analysis.csv
        outputs/tables/cost_adjusted_returns.csv
        outputs/figures/turnover_timeseries.png
        outputs/figures/sharpe_after_costs.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PRED_DIR = os.path.join('data', 'predictions')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

MODEL_FILES = {
    'OLS_FamaMacBeth': 'OLS_FamaMacBeth_predictions.csv',
    'Ridge': 'Ridge_predictions.csv',
    'Lasso': 'Lasso_predictions.csv',
    'NN_64_32': 'NN_64_32_predictions.csv',
}

COLORS = {
    'OLS_FamaMacBeth': '#3498db',
    'Ridge': '#2ecc71',
    'Lasso': '#e74c3c',
    'NN_64_32': '#9b59b6',
}

COST_LEVELS_BPS = [0, 10, 20, 50]

# ============================================================
# Load predictions
# ============================================================
print("=" * 70)
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
# Turnover computation
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING PORTFOLIO TURNOVER")
print("=" * 70)


def compute_turnover_and_returns(preds, n_decile=10):
    """
    For each month:
      - Identify top decile (long) and bottom decile (short) by predicted_return.
      - Compare to previous month's long/short sets to compute turnover.
      - Compute long-short return.

    Returns:
        DataFrame with Date, long_ret, short_ret, ls_ret,
        long_turnover, short_turnover, total_turnover
    """
    dates = sorted(preds['Date'].unique())
    prev_long_set = set()
    prev_short_set = set()
    records = []

    for date in dates:
        group = preds[preds['Date'] == date].copy()
        if len(group) < n_decile * 2:
            continue

        group = group.sort_values('predicted_return', ascending=False)
        n = len(group) // n_decile

        long_stocks = group.head(n)
        short_stocks = group.tail(n)

        long_set = set(long_stocks['Ticker'].values)
        short_set = set(short_stocks['Ticker'].values)

        long_ret = long_stocks['actual_return'].mean()
        short_ret = short_stocks['actual_return'].mean()
        ls_ret = long_ret - short_ret

        # Turnover: fraction of stocks that changed vs previous month
        if len(prev_long_set) > 0:
            long_new = len(long_set - prev_long_set)
            long_turnover = long_new / len(long_set)
            short_new = len(short_set - prev_short_set)
            short_turnover = short_new / len(short_set)
        else:
            # First month — no previous portfolio to compare
            long_turnover = np.nan
            short_turnover = np.nan

        total_turnover = np.nanmean([long_turnover, short_turnover])

        records.append({
            'Date': date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': ls_ret,
            'long_turnover': long_turnover,
            'short_turnover': short_turnover,
            'total_turnover': total_turnover,
        })

        prev_long_set = long_set
        prev_short_set = short_set

    return pd.DataFrame(records)


model_turnover = {}
for name in model_preds:
    df = compute_turnover_and_returns(model_preds[name])
    model_turnover[name] = df
    valid = df.dropna(subset=['total_turnover'])
    print(f"  {name}: {len(df)} months, "
          f"avg turnover = {valid['total_turnover'].mean():.2%}")

# ============================================================
# Turnover summary table
# ============================================================
print("\n" + "=" * 70)
print("TURNOVER SUMMARY")
print("=" * 70)

turnover_rows = []
models = list(model_turnover.keys())
for name in models:
    df = model_turnover[name].dropna(subset=['total_turnover'])
    turnover_rows.append({
        'Model': name,
        'Avg_Long_Turnover': df['long_turnover'].mean(),
        'Avg_Short_Turnover': df['short_turnover'].mean(),
        'Avg_Total_Turnover': df['total_turnover'].mean(),
        'Months': len(df),
    })
    print(f"  {name}:")
    print(f"    Long turnover:  {df['long_turnover'].mean():.2%}")
    print(f"    Short turnover: {df['short_turnover'].mean():.2%}")
    print(f"    Total turnover: {df['total_turnover'].mean():.2%}")

turnover_df = pd.DataFrame(turnover_rows)
turnover_df.to_csv(os.path.join('outputs', 'tables', 'turnover_analysis.csv'), index=False)
print(f"\nSaved turnover_analysis.csv")

# ============================================================
# Transaction-cost-adjusted returns
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING COST-ADJUSTED RETURNS")
print("=" * 70)


def compute_metrics(returns_series):
    """Compute annualized return, volatility, Sharpe, max drawdown."""
    monthly_mean = returns_series.mean()
    monthly_std = returns_series.std()

    ann_return = monthly_mean * 12
    ann_vol = monthly_std * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        'Ann_Return': ann_return,
        'Ann_Volatility': ann_vol,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd,
    }


cost_rows = []
for name in models:
    df = model_turnover[name].dropna(subset=['total_turnover']).copy()
    raw_ret = df['ls_ret']
    avg_turnover = df['total_turnover'].mean()

    for cost_bps in COST_LEVELS_BPS:
        cost_per_trade = cost_bps / 10000.0
        # Monthly cost = turnover * cost_per_trade * 2 (long + short legs)
        monthly_cost = df['total_turnover'] * cost_per_trade * 2
        adjusted_ret = raw_ret - monthly_cost

        metrics = compute_metrics(adjusted_ret)

        cost_rows.append({
            'Model': name,
            'Cost_bps': cost_bps,
            'Avg_Turnover': avg_turnover,
            'Ann_Return': metrics['Ann_Return'],
            'Ann_Volatility': metrics['Ann_Volatility'],
            'Sharpe_Ratio': metrics['Sharpe_Ratio'],
            'Max_Drawdown': metrics['Max_Drawdown'],
        })

    print(f"  {name}:")
    for cost_bps in COST_LEVELS_BPS:
        row = [r for r in cost_rows if r['Model'] == name and r['Cost_bps'] == cost_bps][0]
        print(f"    {cost_bps:>3d} bps: Sharpe={row['Sharpe_Ratio']:.3f}  "
              f"Return={row['Ann_Return']:.2%}  MaxDD={row['Max_Drawdown']:.2%}")

cost_df = pd.DataFrame(cost_rows)
cost_df.to_csv(os.path.join('outputs', 'tables', 'cost_adjusted_returns.csv'), index=False)
print(f"\nSaved cost_adjusted_returns.csv")

# ============================================================
# Plot 1: Turnover time series
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING TURNOVER TIME SERIES")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 5))

for name in models:
    df = model_turnover[name].dropna(subset=['total_turnover'])
    ax.plot(df['Date'], df['total_turnover'] * 100, label=name,
            linewidth=1.2, color=COLORS.get(name, 'gray'), alpha=0.85)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Monthly Turnover (%)', fontsize=12)
ax.set_title('Portfolio Turnover Over Time (Long-Short, Top/Bottom Decile)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'turnover_timeseries.png'), dpi=150)
print("Saved turnover_timeseries.png")
plt.close()

# ============================================================
# Plot 2: Sharpe after costs (grouped bar chart)
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING SHARPE AFTER COSTS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 7))

cost_labels = [f'{c} bps' for c in COST_LEVELS_BPS]
x = np.arange(len(COST_LEVELS_BPS))
n_models = len(models)
width = 0.8 / n_models

for i, name in enumerate(models):
    sharpes = []
    for cost_bps in COST_LEVELS_BPS:
        row = cost_df[(cost_df['Model'] == name) & (cost_df['Cost_bps'] == cost_bps)]
        sharpes.append(row['Sharpe_Ratio'].values[0] if len(row) > 0 else 0)
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, sharpes, width, label=name,
                  color=COLORS.get(name, '#95a5a6'), edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Transaction Cost Level', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Long-Short Sharpe Ratio at Different Transaction Cost Levels', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(cost_labels)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'sharpe_after_costs.png'), dpi=150)
print("Saved sharpe_after_costs.png")
plt.close()

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 70)
print("COST-ADJUSTED RETURNS SUMMARY")
print("=" * 70)

display_df = cost_df.copy()
display_df['Ann_Return'] = display_df['Ann_Return'].map(lambda x: f"{x:.2%}")
display_df['Ann_Volatility'] = display_df['Ann_Volatility'].map(lambda x: f"{x:.2%}")
display_df['Sharpe_Ratio'] = cost_df['Sharpe_Ratio'].map(lambda x: f"{x:.3f}")
display_df['Max_Drawdown'] = display_df['Max_Drawdown'].map(lambda x: f"{x:.2%}")
display_df['Avg_Turnover'] = display_df['Avg_Turnover'].map(lambda x: f"{x:.2%}")
print(display_df[['Model', 'Cost_bps', 'Ann_Return', 'Sharpe_Ratio', 'Max_Drawdown']].to_string(index=False))

print("\nDone!")
