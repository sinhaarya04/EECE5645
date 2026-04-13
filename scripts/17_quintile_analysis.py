"""
17_quintile_analysis.py
Quintile spread analysis: sort stocks into quintiles by predicted return,
measure actual performance, test for monotonicity.

Requires: scripts 05, 06, 09 already run (predictions saved to data/predictions/).

Input:  data/predictions/*.csv
Output: outputs/tables/quintile_returns.csv
        outputs/figures/quintile_monotonicity.png
"""

import pandas as pd
import numpy as np
from scipy import stats
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

QUINTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

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
# Quintile analysis
# ============================================================
print("\n" + "=" * 70)
print("QUINTILE ANALYSIS")
print("=" * 70)


def quintile_analysis(preds, n_quintiles=5):
    """
    For each month, sort stocks by predicted_return (descending),
    split into quintiles, and compute mean actual_return per quintile.
    Q1 = top 20% (highest predicted), Q5 = bottom 20%.

    Returns:
        monthly_quintile_returns: DataFrame with Date, Q1..Q5
        monthly_spreads: Series of Q1-Q5 each month
    """
    monthly_rows = []

    for date, group in preds.groupby('Date'):
        if len(group) < n_quintiles * 2:
            continue

        group = group.sort_values('predicted_return', ascending=False).reset_index(drop=True)
        n = len(group)
        quintile_size = n // n_quintiles

        row = {'Date': date}
        for q in range(n_quintiles):
            start = q * quintile_size
            # Last quintile gets any remainder
            end = (q + 1) * quintile_size if q < n_quintiles - 1 else n
            q_label = f'Q{q + 1}'
            row[q_label] = group.iloc[start:end]['actual_return'].mean()

        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df['Spread'] = monthly_df['Q1'] - monthly_df['Q5']

    return monthly_df


results = []
models = list(model_preds.keys())

for name in models:
    preds = model_preds[name]
    monthly_q = quintile_analysis(preds)

    n_months = len(monthly_q)

    # Average returns across months
    mean_returns = {q: monthly_q[q].mean() for q in QUINTILE_LABELS}

    # Spread statistics
    spread_mean = monthly_q['Spread'].mean()
    spread_std = monthly_q['Spread'].std()
    spread_t = spread_mean / (spread_std / np.sqrt(n_months)) if spread_std > 0 else 0.0

    # Monotonicity check: Q1 > Q2 > Q3 > Q4 > Q5
    q_means = [mean_returns[q] for q in QUINTILE_LABELS]
    monotonic = all(q_means[i] > q_means[i + 1] for i in range(len(q_means) - 1))

    row = {'Model': name}
    row.update(mean_returns)
    row['Spread'] = spread_mean
    row['Spread_t_stat'] = spread_t
    row['Monotonic'] = monotonic

    results.append(row)

    print(f"\n{name} ({n_months} months):")
    for q in QUINTILE_LABELS:
        print(f"  {q}: {mean_returns[q]:+.4f} ({mean_returns[q] * 100:+.2f}%)")
    print(f"  Spread (Q1-Q5): {spread_mean:+.4f}  t-stat: {spread_t:.2f}")
    print(f"  Monotonic: {monotonic}")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join('outputs', 'tables', 'quintile_returns.csv'), index=False)
print(f"\nSaved quintile_returns.csv")

# ============================================================
# Console summary
# ============================================================
print("\n" + "=" * 70)
print("QUINTILE RETURNS SUMMARY")
print("=" * 70)

# Format for display
display_df = results_df.copy()
for q in QUINTILE_LABELS:
    display_df[q] = display_df[q].map(lambda x: f"{x * 100:.3f}%")
display_df['Spread'] = display_df['Spread'].map(lambda x: f"{x * 100:.3f}%")
display_df['Spread_t_stat'] = results_df['Spread_t_stat'].map(lambda x: f"{x:.2f}")
print(display_df.to_string(index=False))

# ============================================================
# Plot: Quintile Monotonicity Bar Chart
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING QUINTILE MONOTONICITY CHART")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(QUINTILE_LABELS))
n_models = len(models)
width = 0.8 / n_models

for i, name in enumerate(models):
    row = results_df[results_df['Model'] == name].iloc[0]
    vals = [row[q] * 100 for q in QUINTILE_LABELS]  # convert to %
    offset = (i - (n_models - 1) / 2) * width
    ax.bar(x + offset, vals, width, label=name, color=COLORS.get(name, '#95a5a6'),
           edgecolor='white', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Quintile (Q1 = Highest Predicted Return)', fontsize=12)
ax.set_ylabel('Mean Monthly Return (%)', fontsize=12)
ax.set_title('Quintile Monotonicity: Mean Actual Return by Predicted Quintile', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(QUINTILE_LABELS)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'quintile_monotonicity.png'), dpi=150)
print("Saved quintile_monotonicity.png")
plt.close()

print("\nDone!")
