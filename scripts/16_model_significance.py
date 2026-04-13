"""
16_model_significance.py
Statistical significance tests between models + IC distribution analysis.

Input:  data/predictions/OLS_FamaMacBeth_predictions.csv
        data/predictions/Ridge_predictions.csv
        data/predictions/Lasso_predictions.csv
        data/predictions/NN_64_32_predictions.csv
Output: outputs/tables/ic_information_ratio.csv
        outputs/tables/model_significance_tests.csv
        outputs/figures/ic_histogram.png
        outputs/figures/ic_boxplot.png
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations

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

# ============================================================
# 1. LOAD PREDICTIONS
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
# 2. COMPUTE MONTHLY SPEARMAN IC FOR EACH MODEL
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING MONTHLY SPEARMAN IC")
print("=" * 70)

monthly_ics = {}  # {model_name: DataFrame with Date, IC}

for name, preds in model_preds.items():
    ic_records = []
    for date, group in preds.groupby('Date'):
        if len(group) < 10:
            continue
        pred = group['predicted_return'].values
        actual = group['actual_return'].values
        valid = ~(np.isnan(pred) | np.isnan(actual))
        if valid.sum() < 10:
            continue
        ic, _ = stats.spearmanr(pred[valid], actual[valid])
        ic_records.append({'Date': date, 'IC': ic})
    ic_df = pd.DataFrame(ic_records)
    monthly_ics[name] = ic_df
    print(f"  {name}: {len(ic_df)} months, Mean IC = {ic_df['IC'].mean():.4f}, "
          f"Std = {ic_df['IC'].std():.4f}")

# ============================================================
# 3. IC INFORMATION RATIO TABLE
# ============================================================
print("\n" + "=" * 70)
print("IC INFORMATION RATIO")
print("=" * 70)

ir_rows = []
for name in model_preds:
    ic_df = monthly_ics[name]
    n_months = len(ic_df)
    mean_ic = ic_df['IC'].mean()
    ic_std = ic_df['IC'].std()
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0
    ic_t = mean_ic / (ic_std / np.sqrt(n_months)) if ic_std > 0 else 0
    hit_rate = (ic_df['IC'] > 0).mean()

    ir_rows.append({
        'Model': name,
        'Mean_IC': mean_ic,
        'IC_Std': ic_std,
        'IC_IR': ic_ir,
        'IC_t_stat': ic_t,
        'n_months': n_months,
        'Hit_Rate': hit_rate,
    })

    print(f"  {name}:")
    print(f"    Mean IC    = {mean_ic:.4f}")
    print(f"    IC Std     = {ic_std:.4f}")
    print(f"    IC IR      = {ic_ir:.4f}")
    print(f"    t-stat     = {ic_t:.4f}")
    print(f"    Hit Rate   = {hit_rate:.2%}")
    print(f"    n_months   = {n_months}")

ir_df = pd.DataFrame(ir_rows)
ir_df.to_csv(os.path.join('outputs', 'tables', 'ic_information_ratio.csv'), index=False)
print("\nSaved ic_information_ratio.csv")

# ============================================================
# 4. PAIRWISE SIGNIFICANCE TESTS
# ============================================================
print("\n" + "=" * 70)
print("PAIRWISE MODEL SIGNIFICANCE TESTS")
print("=" * 70)

# Also compute monthly MSE for Diebold-Mariano test
monthly_mse = {}  # {model_name: DataFrame with Date, MSE}

for name, preds in model_preds.items():
    mse_records = []
    for date, group in preds.groupby('Date'):
        if len(group) < 10:
            continue
        mse = ((group['predicted_return'] - group['actual_return']) ** 2).mean()
        mse_records.append({'Date': date, 'MSE': mse})
    monthly_mse[name] = pd.DataFrame(mse_records)

model_names = list(model_preds.keys())
sig_rows = []

for name_a, name_b in combinations(model_names, 2):
    print(f"\n  {name_a} vs {name_b}:")

    # --- Paired t-test on monthly ICs ---
    ic_a = monthly_ics[name_a].set_index('Date')['IC']
    ic_b = monthly_ics[name_b].set_index('Date')['IC']
    common_dates_ic = ic_a.index.intersection(ic_b.index)
    ic_a_common = ic_a.loc[common_dates_ic].values
    ic_b_common = ic_b.loc[common_dates_ic].values

    if len(common_dates_ic) >= 3:
        t_ic, p_ic = stats.ttest_rel(ic_a_common, ic_b_common)
    else:
        t_ic, p_ic = np.nan, np.nan

    print(f"    IC paired t-test: t = {t_ic:.4f}, p = {p_ic:.4f}")

    # --- Diebold-Mariano test on monthly MSE ---
    mse_a = monthly_mse[name_a].set_index('Date')['MSE']
    mse_b = monthly_mse[name_b].set_index('Date')['MSE']
    common_dates_mse = mse_a.index.intersection(mse_b.index)
    mse_a_common = mse_a.loc[common_dates_mse].values
    mse_b_common = mse_b.loc[common_dates_mse].values

    d = mse_a_common - mse_b_common  # positive => A is worse
    n_dm = len(d)
    if n_dm >= 3 and np.std(d, ddof=1) > 0:
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n_dm))
        dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        dm_stat, dm_pval = np.nan, np.nan

    print(f"    Diebold-Mariano:  DM = {dm_stat:.4f}, p = {dm_pval:.4f}")

    # Determine better model: higher IC is better, lower MSE is better
    mean_ic_a = ic_a.loc[common_dates_ic].mean()
    mean_ic_b = ic_b.loc[common_dates_ic].mean()
    better = name_a if mean_ic_a > mean_ic_b else name_b
    print(f"    Better model (by IC): {better}")

    sig_rows.append({
        'Model_A': name_a,
        'Model_B': name_b,
        'IC_ttest_stat': t_ic,
        'IC_ttest_pval': p_ic,
        'DM_stat': dm_stat,
        'DM_pval': dm_pval,
        'Better_Model': better,
    })

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join('outputs', 'tables', 'model_significance_tests.csv'), index=False)
print(f"\nSaved model_significance_tests.csv ({len(sig_df)} pairs)")

# ============================================================
# 5. IC HISTOGRAM
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING IC HISTOGRAM")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 6))

for name in model_preds:
    ic_vals = monthly_ics[name]['IC'].values
    mean_ic = ic_vals.mean()
    ax.hist(ic_vals, bins=30, alpha=0.5, label=f'{name} (mean={mean_ic:.4f})',
            color=COLORS.get(name, 'gray'), edgecolor='white', linewidth=0.5)
    ax.axvline(mean_ic, color=COLORS.get(name, 'gray'), linestyle='--', linewidth=2)

ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
ax.set_xlabel('Monthly Spearman IC', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Monthly Spearman IC by Model', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'ic_histogram.png'), dpi=150)
print("Saved ic_histogram.png")
plt.close()

# ============================================================
# 6. IC BOX PLOT
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING IC BOX PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

box_data = []
box_labels = []
box_colors = []
for name in model_preds:
    box_data.append(monthly_ics[name]['IC'].values)
    box_labels.append(name)
    box_colors.append(COLORS.get(name, 'gray'))

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set_ylabel('Monthly Spearman IC', fontsize=12)
ax.set_title('Monthly IC Distribution by Model', fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'ic_boxplot.png'), dpi=150)
print("Saved ic_boxplot.png")
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nIC Information Ratios:")
print(ir_df[['Model', 'Mean_IC', 'IC_IR', 'IC_t_stat', 'Hit_Rate']].to_string(
    index=False, float_format='{:.4f}'.format))

print("\nSignificance Tests (p < 0.05 highlighted):")
for _, row in sig_df.iterrows():
    ic_sig = "*" if row['IC_ttest_pval'] < 0.05 else " "
    dm_sig = "*" if row['DM_pval'] < 0.05 else " "
    print(f"  {row['Model_A']:>16s} vs {row['Model_B']:<16s}  "
          f"IC t-test p={row['IC_ttest_pval']:.4f}{ic_sig}  "
          f"DM p={row['DM_pval']:.4f}{dm_sig}  "
          f"Better: {row['Better_Model']}")

print("\nDone!")
