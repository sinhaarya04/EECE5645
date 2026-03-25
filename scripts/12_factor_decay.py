"""
12_factor_decay.py
Factor decay analysis: how quickly does predictive alpha disappear?
After ranking stocks at time t, track returns at t+1, t+2, t+3, ..., t+12.
Compare decay speed in early (2015-2019) vs late (2020-2025) periods.

Loads pre-computed predictions from data/predictions/ instead of retraining.

Requires: scripts 05, 06, 09 already run.

Input:  data/predictions/*.csv, data/processed/panel_features.csv
Output: outputs/tables/factor_decay_results.csv
        outputs/tables/factor_decay_early_vs_late.csv
        outputs/figures/factor_decay_curves.png
        outputs/figures/factor_decay_early_vs_late.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
PRED_DIR = os.path.join('data', 'predictions')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

HOLDING_PERIODS = [1, 2, 3, 4, 5, 6, 9, 12]

MODEL_FILES = {
    'OLS_FamaMacBeth': 'OLS_FamaMacBeth_predictions.csv',
    'Ridge': 'Ridge_predictions.csv',
    'Lasso': 'Lasso_predictions.csv',
    'NN_64_32': 'NN_64_32_predictions.csv',
}

# ============================================================
# Load panel data for multi-horizon forward returns
# ============================================================
print("Loading panel data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

print("Computing multi-horizon forward returns...")
for h in HOLDING_PERIODS:
    col_name = f'FWD_RET_{h}M'
    df[col_name] = (
        df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS']
        .pct_change(periods=h)
        .shift(-h)
        .values
    )
print(f"Added forward returns for horizons: {HOLDING_PERIODS}")

# Keep only columns needed for merging
fwd_cols = [f'FWD_RET_{h}M' for h in HOLDING_PERIODS]
panel_for_merge = df[['Date', 'Ticker'] + fwd_cols].copy()

# ============================================================
# Load predictions
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
# Decay analysis: rank by predicted return, measure IC at each horizon
# ============================================================
def compute_decay(preds_df, panel_df, label):
    """
    For each month in the predictions:
    1. Use predicted_return to rank stocks
    2. Merge with panel to get multi-horizon forward returns
    3. Compute Spearman IC between ranking and actual returns at each horizon
    """
    print(f"\n  Computing decay for {label}...")

    # Merge predictions with multi-horizon returns
    merged = preds_df.merge(panel_df, on=['Date', 'Ticker'], how='left')

    decay_ics = {h: [] for h in HOLDING_PERIODS}

    for date, group in merged.groupby('Date'):
        if len(group) < 20:
            continue

        predicted = group['predicted_return'].values

        for h in HOLDING_PERIODS:
            fwd_col = f'FWD_RET_{h}M'
            actual = group[fwd_col].values

            valid = ~np.isnan(actual)
            if valid.sum() < 20:
                continue

            ic = stats.spearmanr(predicted[valid], actual[valid])[0]
            decay_ics[h].append({'Date': date, 'IC': ic})

    # Summarize
    decay_summary = {}
    for h in HOLDING_PERIODS:
        if decay_ics[h]:
            ic_df = pd.DataFrame(decay_ics[h])
            decay_summary[h] = {
                'mean_ic': ic_df['IC'].mean(),
                'std_ic': ic_df['IC'].std(),
                'ic_df': ic_df
            }
        else:
            decay_summary[h] = {'mean_ic': 0, 'std_ic': 0, 'ic_df': pd.DataFrame()}

    print(f"  {label} done.")
    return decay_summary


print("\n" + "=" * 70)
print("FACTOR DECAY ANALYSIS")
print("=" * 70)

all_decay = {}
for name, preds in model_preds.items():
    all_decay[name] = compute_decay(preds, panel_for_merge, name)

# ============================================================
# Print decay curves
# ============================================================
print("\n" + "=" * 70)
print("DECAY CURVES (Mean IC by Holding Period)")
print("=" * 70)

decay_table = []
for name in model_preds:
    print(f"\n{name}:")
    for h in HOLDING_PERIODS:
        mean_ic = all_decay[name][h]['mean_ic']
        print(f"  {h:2d}-month: IC = {mean_ic:.4f}")
        decay_table.append({
            'Model': name,
            'Holding_Period': h,
            'Mean_IC': mean_ic,
            'IC_Std': all_decay[name][h]['std_ic']
        })

decay_df = pd.DataFrame(decay_table)

# ============================================================
# Early vs Late decay comparison
# ============================================================
print("\n" + "=" * 70)
print("DECAY: EARLY (2015-2019) vs LATE (2020-2025)")
print("=" * 70)

early_start = pd.Timestamp('2015-01-01')
early_end = pd.Timestamp('2019-12-31')
late_start = pd.Timestamp('2020-01-01')
late_end = pd.Timestamp('2025-12-31')

period_decay = []

for name in model_preds:
    print(f"\n{name}:")
    print(f"  {'Horizon':<10} {'Early IC':<12} {'Late IC':<12} {'Faster Decay?'}")
    print(f"  {'-'*46}")

    early_base = None
    late_base = None

    for h in HOLDING_PERIODS:
        ic_df = all_decay[name][h]['ic_df']
        if len(ic_df) == 0:
            continue

        ic_df['Date'] = pd.to_datetime(ic_df['Date'])
        early = ic_df[(ic_df['Date'] >= early_start) & (ic_df['Date'] <= early_end)]
        late = ic_df[(ic_df['Date'] >= late_start) & (ic_df['Date'] <= late_end)]

        early_ic = early['IC'].mean() if len(early) > 0 else np.nan
        late_ic = late['IC'].mean() if len(late) > 0 else np.nan

        print(f"  {h:2d}-month   {early_ic:+.4f}      {late_ic:+.4f}      ", end='')

        if h == 1:
            early_base = early_ic
            late_base = late_ic
            print("")
        else:
            early_decay = (early_ic / early_base) if early_base and early_base != 0 else 0
            late_decay = (late_ic / late_base) if late_base and late_base != 0 else 0
            faster = "YES" if late_decay < early_decay else "NO"
            print(f"  (Early retain: {early_decay:.1%}, Late retain: {late_decay:.1%}) [{faster}]")

        period_decay.append({
            'Model': name, 'Holding_Period': h,
            'Early_IC': early_ic, 'Late_IC': late_ic
        })

period_decay_df = pd.DataFrame(period_decay)

# ============================================================
# Half-life calculation
# ============================================================
print("\n" + "=" * 70)
print("SIGNAL HALF-LIFE (months until IC drops to 50% of 1-month IC)")
print("=" * 70)

for name in model_preds:
    ics = [all_decay[name][h]['mean_ic'] for h in HOLDING_PERIODS]
    base_ic = ics[0]
    half_ic = base_ic / 2

    half_life = None
    for j, (h, ic) in enumerate(zip(HOLDING_PERIODS, ics)):
        if ic <= half_ic:
            if j > 0:
                prev_h = HOLDING_PERIODS[j-1]
                prev_ic = ics[j-1]
                if prev_ic != ic:
                    half_life = prev_h + (half_ic - prev_ic) / (ic - prev_ic) * (h - prev_h)
                else:
                    half_life = h
            else:
                half_life = h
            break

    if half_life:
        print(f"  {name}: ~{half_life:.1f} months")
    else:
        print(f"  {name}: > {HOLDING_PERIODS[-1]} months (signal persists)")

# ============================================================
# Save results
# ============================================================
decay_df.to_csv(os.path.join('outputs', 'tables', 'factor_decay_results.csv'), index=False)
period_decay_df.to_csv(os.path.join('outputs', 'tables', 'factor_decay_early_vs_late.csv'), index=False)

# ============================================================
# Plot 1: Decay curves for all models
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'OLS_FamaMacBeth': '#3498db', 'Ridge': '#2ecc71',
          'Lasso': '#e74c3c', 'NN_64_32': '#9b59b6'}

for name in model_preds:
    ics = [all_decay[name][h]['mean_ic'] for h in HOLDING_PERIODS]
    ax.plot(HOLDING_PERIODS, ics, 'o-', label=name, linewidth=2,
            markersize=8, color=colors.get(name, 'gray'))

ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Holding Period (months)', fontsize=12)
ax.set_ylabel('Mean IC', fontsize=12)
ax.set_title('Factor Signal Decay: IC by Holding Period', fontsize=14)
ax.set_xticks(HOLDING_PERIODS)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'factor_decay_curves.png'), dpi=150)
print("\nSaved factor_decay_curves.png")
plt.close()

# ============================================================
# Plot 2: Early vs Late decay comparison
# ============================================================
n_models = len(model_preds)
fig, axes = plt.subplots(1, min(n_models, 4), figsize=(6 * min(n_models, 4), 6))
if n_models == 1:
    axes = [axes]

for ax, name in zip(axes, model_preds):
    model_data = period_decay_df[period_decay_df['Model'] == name]

    ax.plot(model_data['Holding_Period'], model_data['Early_IC'],
            'o-', label='2015-2019', linewidth=2, markersize=8, color='#3498db')
    ax.plot(model_data['Holding_Period'], model_data['Late_IC'],
            's-', label='2020-2025', linewidth=2, markersize=8, color='#e74c3c')

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Holding Period (months)', fontsize=11)
    ax.set_ylabel('Mean IC', fontsize=11)
    ax.set_title(f'{name}: Signal Decay by Period', fontsize=13)
    ax.set_xticks(HOLDING_PERIODS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'factor_decay_early_vs_late.png'), dpi=150)
print("Saved factor_decay_early_vs_late.png")
plt.close()

print("\nDone!")
