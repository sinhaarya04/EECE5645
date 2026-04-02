"""
12_factor_decay.py
Factor decay analysis: how quickly does predictive alpha disappear?
After ranking stocks at time t, track returns at t+1, t+2, t+3, ..., t+12.
Compare decay speed in early (2015-2019) vs late (2020-2025) periods.

Input:  data/processed/panel_features.csv
Output: outputs/tables/factor_decay_results.csv
        outputs/figures/factor_decay_curves.png
        outputs/figures/factor_decay_early_vs_late.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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
HOLDING_PERIODS = [1, 2, 3, 4, 5, 6, 9, 12]  # months ahead

# ============================================================
# Compute forward returns at multiple horizons
# ============================================================
print("Computing multi-horizon forward returns...")
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

for h in HOLDING_PERIODS:
    col_name = f'FWD_RET_{h}M'
    df[col_name] = (
        df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS']
        .pct_change(periods=h)
        .shift(-h)
        .values
    )

print(f"Added forward returns for horizons: {HOLDING_PERIODS}")

# ============================================================
# Models to test
# ============================================================
MODELS = {
    'Ridge': lambda: Ridge(alpha=0.01),
    'NN_64_32': lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        alpha=0.01, learning_rate='adaptive', learning_rate_init=0.001,
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42
    ),
}

# ============================================================
# Decay analysis: predict at time t, measure IC at t+1, t+2, ...
# ============================================================
def compute_decay(df, dates, features, model_fn, label):
    """
    At each month t:
    1. Train model on data up to t
    2. Rank stocks by predicted 1-month return
    3. Measure rank correlation with ACTUAL returns at t+1, t+2, ..., t+12
    
    If the signal decays, the IC at t+6 will be lower than at t+1.
    """
    print(f"\n  Running {label}...")
    start = time.time()
    
    # Store IC for each holding period
    decay_ics = {h: [] for h in HOLDING_PERIODS}
    
    for i in range(MIN_TRAIN_MONTHS, len(dates) - max(HOLDING_PERIODS) - 1):
        train_dates = dates[:i]
        test_date = dates[i]
        
        train = df[df['Date'].isin(train_dates)][features + [TARGET]].dropna()
        test = df[df['Date'] == test_date].dropna(subset=features)
        
        if len(train) < 100 or len(test) < 20:
            continue
        
        X_train, y_train = train[features].values, train[TARGET].values
        X_test = test[features].values
        
        model = model_fn()
        model.fit(X_train, y_train)
        predicted_rank = model.predict(X_test)
        
        # Now check: does this ranking still predict returns at different horizons?
        for h in HOLDING_PERIODS:
            fwd_col = f'FWD_RET_{h}M'
            actual = test[fwd_col].values
            
            # Drop NaN pairs
            valid = ~np.isnan(actual)
            if valid.sum() < 20:
                continue
            
            ic = stats.spearmanr(predicted_rank[valid], actual[valid])[0]
            decay_ics[h].append({'Date': test_date, 'IC': ic})
    
    elapsed = time.time() - start
    print(f"  {label} completed in {elapsed:.1f}s")
    
    # Average IC per holding period
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
    
    return decay_summary


print("=" * 70)
print("FACTOR DECAY ANALYSIS")
print("=" * 70)

all_decay = {}
for name, model_fn in MODELS.items():
    all_decay[name] = compute_decay(df, dates, FEATURES, model_fn, name)

# ============================================================
# Print decay curves
# ============================================================
print("\n" + "=" * 70)
print("DECAY CURVES (Mean IC by Holding Period)")
print("=" * 70)

decay_table = []
for name in MODELS:
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

for name in MODELS:
    print(f"\n{name}:")
    print(f"  {'Horizon':<10} {'Early IC':<12} {'Late IC':<12} {'Faster Decay?'}")
    print(f"  {'-'*46}")
    
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
            # Decay = how much IC dropped relative to 1-month IC
            early_decay = (early_ic / early_base) if early_base != 0 else 0
            late_decay = (late_ic / late_base) if late_base != 0 else 0
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

for name in MODELS:
    ics = [all_decay[name][h]['mean_ic'] for h in HOLDING_PERIODS]
    base_ic = ics[0]  # 1-month IC
    half_ic = base_ic / 2
    
    half_life = None
    for j, (h, ic) in enumerate(zip(HOLDING_PERIODS, ics)):
        if ic <= half_ic:
            # Interpolate
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
colors = {'Ridge': '#2ecc71', 'NN_64_32': '#9b59b6'}

for name in MODELS:
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
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, name in zip(axes, MODELS):
    early_data = period_decay_df[period_decay_df['Model'] == name]
    
    ax.plot(early_data['Holding_Period'], early_data['Early_IC'],
            'o-', label='2015-2019', linewidth=2, markersize=8, color='#3498db')
    ax.plot(early_data['Holding_Period'], early_data['Late_IC'],
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