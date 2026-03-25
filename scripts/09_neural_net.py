"""
09_neural_net.py
Shallow feedforward neural network for return prediction.
Tests the nonlinearity hypothesis: does the neural net outperform
linear models more in later periods (2020-2025) vs earlier (2010-2015)?

Input:  data/processed/panel_features.csv
Output: outputs/tables/neural_net_results.csv
        outputs/figures/nn_vs_linear_by_period.png
        outputs/figures/nn_rolling_ic.png
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from scipy import stats
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
PRED_DIR = os.path.join('data', 'predictions')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

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
# Neural net architectures to compare
# ============================================================
NN_CONFIGS = {
    'NN_32': (32,),                    # 1 hidden layer, 32 neurons
    'NN_64_32': (64, 32),              # 2 hidden layers
    'NN_128_64_32': (128, 64, 32),     # 3 hidden layers (deeper)
}

# ============================================================
# Expanding window prediction
# ============================================================
def run_model_expanding(df, dates, features, target, model_fn, label="Model"):
    """
    Expanding window: train on [0, t), predict at t.
    model_fn: function that returns a fresh model instance.
    Returns (ic_df, elapsed_time, stock_level_predictions).
    """
    ic_list = []
    stock_preds = []
    start = time.time()

    for i in range(MIN_TRAIN_MONTHS, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]

        train = df[df['Date'].isin(train_dates)][features + [target]].dropna()
        test = df[df['Date'] == test_date].dropna(subset=features + [target])

        if len(train) < 100 or len(test) < 20:
            continue

        X_train, y_train = train[features].values, train[target].values
        X_test, y_test = test[features].values, test[target].values

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ic = stats.spearmanr(y_pred, y_test)[0]
        ic_list.append({'Date': test_date, 'IC': ic})

        pred_df = test[['Date', 'Ticker']].copy()
        pred_df['predicted_return'] = y_pred
        pred_df['actual_return'] = y_test
        stock_preds.append(pred_df)

    elapsed = time.time() - start
    ic_df = pd.DataFrame(ic_list)
    preds_out = pd.concat(stock_preds, ignore_index=True) if stock_preds else pd.DataFrame()

    print(f"\n{label}:")
    print(f"  Mean IC: {ic_df['IC'].mean():.4f}")
    print(f"  IC Std: {ic_df['IC'].std():.4f}")
    print(f"  IC > 0 pct: {(ic_df['IC'] > 0).mean():.2%}")
    print(f"  IC t-stat: {ic_df['IC'].mean() / (ic_df['IC'].std() / np.sqrt(len(ic_df))):.3f}")
    print(f"  Time: {elapsed:.2f}s")

    return ic_df, elapsed, preds_out


# ============================================================
# Run all neural net configs + Ridge baseline
# ============================================================
print("=" * 70)
print("NEURAL NETWORK vs LINEAR MODEL COMPARISON")
print("=" * 70)

results = {}

# Ridge baseline
print("\nRunning Ridge baseline...")
ridge_ic, ridge_time, ridge_preds = run_model_expanding(
    df, dates, FEATURES, TARGET,
    model_fn=lambda: Ridge(alpha=0.01),
    label="Ridge (baseline)"
)
results['Ridge'] = {'ic_df': ridge_ic, 'time': ridge_time}

# Neural nets
nn_preds_dict = {}
for name, layers in NN_CONFIGS.items():
    print(f"\nRunning {name} {layers}...")
    nn_ic, nn_time, nn_preds = run_model_expanding(
        df, dates, FEATURES, TARGET,
        model_fn=lambda l=layers: MLPRegressor(
            hidden_layer_sizes=l,
            activation='relu',
            solver='adam',
            alpha=0.01,         # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ),
        label=f"Neural Net {name} {layers}"
    )
    results[name] = {'ic_df': nn_ic, 'time': nn_time}
    nn_preds_dict[name] = nn_preds

# Save NN_64_32 predictions (the primary architecture used in backtests)
if 'NN_64_32' in nn_preds_dict and len(nn_preds_dict['NN_64_32']) > 0:
    nn_pred_path = os.path.join(PRED_DIR, 'NN_64_32_predictions.csv')
    nn_preds_dict['NN_64_32'].to_csv(nn_pred_path, index=False)
    print(f"\nSaved NN_64_32 predictions ({len(nn_preds_dict['NN_64_32'])} rows) to {nn_pred_path}")

# ============================================================
# Nonlinearity over time analysis
# Split into early (2015-2019) vs late (2020-2025)
# ============================================================
print("\n" + "=" * 70)
print("NONLINEARITY HYPOTHESIS: EARLY vs LATE PERIOD")
print("=" * 70)

early_start = pd.Timestamp('2015-01-01')
early_end = pd.Timestamp('2019-12-31')
late_start = pd.Timestamp('2020-01-01')
late_end = pd.Timestamp('2025-12-31')

period_results = []

for name, res in results.items():
    ic_df = res['ic_df'].copy()
    ic_df['Date'] = pd.to_datetime(ic_df['Date'])
    
    early = ic_df[(ic_df['Date'] >= early_start) & (ic_df['Date'] <= early_end)]
    late = ic_df[(ic_df['Date'] >= late_start) & (ic_df['Date'] <= late_end)]
    
    early_ic = early['IC'].mean() if len(early) > 0 else np.nan
    late_ic = late['IC'].mean() if len(late) > 0 else np.nan
    diff = late_ic - early_ic
    
    print(f"\n{name}:")
    print(f"  Early (2015-2019) IC: {early_ic:.4f} ({len(early)} months)")
    print(f"  Late  (2020-2025) IC: {late_ic:.4f} ({len(late)} months)")
    print(f"  Difference: {diff:+.4f}")
    
    period_results.append({
        'Model': name,
        'Early_IC': early_ic,
        'Late_IC': late_ic,
        'Difference': diff
    })

period_df = pd.DataFrame(period_results)

# Key test: does NN advantage over Ridge grow in late period?
for name in NN_CONFIGS:
    ridge_early = period_df[period_df['Model'] == 'Ridge']['Early_IC'].values[0]
    ridge_late = period_df[period_df['Model'] == 'Ridge']['Late_IC'].values[0]
    nn_early = period_df[period_df['Model'] == name]['Early_IC'].values[0]
    nn_late = period_df[period_df['Model'] == name]['Late_IC'].values[0]
    
    nn_advantage_early = nn_early - ridge_early
    nn_advantage_late = nn_late - ridge_late
    
    print(f"\n{name} advantage over Ridge:")
    print(f"  Early: {nn_advantage_early:+.4f}")
    print(f"  Late:  {nn_advantage_late:+.4f}")
    print(f"  Growing nonlinearity: {'YES' if nn_advantage_late > nn_advantage_early else 'NO'}")

# ============================================================
# By regime analysis
# ============================================================
print("\n" + "=" * 70)
print("PERFORMANCE BY REGIME")
print("=" * 70)

regime_map = df[['Date', 'REGIME']].drop_duplicates()

for name, res in results.items():
    ic_df = res['ic_df'].copy()
    ic_df['Date'] = pd.to_datetime(ic_df['Date'])
    ic_df = ic_df.merge(regime_map, on='Date', how='left')
    
    print(f"\n{name}:")
    for regime in ['BULL', 'NEUTRAL', 'CRASH']:
        sub = ic_df[ic_df['REGIME'] == regime]
        if len(sub) > 0:
            print(f"  {regime:10s}: IC={sub['IC'].mean():.4f} ({len(sub)} months)")

# ============================================================
# Save results
# ============================================================
summary_rows = []
for name, res in results.items():
    ic_df = res['ic_df']
    summary_rows.append({
        'Model': name,
        'Mean_IC': ic_df['IC'].mean(),
        'IC_Std': ic_df['IC'].std(),
        'IC_Hit_Rate': (ic_df['IC'] > 0).mean(),
        'Time_sec': res['time']
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join('outputs', 'tables', 'neural_net_results.csv'), index=False)
period_df.to_csv(os.path.join('outputs', 'tables', 'nonlinearity_analysis.csv'), index=False)

# ============================================================
# Plot 1: Early vs Late IC comparison (grouped bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(period_df))
width = 0.35

bars1 = ax.bar(x - width/2, period_df['Early_IC'], width, label='2015-2019', color='#3498db')
bars2 = ax.bar(x + width/2, period_df['Late_IC'], width, label='2020-2025', color='#e74c3c')

ax.set_ylabel('Mean IC')
ax.set_title('Nonlinearity Hypothesis: Model Performance by Period')
ax.set_xticks(x)
ax.set_xticklabels(period_df['Model'])
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'nn_vs_linear_by_period.png'), dpi=150)
print("\nSaved nn_vs_linear_by_period.png")
plt.close()

# ============================================================
# Plot 2: Rolling IC comparison (Ridge vs best NN)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
window = 12

for name, res in results.items():
    ic_df = res['ic_df'].copy()
    ic_df['Date'] = pd.to_datetime(ic_df['Date'])
    rolling = ic_df.set_index('Date')['IC'].rolling(window).mean()
    ax.plot(rolling.index, rolling.values, label=name, linewidth=1.2)

ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('12-Month Rolling IC')
ax.set_title('Neural Net vs Ridge: Rolling Information Coefficient')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'nn_rolling_ic.png'), dpi=150)
print("Saved nn_rolling_ic.png")
plt.close()

# ============================================================
# Final full model comparison
# ============================================================
print("\n" + "=" * 70)
print("COMPLETE MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'Mean IC':<10} {'Hit Rate':<10} {'Time (s)':<10}")
print("-" * 55)
print(f"{'OLS (Fama-MacBeth)':<25} {0.0602:<10.4f} {0.7095:<10.2%} {0.24:<10.2f}")
print(f"{'Ridge':<25} {results['Ridge']['ic_df']['IC'].mean():<10.4f} {(results['Ridge']['ic_df']['IC'] > 0).mean():<10.2%} {results['Ridge']['time']:<10.2f}")
print(f"{'Lasso':<25} {0.0512:<10.4f} {0.6419:<10.2%} {3.45:<10.2f}")
print(f"{'Logistic':<25} {'0.5440':<10} {0.6959:<10.2%} {2.56:<10.2f}")
print(f"{'SGD (bs=256)':<25} {0.0587:<10.4f} {0.7027:<10.2%} {257.93:<10.2f}")
for name in NN_CONFIGS:
    ic_df = results[name]['ic_df']
    print(f"{('NN ' + name):<25} {ic_df['IC'].mean():<10.4f} {(ic_df['IC'] > 0).mean():<10.2%} {results[name]['time']:<10.2f}")

print("\nDone!")