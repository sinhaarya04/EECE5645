"""
05_ols_fama_macbeth.py
Runs rolling Fama-MacBeth cross-sectional regressions.
At each time step, regress forward returns on factor exposures across all stocks.
Then average the coefficients over time and test significance.
Also splits results by regime.

Input:  data/processed/panel_features.csv
Output: outputs/tables/fama_macbeth_results.csv
        outputs/figures/factor_premia.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import time

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
TABLE_OUT = os.path.join('outputs', 'tables', 'fama_macbeth_results.csv')
FIG_OUT = os.path.join('outputs', 'figures', 'factor_premia.png')
PRED_DIR = os.path.join('data', 'predictions')
PRED_OUT = os.path.join(PRED_DIR, 'OLS_FamaMacBeth_predictions.csv')

os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Features to use in regression
FEATURES = [
    'PE_RATIO', 'PX_TO_SALES_RATIO', 'CURRENT_EV_TO_T12M_EBITDA',
    'PX_TO_FREE_CASH_FLOW', 'EQY_DVD_YLD_12M', 'CUR_MKT_CAP',
    'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'GROSS_MARGIN', 'OPER_MARGIN',
    'TOT_DEBT_TO_TOT_EQY', 'CUR_RATIO', 'BETA_RAW_OVERRIDABLE',
    'EARN_YLD', 'VOLATILITY_90D', 'VOLUME_AVG_30D',
    'MOMENTUM_12_1', 'REVERSAL_3M', 'HIGH_52W_RATIO'
]

TARGET = 'FWD_RETURN'

# ============================================================
# Fama-MacBeth: cross-sectional regression at each date
# ============================================================
def run_fama_macbeth(data, features, target):
    """
    For each date, run OLS: R_i = alpha + beta_1*X_1 + ... + beta_k*X_k + e_i
    Returns DataFrame of coefficients per date.
    """
    dates = sorted(data['Date'].unique())
    coef_list = []

    for date in dates:
        cross = data[data['Date'] == date][features + [target]].dropna()
        if len(cross) < 30:  # need enough stocks for regression
            continue

        X = cross[features].values
        y = cross[target].values

        # Add intercept
        X_int = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
            row = {'Date': date, 'intercept': beta[0]}
            for i, f in enumerate(features):
                row[f] = beta[i + 1]
            coef_list.append(row)
        except np.linalg.LinAlgError:
            continue

    return pd.DataFrame(coef_list)


print("Running Fama-MacBeth regressions...")
start = time.time()
coefs = run_fama_macbeth(df, FEATURES, TARGET)
elapsed = time.time() - start
print(f"Completed in {elapsed:.2f} seconds")
print(f"Number of cross-sections: {len(coefs)}")

# ============================================================
# Analyze: time-series average of coefficients + t-stats
# ============================================================
def summarize_coefs(coefs_df, features, label="ALL"):
    """Compute mean coefficient, t-stat, and significance for each factor."""
    results = []
    for f in features:
        series = coefs_df[f].dropna()
        mean = series.mean()
        se = series.std() / np.sqrt(len(series))
        t_stat = mean / se if se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(series) - 1))
        results.append({
            'Factor': f,
            'Regime': label,
            'Mean_Coef': round(mean, 6),
            'Std': round(series.std(), 6),
            't_stat': round(t_stat, 3),
            'p_value': round(p_val, 4),
            'Significant': '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
        })
    return pd.DataFrame(results)


# Overall results
print("\n" + "=" * 70)
print("OVERALL FAMA-MACBETH RESULTS")
print("=" * 70)
results_all = summarize_coefs(coefs, FEATURES, "ALL")
print(results_all.to_string(index=False))

# By regime
coefs['Date'] = pd.to_datetime(coefs['Date'])
regime_map = df[['Date', 'REGIME']].drop_duplicates()
coefs = coefs.merge(regime_map, on='Date', how='left')

results_by_regime = []
for regime in ['BULL', 'NEUTRAL', 'CRASH']:
    sub = coefs[coefs['REGIME'] == regime]
    if len(sub) > 5:
        print(f"\n{'=' * 70}")
        print(f"REGIME: {regime} ({len(sub)} months)")
        print("=" * 70)
        res = summarize_coefs(sub, FEATURES, regime)
        print(res.to_string(index=False))
        results_by_regime.append(res)

# Combine and save
all_results = pd.concat([results_all] + results_by_regime, ignore_index=True)
all_results.to_csv(TABLE_OUT, index=False)
print(f"\nSaved results to {TABLE_OUT}")

# ============================================================
# Plot: factor premia bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
sig_mask = results_all['p_value'] < 0.05
colors = ['#2ecc71' if s else '#95a5a6' for s in sig_mask]

bars = ax.bar(results_all['Factor'], results_all['Mean_Coef'], color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Mean Monthly Coefficient')
ax.set_title('Fama-MacBeth Factor Premia (Green = Significant at 5%)')
ax.set_xticklabels(results_all['Factor'], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
print(f"Saved figure to {FIG_OUT}")
plt.close()

# ============================================================
# Predictive performance: out-of-sample R² and IC
# ============================================================
print("\n" + "=" * 70)
print("PREDICTIVE METRICS")
print("=" * 70)

# Information Coefficient: rank correlation between predicted and realized returns
# Use expanding window: train on all data up to month t, predict month t+1
dates = sorted(df['Date'].unique())
ic_list = []
all_preds = []

for i in range(60, len(dates) - 1):  # start after 60 months of data
    train_dates = dates[:i]
    test_date = dates[i]

    train = df[df['Date'].isin(train_dates)]
    test = df[df['Date'] == test_date].dropna(subset=FEATURES + [TARGET])

    if len(test) < 30:
        continue

    # Compute mean coefficients from training period
    train_coefs = run_fama_macbeth(train, FEATURES, TARGET)
    if len(train_coefs) == 0:
        continue

    mean_betas = train_coefs[FEATURES].mean().values

    # Predict
    X_test = test[FEATURES].values
    y_pred = X_test @ mean_betas
    y_true = test[TARGET].values

    # Rank IC (Spearman correlation)
    ic = stats.spearmanr(y_pred, y_true)[0]
    ic_list.append({'Date': test_date, 'IC': ic})

    # Save stock-level predictions
    pred_df = test[['Date', 'Ticker']].copy()
    pred_df['predicted_return'] = y_pred
    pred_df['actual_return'] = y_true
    all_preds.append(pred_df)

ic_df = pd.DataFrame(ic_list)
print(f"Mean IC: {ic_df['IC'].mean():.4f}")
print(f"IC Std: {ic_df['IC'].std():.4f}")
print(f"IC > 0 pct: {(ic_df['IC'] > 0).mean():.2%}")
print(f"IC t-stat: {ic_df['IC'].mean() / (ic_df['IC'].std() / np.sqrt(len(ic_df))):.3f}")

# Save predictions to disk
preds_all = pd.concat(all_preds, ignore_index=True)
preds_all.to_csv(PRED_OUT, index=False)
print(f"Saved {len(preds_all)} stock-level predictions to {PRED_OUT}")