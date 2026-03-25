"""
06_ridge_lasso.py
Ridge (L2) and Lasso (L1) regression with expanding-window cross-validation.
Respects temporal ordering — no future data leakage.
Compares factor selection, predictive performance, and portfolio metrics.

Input:  data/processed/panel_features.csv
Output: outputs/tables/regularized_results.csv
        outputs/figures/lasso_coefficients.png
        outputs/figures/ridge_vs_lasso_ic.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
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
MIN_TRAIN_MONTHS = 60  # 5 years minimum training window

# Lambda values to search over
ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

# ============================================================
# Expanding-window CV to select best alpha
# ============================================================
def expanding_window_cv(df, dates, model_class, alphas, features, target, n_val=12):
    """
    For each alpha, run expanding-window validation:
    Train on months [0, t-n_val), validate on [t-n_val, t).
    Returns best alpha based on average validation IC.
    """
    results = {a: [] for a in alphas}
    
    # Use the last n_val months before test period as validation
    val_start = MIN_TRAIN_MONTHS
    val_end = val_start + n_val
    
    for alpha in alphas:
        for t in range(val_start, min(val_end + 12, len(dates) - 1)):
            train_dates = dates[:t]
            val_date = dates[t]
            
            train = df[df['Date'].isin(train_dates)][features + [target]].dropna()
            val = df[df['Date'] == val_date][features + [target]].dropna()
            
            if len(train) < 100 or len(val) < 20:
                continue
            
            X_train, y_train = train[features].values, train[target].values
            X_val, y_val = val[features].values, val[target].values
            
            model = model_class(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            ic = stats.spearmanr(y_pred, y_val)[0]
            results[alpha].append(ic)
    
    # Average IC per alpha
    avg_ic = {a: np.mean(ics) if ics else -999 for a, ics in results.items()}
    best_alpha = max(avg_ic, key=avg_ic.get)
    
    return best_alpha, avg_ic


# ============================================================
# Out-of-sample prediction with expanding window
# ============================================================
def run_expanding_window(df, dates, model_class, alpha, features, target):
    """
    Train on all data up to month t, predict month t+1.
    Returns ICs, coefficients, and stock-level predictions.
    """
    ic_list = []
    stock_preds = []
    coef_history = []

    for i in range(MIN_TRAIN_MONTHS, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]

        train = df[df['Date'].isin(train_dates)][features + [target]].dropna()
        test = df[df['Date'] == test_date].dropna(subset=features + [target])

        if len(train) < 100 or len(test) < 20:
            continue

        X_train, y_train = train[features].values, train[target].values
        X_test, y_test = test[features].values, test[target].values

        model = model_class(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Rank IC
        ic = stats.spearmanr(y_pred, y_test)[0]
        ic_list.append({'Date': test_date, 'IC': ic})

        # Store coefficients
        coef_history.append(dict(zip(features, model.coef_)))

        # Store stock-level predictions
        pred_df = test[['Date', 'Ticker']].copy()
        pred_df['predicted_return'] = y_pred
        pred_df['actual_return'] = y_test
        stock_preds.append(pred_df)

    return pd.DataFrame(ic_list), pd.DataFrame(coef_history), pd.concat(stock_preds, ignore_index=True)


# ============================================================
# Run Ridge
# ============================================================
print("\n" + "=" * 70)
print("RIDGE REGRESSION (L2)")
print("=" * 70)

print("Selecting best alpha via expanding-window CV...")
start = time.time()
best_ridge_alpha, ridge_cv_scores = expanding_window_cv(
    df, dates, Ridge, ALPHAS, FEATURES, TARGET
)
print(f"Best Ridge alpha: {best_ridge_alpha}")
print(f"CV scores: {{{', '.join(f'{a}: {s:.4f}' for a, s in sorted(ridge_cv_scores.items()))}}}")

print(f"\nRunning out-of-sample predictions with alpha={best_ridge_alpha}...")
ridge_ic, ridge_coefs, ridge_preds = run_expanding_window(
    df, dates, Ridge, best_ridge_alpha, FEATURES, TARGET
)
ridge_time = time.time() - start

print(f"Completed in {ridge_time:.2f} seconds")
print(f"Mean IC: {ridge_ic['IC'].mean():.4f}")
print(f"IC Std: {ridge_ic['IC'].std():.4f}")
print(f"IC > 0 pct: {(ridge_ic['IC'] > 0).mean():.2%}")
print(f"IC t-stat: {ridge_ic['IC'].mean() / (ridge_ic['IC'].std() / np.sqrt(len(ridge_ic))):.3f}")

# ============================================================
# Run Lasso
# ============================================================
print("\n" + "=" * 70)
print("LASSO REGRESSION (L1)")
print("=" * 70)

print("Selecting best alpha via expanding-window CV...")
start = time.time()
best_lasso_alpha, lasso_cv_scores = expanding_window_cv(
    df, dates, Lasso, ALPHAS, FEATURES, TARGET
)
print(f"Best Lasso alpha: {best_lasso_alpha}")
print(f"CV scores: {{{', '.join(f'{a}: {s:.4f}' for a, s in sorted(lasso_cv_scores.items()))}}}")

print(f"\nRunning out-of-sample predictions with alpha={best_lasso_alpha}...")
lasso_ic, lasso_coefs, lasso_preds = run_expanding_window(
    df, dates, Lasso, best_lasso_alpha, FEATURES, TARGET
)
lasso_time = time.time() - start

print(f"Completed in {lasso_time:.2f} seconds")
print(f"Mean IC: {lasso_ic['IC'].mean():.4f}")
print(f"IC Std: {lasso_ic['IC'].std():.4f}")
print(f"IC > 0 pct: {(lasso_ic['IC'] > 0).mean():.2%}")
print(f"IC t-stat: {lasso_ic['IC'].mean() / (lasso_ic['IC'].std() / np.sqrt(len(lasso_ic))):.3f}")

# ============================================================
# Lasso factor selection — which factors survived?
# ============================================================
print("\n" + "=" * 70)
print("LASSO FACTOR SELECTION (Final Model)")
print("=" * 70)

final_coefs = lasso_coefs.iloc[-1]  # last model's coefficients
for f in FEATURES:
    coef = final_coefs[f]
    status = "KEPT" if abs(coef) > 1e-6 else "DROPPED"
    print(f"  {f:35s} {coef:+.6f}  [{status}]")

kept = sum(1 for f in FEATURES if abs(final_coefs[f]) > 1e-6)
dropped = len(FEATURES) - kept
print(f"\nLasso kept {kept}/{len(FEATURES)} factors, dropped {dropped}")

# ============================================================
# Comparison table
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# Load OLS FM predictions to compute IC metrics (instead of hardcoding)
fm_pred_path = os.path.join(PRED_DIR, 'OLS_FamaMacBeth_predictions.csv')
if os.path.exists(fm_pred_path):
    fm_preds = pd.read_csv(fm_pred_path)
    fm_preds['Date'] = pd.to_datetime(fm_preds['Date'])
    fm_ic_list = []
    for date, grp in fm_preds.groupby('Date'):
        ic = stats.spearmanr(grp['predicted_return'], grp['actual_return'])[0]
        fm_ic_list.append(ic)
    fm_mean_ic = np.mean(fm_ic_list)
    fm_ic_std = np.std(fm_ic_list)
    fm_hit_rate = np.mean([ic > 0 for ic in fm_ic_list])
else:
    print(f"WARNING: {fm_pred_path} not found. Run script 05 first. Using NaN.")
    fm_mean_ic, fm_ic_std, fm_hit_rate = np.nan, np.nan, np.nan

comparison = pd.DataFrame({
    'Model': ['OLS (Fama-MacBeth)', 'Ridge', 'Lasso'],
    'Mean_IC': [fm_mean_ic, ridge_ic['IC'].mean(), lasso_ic['IC'].mean()],
    'IC_Std': [fm_ic_std, ridge_ic['IC'].std(), lasso_ic['IC'].std()],
    'IC_Hit_Rate': [fm_hit_rate, (ridge_ic['IC'] > 0).mean(), (lasso_ic['IC'] > 0).mean()],
    'Time_sec': [np.nan, ridge_time, lasso_time]
})
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join('outputs', 'tables', 'regularized_results.csv'), index=False)

# Save stock-level predictions
ridge_preds.to_csv(os.path.join(PRED_DIR, 'Ridge_predictions.csv'), index=False)
lasso_preds.to_csv(os.path.join(PRED_DIR, 'Lasso_predictions.csv'), index=False)
print(f"Saved Ridge predictions ({len(ridge_preds)} rows) to {PRED_DIR}/Ridge_predictions.csv")
print(f"Saved Lasso predictions ({len(lasso_preds)} rows) to {PRED_DIR}/Lasso_predictions.csv")

# ============================================================
# Plot 1: Lasso coefficient path over time
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
for f in FEATURES:
    vals = lasso_coefs[f].values
    if np.any(np.abs(vals) > 1e-6):  # only plot non-zero factors
        ax.plot(vals, label=f, linewidth=1)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Month (expanding window)')
ax.set_ylabel('Lasso Coefficient')
ax.set_title('Lasso Coefficient Evolution Over Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'lasso_coefficients.png'), dpi=150)
print("Saved lasso_coefficients.png")
plt.close()

# ============================================================
# Plot 2: Rolling IC comparison (OLS vs Ridge vs Lasso)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
window = 12  # 12-month rolling IC

ridge_ic['IC_rolling'] = ridge_ic['IC'].rolling(window).mean()
lasso_ic['IC_rolling'] = lasso_ic['IC'].rolling(window).mean()

ax.plot(ridge_ic['Date'], ridge_ic['IC_rolling'], label='Ridge', linewidth=1.5)
ax.plot(lasso_ic['Date'], lasso_ic['IC_rolling'], label='Lasso', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('12-Month Rolling IC')
ax.set_title('Ridge vs Lasso: Rolling Information Coefficient')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'ridge_vs_lasso_ic.png'), dpi=150)
print("Saved ridge_vs_lasso_ic.png")
plt.close()

print("\nDone!")