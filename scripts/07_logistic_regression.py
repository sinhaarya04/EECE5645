"""
07_logistic_regression.py
Classification approach: predict top vs bottom quintile of returns.
Uses expanding-window evaluation with ROC/AUC metrics.

Input:  data/processed/panel_features.csv
Output: outputs/tables/logistic_results.csv
        outputs/figures/roc_curves.png
        outputs/figures/classification_comparison.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
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

# ============================================================
# Create binary target: top quintile (1) vs bottom quintile (0)
# Middle 60% is excluded from training/testing
# ============================================================
def create_quintile_labels(group):
    """For each month, label top 20% as 1, bottom 20% as 0, middle as NaN."""
    q80 = group[TARGET].quantile(0.80)
    q20 = group[TARGET].quantile(0.20)
    group['LABEL'] = np.where(group[TARGET] >= q80, 1,
                     np.where(group[TARGET] <= q20, 0, np.nan))
    return group

print("Creating quintile labels...")
df = df.groupby('Date', group_keys=False).apply(create_quintile_labels)
df_labeled = df.dropna(subset=['LABEL'])
df_labeled['LABEL'] = df_labeled['LABEL'].astype(int)
print(f"Labeled samples: {len(df_labeled)} ({len(df_labeled[df_labeled['LABEL']==1])} top, {len(df_labeled[df_labeled['LABEL']==0])} bottom)")

# ============================================================
# Expanding-window logistic regression
# ============================================================
def run_logistic(df, dates, features, C_val=1.0):
    """
    Expanding window logistic regression.
    Returns AUC, accuracy, and predicted probabilities per month.
    """
    results = []
    all_y_true = []
    all_y_prob = []
    
    for i in range(MIN_TRAIN_MONTHS, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]
        
        train = df[df['Date'].isin(train_dates)][features + ['LABEL']].dropna()
        test = df[df['Date'] == test_date][features + ['LABEL']].dropna()
        
        if len(train) < 100 or len(test) < 10:
            continue
        
        X_train, y_train = train[features].values, train['LABEL'].values
        X_test, y_test = test[features].values, test['LABEL'].values
        
        # Need both classes in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        model = LogisticRegression(C=C_val, max_iter=10000, solver='lbfgs')
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Date': test_date,
            'AUC': auc,
            'Accuracy': acc,
            'N_test': len(y_test)
        })
        
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
    
    return pd.DataFrame(results), np.array(all_y_true), np.array(all_y_prob)


# ============================================================
# Run with different regularization strengths
# ============================================================
C_VALUES = [0.01, 0.1, 1.0, 10.0]

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION — EXPANDING WINDOW")
print("=" * 70)

best_auc = -1
best_C = None
all_results = {}

for C in C_VALUES:
    print(f"\nC = {C}:")
    start = time.time()
    res, y_true, y_prob = run_logistic(df_labeled, dates, FEATURES, C_val=C)
    elapsed = time.time() - start
    
    mean_auc = res['AUC'].mean()
    mean_acc = res['Accuracy'].mean()
    
    print(f"  Mean AUC: {mean_auc:.4f}")
    print(f"  Mean Accuracy: {mean_acc:.4f}")
    print(f"  AUC > 0.5 pct: {(res['AUC'] > 0.5).mean():.2%}")
    print(f"  Time: {elapsed:.2f}s")
    
    all_results[C] = {'res': res, 'y_true': y_true, 'y_prob': y_prob,
                      'mean_auc': mean_auc, 'mean_acc': mean_acc}
    
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_C = C

print(f"\nBest C: {best_C} (AUC: {best_auc:.4f})")

# ============================================================
# Detailed results for best model
# ============================================================
best = all_results[best_C]
best_res = best['res']

print("\n" + "=" * 70)
print(f"BEST MODEL (C={best_C}) DETAILED RESULTS")
print("=" * 70)
print(f"Mean AUC: {best_res['AUC'].mean():.4f}")
print(f"Mean Accuracy: {best_res['Accuracy'].mean():.4f}")
print(f"AUC Std: {best_res['AUC'].std():.4f}")
print(f"AUC > 0.5 pct: {(best_res['AUC'] > 0.5).mean():.2%}")
print(f"Months evaluated: {len(best_res)}")

# By regime
best_res['Date'] = pd.to_datetime(best_res['Date'])
regime_map = df[['Date', 'REGIME']].drop_duplicates()
best_res = best_res.merge(regime_map, on='Date', how='left')

print("\nAUC by Regime:")
for regime in ['BULL', 'NEUTRAL', 'CRASH']:
    sub = best_res[best_res['REGIME'] == regime]
    if len(sub) > 0:
        print(f"  {regime:10s}: AUC={sub['AUC'].mean():.4f}, Acc={sub['Accuracy'].mean():.4f} ({len(sub)} months)")

# Save results
summary = pd.DataFrame([{
    'Model': f'Logistic (C={C})',
    'Mean_AUC': all_results[C]['mean_auc'],
    'Mean_Accuracy': all_results[C]['mean_acc'],
} for C in C_VALUES])
summary.to_csv(os.path.join('outputs', 'tables', 'logistic_results.csv'), index=False)

# ============================================================
# Plot 1: ROC curve (aggregated)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

for C in C_VALUES:
    y_true = all_results[C]['y_true']
    y_prob = all_results[C]['y_prob']
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f'C={C} (AUC={auc_val:.3f})', linewidth=1.5)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Logistic Regression (Top vs Bottom Quintile)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'roc_curves.png'), dpi=150)
print("\nSaved roc_curves.png")
plt.close()

# ============================================================
# Plot 2: Rolling AUC over time
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
rolling_auc = best_res.set_index('Date')['AUC'].rolling(12).mean()
ax.plot(rolling_auc.index, rolling_auc.values, linewidth=1.5, color='#2ecc71')
ax.axhline(y=0.5, color='black', linewidth=0.8, linestyle='--', label='Random (0.5)')
ax.set_xlabel('Date')
ax.set_ylabel('12-Month Rolling AUC')
ax.set_title(f'Logistic Regression Rolling AUC (C={best_C})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'classification_comparison.png'), dpi=150)
print("Saved classification_comparison.png")
plt.close()

# ============================================================
# Compare with regression models
# ============================================================
print("\n" + "=" * 70)
print("FULL MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'Metric':<12} {'Value':<10}")
print("-" * 47)
print(f"{'OLS (Fama-MacBeth)':<25} {'IC':<12} {0.0602:<10.4f}")
print(f"{'Ridge':<25} {'IC':<12} {0.0623:<10.4f}")
print(f"{'Lasso':<25} {'IC':<12} {0.0512:<10.4f}")
print(f"{'Logistic (C={best_C})':<25} {'AUC':<12} {best_auc:<10.4f}")

print("\nDone!")