"""
19_shap_analysis.py
Neural net feature importance via permutation importance, plus cross-model
comparison (Fama-MacBeth |t-stat|, Lasso |coefficient|, NN permutation importance).

Input:  data/processed/panel_features.csv
        outputs/tables/fama_macbeth_results.csv
Output: outputs/tables/nn_feature_importance.csv
        outputs/tables/feature_importance_comparison.csv
        outputs/figures/nn_feature_importance.png
        outputs/figures/feature_importance_comparison.png
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
FM_PATH = os.path.join('outputs', 'tables', 'fama_macbeth_results.csv')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

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
# Load data
# ============================================================
print("Loading data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])

dates = sorted(df['Date'].unique())
n_dates = len(dates)

# Temporal split: last 24 months as test, rest as train
split_idx = n_dates - 24
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

print(f"Total months: {n_dates}")
print(f"Train months: {len(train_dates)} ({str(train_dates[0])[:10]} to {str(train_dates[-1])[:10]})")
print(f"Test months:  {len(test_dates)} ({str(test_dates[0])[:10]} to {str(test_dates[-1])[:10]})")

train_df = df[df['Date'].isin(train_dates)][FEATURES + [TARGET]].dropna()
test_df = df[df['Date'].isin(test_dates)][FEATURES + [TARGET]].dropna()

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values

print(f"Train samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")

# ============================================================
# 1. Train NN_64_32
# ============================================================
print("\n" + "=" * 70)
print("TRAINING NEURAL NETWORK (64, 32)")
print("=" * 70)

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
nn_model.fit(X_train, y_train)

train_score = nn_model.score(X_train, y_train)
test_score = nn_model.score(X_test, y_test)
print(f"Train R2: {train_score:.4f}")
print(f"Test R2:  {test_score:.4f}")
print(f"Iterations: {nn_model.n_iter_}")

# ============================================================
# 2. Permutation importance on test set
# ============================================================
print("\nComputing permutation importance (n_repeats=10)...")
perm_result = permutation_importance(
    nn_model, X_test, y_test,
    n_repeats=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

nn_imp = pd.DataFrame({
    'Feature': FEATURES,
    'Importance_Mean': perm_result.importances_mean,
    'Importance_Std': perm_result.importances_std
}).sort_values('Importance_Mean', ascending=False).reset_index(drop=True)

nn_imp.to_csv(os.path.join('outputs', 'tables', 'nn_feature_importance.csv'), index=False)
print("Saved nn_feature_importance.csv")

print("\nNN Permutation Importance (top 10):")
print(nn_imp.head(10).to_string(index=False))

# ============================================================
# 3. Load Fama-MacBeth results (ALL regime, |t-stat|)
# ============================================================
print("\n" + "=" * 70)
print("LOADING FAMA-MACBETH RESULTS")
print("=" * 70)

fm_df = pd.read_csv(FM_PATH)
fm_all = fm_df[fm_df['Regime'] == 'ALL'][['Factor', 't_stat']].copy()
fm_all['FM_abs_tstat'] = fm_all['t_stat'].abs()
fm_all = fm_all.rename(columns={'Factor': 'Feature'})[['Feature', 'FM_abs_tstat']]
print(f"Loaded {len(fm_all)} features from Fama-MacBeth")

# ============================================================
# 4. Train Lasso on same train set, get |coefficients|
# ============================================================
print("\n" + "=" * 70)
print("TRAINING LASSO (alpha=0.001)")
print("=" * 70)

lasso_model = Lasso(alpha=0.001, max_iter=10000, random_state=42)
lasso_model.fit(X_train, y_train)

lasso_imp = pd.DataFrame({
    'Feature': FEATURES,
    'Lasso_abs_coef': np.abs(lasso_model.coef_)
})

lasso_r2 = lasso_model.score(X_test, y_test)
print(f"Lasso Test R2: {lasso_r2:.4f}")
print(f"Non-zero coefficients: {np.sum(np.abs(lasso_model.coef_) > 1e-8)}/{len(FEATURES)}")

# ============================================================
# 5. Build comparison table
# ============================================================
print("\n" + "=" * 70)
print("CROSS-MODEL FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

nn_comp = nn_imp[['Feature', 'Importance_Mean']].rename(
    columns={'Importance_Mean': 'NN_perm_importance'}
)

comparison = nn_comp.merge(fm_all, on='Feature', how='left')
comparison = comparison.merge(lasso_imp, on='Feature', how='left')

# Normalize each column to [0, 1]
for col in ['FM_abs_tstat', 'Lasso_abs_coef', 'NN_perm_importance']:
    vals = comparison[col].values
    col_min = vals.min()
    col_max = vals.max()
    if col_max > col_min:
        comparison[col] = (vals - col_min) / (col_max - col_min)
    else:
        comparison[col] = 0.0

comparison = comparison[['Feature', 'FM_abs_tstat', 'Lasso_abs_coef', 'NN_perm_importance']]
comparison.to_csv(os.path.join('outputs', 'tables', 'feature_importance_comparison.csv'), index=False)
print("Saved feature_importance_comparison.csv")
print(comparison.to_string(index=False))

# ============================================================
# 6. Print top 5 features for each method
# ============================================================
print("\n" + "=" * 70)
print("TOP 5 FEATURES BY METHOD")
print("=" * 70)

for col, label in [('FM_abs_tstat', 'Fama-MacBeth |t-stat|'),
                    ('Lasso_abs_coef', 'Lasso |coefficient|'),
                    ('NN_perm_importance', 'NN Permutation Importance')]:
    top5 = comparison.nlargest(5, col)[['Feature', col]]
    print(f"\n{label}:")
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"  {i}. {row.Feature:<35s} {getattr(row, col):.4f}")

# ============================================================
# Plot 1: NN Feature Importance (horizontal bar chart)
# ============================================================
print("\nGenerating plots...")

fig, ax = plt.subplots(figsize=(10, 8))

# Sort by importance ascending for horizontal bars (top feature at top)
nn_sorted = nn_imp.sort_values('Importance_Mean', ascending=True)

y_pos = np.arange(len(nn_sorted))
ax.barh(y_pos, nn_sorted['Importance_Mean'], xerr=nn_sorted['Importance_Std'],
        color='#9b59b6', edgecolor='black', linewidth=0.5, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(nn_sorted['Feature'], fontsize=9)
ax.set_xlabel('Permutation Importance (decrease in neg MSE)')
ax.set_title('Neural Network Feature Importance (Permutation-Based)')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'nn_feature_importance.png'), dpi=150)
print("Saved nn_feature_importance.png")
plt.close()

# ============================================================
# Plot 2: 3-Panel Comparison (same feature order, sorted by NN importance)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Sort by NN importance ascending (so top feature is at top of bar chart)
comp_sorted = comparison.sort_values('NN_perm_importance', ascending=True)
y_pos = np.arange(len(comp_sorted))

# Panel 1: Fama-MacBeth |t-stat|
axes[0].barh(y_pos, comp_sorted['FM_abs_tstat'], color='#3498db',
             edgecolor='black', linewidth=0.5)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(comp_sorted['Feature'], fontsize=8)
axes[0].set_xlabel('Normalized |t-stat|')
axes[0].set_title('Fama-MacBeth |t-stat|')
axes[0].set_xlim(0, 1.05)

# Panel 2: Lasso |coefficient|
axes[1].barh(y_pos, comp_sorted['Lasso_abs_coef'], color='#e74c3c',
             edgecolor='black', linewidth=0.5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(comp_sorted['Feature'], fontsize=8)
axes[1].set_xlabel('Normalized |coefficient|')
axes[1].set_title('Lasso |coefficient|')
axes[1].set_xlim(0, 1.05)

# Panel 3: NN permutation importance
axes[2].barh(y_pos, comp_sorted['NN_perm_importance'], color='#9b59b6',
             edgecolor='black', linewidth=0.5)
axes[2].set_yticks(y_pos)
axes[2].set_yticklabels(comp_sorted['Feature'], fontsize=8)
axes[2].set_xlabel('Normalized Importance')
axes[2].set_title('NN Permutation Importance')
axes[2].set_xlim(0, 1.05)

fig.suptitle('Feature Importance Comparison Across Methods (Normalized to [0, 1])',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join('outputs', 'figures', 'feature_importance_comparison.png'), dpi=150)
print("Saved feature_importance_comparison.png")
plt.close()

print("\nDone!")
