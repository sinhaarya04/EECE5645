"""
15_descriptive_stats.py
Descriptive statistics of raw features + correlation analysis.

Input:  data/processed/panel_clean.csv   (raw pre-normalization data)
        data/processed/panel_features.csv (post-normalization, 19 features)
Output: outputs/tables/descriptive_statistics.csv
        outputs/tables/correlation_matrix.csv
        outputs/figures/correlation_heatmap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_PATH = os.path.join('data', 'processed', 'panel_clean.csv')
FEAT_PATH = os.path.join('data', 'processed', 'panel_features.csv')

os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

RAW_FEATURES = [
    'PE_RATIO', 'PX_TO_SALES_RATIO', 'CURRENT_EV_TO_T12M_EBITDA',
    'PX_TO_FREE_CASH_FLOW', 'EQY_DVD_YLD_12M', 'CUR_MKT_CAP',
    'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'GROSS_MARGIN', 'OPER_MARGIN',
    'TOT_DEBT_TO_TOT_EQY', 'CUR_RATIO', 'BETA_RAW_OVERRIDABLE',
    'EARN_YLD', 'VOLATILITY_90D', 'VOLUME_AVG_30D'
]

NORM_FEATURES = RAW_FEATURES + ['MOMENTUM_12_1', 'REVERSAL_3M', 'HIGH_52W_RATIO']

# ============================================================
# 1. LOAD RAW DATA & COMPUTE DESCRIPTIVE STATISTICS
# ============================================================
print("Loading raw panel data...")
df_raw = pd.read_csv(RAW_PATH)
print(f"  Shape: {df_raw.shape}")

# Ensure numeric conversion for all feature columns
for col in RAW_FEATURES:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS (RAW FEATURES)")
print("=" * 70)

desc = df_raw[RAW_FEATURES].describe().T
desc['skewness'] = df_raw[RAW_FEATURES].skew()
desc['kurtosis'] = df_raw[RAW_FEATURES].kurtosis()

# Rename percentile columns for clarity
desc = desc.rename(columns={'25%': 'p25', '50%': 'median', '75%': 'p75'})

print(f"\n{desc.to_string(float_format='{:.4f}'.format)}")

desc.to_csv(os.path.join('outputs', 'tables', 'descriptive_statistics.csv'))
print("\nSaved descriptive_statistics.csv")

# Print key observations
print("\n--- Key Observations ---")
most_skewed = desc['skewness'].abs().idxmax()
print(f"  Most skewed feature: {most_skewed} (skewness = {desc.loc[most_skewed, 'skewness']:.4f})")
most_kurtotic = desc['kurtosis'].abs().idxmax()
print(f"  Highest kurtosis:    {most_kurtotic} (kurtosis = {desc.loc[most_kurtotic, 'kurtosis']:.4f})")
highest_missing = (df_raw[RAW_FEATURES].isnull().sum() / len(df_raw)).idxmax()
missing_pct = df_raw[RAW_FEATURES].isnull().sum() / len(df_raw)
print(f"  Most missing data:   {highest_missing} ({missing_pct[highest_missing]:.2%} missing)")

# ============================================================
# 2. LOAD NORMALIZED DATA & COMPUTE CORRELATION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS (NORMALIZED FEATURES)")
print("=" * 70)

print("Loading normalized panel data...")
df_feat = pd.read_csv(FEAT_PATH)
print(f"  Shape: {df_feat.shape}")

# Ensure numeric conversion
for col in NORM_FEATURES:
    if col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')

available_features = [f for f in NORM_FEATURES if f in df_feat.columns]
print(f"  Available features: {len(available_features)} / {len(NORM_FEATURES)}")

corr = df_feat[available_features].corr(method='pearson')
corr.to_csv(os.path.join('outputs', 'tables', 'correlation_matrix.csv'))
print("Saved correlation_matrix.csv")

# Print top absolute correlations (off-diagonal)
mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
corr_upper = corr.where(mask_upper)
corr_flat = corr_upper.stack().reset_index()
corr_flat.columns = ['Feature_A', 'Feature_B', 'Correlation']
corr_flat['abs_corr'] = corr_flat['Correlation'].abs()
top_pairs = corr_flat.nlargest(10, 'abs_corr')

print("\nTop 10 absolute pairwise correlations:")
for _, row in top_pairs.iterrows():
    print(f"  {row['Feature_A']:>30s} x {row['Feature_B']:<30s}  r = {row['Correlation']:+.4f}")

# ============================================================
# 3. CORRELATION HEATMAP (matplotlib only, no seaborn)
# ============================================================
print("\n" + "=" * 70)
print("PLOTTING CORRELATION HEATMAP")
print("=" * 70)

n = len(available_features)
fig, ax = plt.subplots(figsize=(14, 12))

# Use imshow with diverging colormap
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson Correlation', fontsize=12)

# Annotate each cell with correlation value
for i in range(n):
    for j in range(n):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=6, color=color)

# Axis labels
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(available_features, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(available_features, fontsize=8)
ax.set_title('Pearson Correlation Matrix (Normalized Features)', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'correlation_heatmap.png'), dpi=150)
print("Saved correlation_heatmap.png")
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Raw features analyzed:        {len(RAW_FEATURES)}")
print(f"  Observations (raw):           {len(df_raw):,}")
print(f"  Normalized features analyzed: {len(available_features)}")
print(f"  Observations (normalized):    {len(df_feat):,}")
print(f"  Avg abs correlation:          {corr_flat['abs_corr'].mean():.4f}")
print(f"  Max abs correlation:          {corr_flat['abs_corr'].max():.4f} "
      f"({top_pairs.iloc[0]['Feature_A']} x {top_pairs.iloc[0]['Feature_B']})")

print("\nDone!")
