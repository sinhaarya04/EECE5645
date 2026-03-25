"""
03_preprocess_data.py
Preprocesses the clean panel: log-transforms market cap,
winsorizes at 1st/99th percentile, cross-sectional z-scores,
imputes missing values.
Input:  data/processed/panel_clean.csv
Output: data/processed/panel_final.csv
"""

import pandas as pd
import numpy as np
import os

# Paths
IN_PATH = os.path.join('data', 'processed', 'panel_clean.csv')
OUT_PATH = os.path.join('data', 'processed', 'panel_final.csv')

print("Loading clean panel data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])
print(f"Shape: {df.shape}")

# Define feature columns (exclude Date, Ticker, target, and raw price cols)
FEATURES = [
    'PE_RATIO', 'PX_TO_SALES_RATIO', 'CURRENT_EV_TO_T12M_EBITDA',
    'PX_TO_FREE_CASH_FLOW', 'EQY_DVD_YLD_12M', 'CUR_MKT_CAP',
    'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'GROSS_MARGIN', 'OPER_MARGIN',
    'TOT_DEBT_TO_TOT_EQY', 'CUR_RATIO', 'BETA_RAW_OVERRIDABLE',
    'EARN_YLD', 'VOLATILITY_90D', 'VOLUME_AVG_30D'
]

# Only keep features that exist in the dataframe
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"Using {len(FEATURES)} features: {FEATURES}")

# 1. Log-transform market cap (highly right-skewed)
if 'CUR_MKT_CAP' in df.columns:
    df['CUR_MKT_CAP'] = np.log1p(df['CUR_MKT_CAP'])
    print("Log-transformed CUR_MKT_CAP")

# 2. Fill dividend yield NaN with 0 (non-payers like AMZN)
if 'EQY_DVD_YLD_12M' in df.columns:
    df['EQY_DVD_YLD_12M'] = df['EQY_DVD_YLD_12M'].fillna(0)
    print("Filled dividend yield NaN with 0")

# 3. Cross-sectional winsorize and z-score per month
def cross_sectional_normalize(group):
    for col in FEATURES:
        s = group[col].copy()
        # Winsorize at 1st and 99th percentile
        lower = s.quantile(0.01)
        upper = s.quantile(0.99)
        s = s.clip(lower, upper)
        # Z-score
        mean, std = s.mean(), s.std()
        if std > 0:
            group[col] = (s - mean) / std
        else:
            group[col] = 0.0
    return group

print("Applying cross-sectional winsorization and z-scoring...")
df = df.groupby('Date', group_keys=False).apply(cross_sectional_normalize)

# 4. Fill remaining NAs with 0 (cross-sectional median after z-scoring is ~0)
df[FEATURES] = df[FEATURES].fillna(0)

# 5. Drop rows where target is missing
before = len(df)
df = df.dropna(subset=['FWD_RETURN'])
print(f"Dropped {before - len(df)} rows with missing target")

print(f"\nFinal shape: {df.shape}")
print(f"\nNull check: {df.isnull().sum().sum()} total nulls")
print(f"\nFeature stats after normalization:")
print(df[FEATURES].describe().round(3))

# Save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")