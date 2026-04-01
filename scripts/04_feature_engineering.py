"""
04_feature_engineering.py
Adds computed features: 12-1 month momentum, 3-month reversal,
52-week high ratio, and market regime labels.
Input:  data/processed/panel_final.csv
Output: data/processed/panel_features.csv
"""

import pandas as pd
import numpy as np
import os

IN_PATH = os.path.join('data', 'processed', 'panel_final.csv')
OUT_PATH = os.path.join('data', 'processed', 'panel_features.csv')

print("Loading preprocessed data...")
df = pd.read_csv(IN_PATH)
df['Date'] = pd.to_datetime(df['Date'])
print(f"Shape: {df.shape}")

# ---------- Clean remaining nulls in PX_LAST ----------
df = df.dropna(subset=['PX_LAST', 'TOT_RETURN_INDEX_GROSS_DVDS'])
print(f"After dropping null prices: {df.shape}")

# ---------- 1. Momentum (12-1): return over past 12 months, skip most recent month ----------
# Classic Jegadeesh-Titman construction
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

df['RET_1M'] = df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(1)
df['RET_12M'] = df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(12)

# 12-1 momentum = 12 month return minus last 1 month return
df['MOMENTUM_12_1'] = df['RET_12M'] - df['RET_1M']

# ---------- 2. Reversal (3-month): trailing 3-month return ----------
df['REVERSAL_3M'] = df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(3)

# ---------- 3. 52-week high ratio: current price / rolling 12-month max ----------
df['HIGH_12M'] = df.groupby('Ticker')['PX_LAST'].transform(
    lambda x: x.rolling(window=12, min_periods=6).max()
)
df['HIGH_52W_RATIO'] = df['PX_LAST'] / df['HIGH_12M']

# ---------- 4. Market Regime Labels ----------
# Use SPX-level returns to classify regimes
# Compute equal-weight market return per month as proxy
market_ret = df.groupby('Date')['FWD_RETURN'].mean().reset_index()
market_ret.columns = ['Date', 'MKT_RETURN']
market_ret = market_ret.sort_values('Date')

# Trailing 6-month cumulative return for bull detection
market_ret['MKT_6M'] = market_ret['MKT_RETURN'].rolling(6).sum()
# Trailing 3-month cumulative return for crash detection
market_ret['MKT_3M'] = market_ret['MKT_RETURN'].rolling(3).sum()

# Regime classification
def classify_regime(row):
    if row['MKT_3M'] < -0.10:
        return 'CRASH'
    elif row['MKT_6M'] > 0.10:
        return 'BULL'
    else:
        return 'NEUTRAL'

market_ret['REGIME'] = market_ret.apply(classify_regime, axis=1)

# Merge regime back to main dataframe
df = df.merge(market_ret[['Date', 'REGIME']], on='Date', how='left')

# ---------- 5. Cross-sectional z-score the new features ----------
new_features = ['MOMENTUM_12_1', 'REVERSAL_3M', 'HIGH_52W_RATIO']

def zscore_new_features(group):
    for col in new_features:
        s = group[col].copy()
        lower = s.quantile(0.01)
        upper = s.quantile(0.99)
        s = s.clip(lower, upper)
        mean, std = s.mean(), s.std()
        if std > 0:
            group[col] = (s - mean) / std
        else:
            group[col] = 0.0
    return group

print("Z-scoring new features...")
df = df.groupby('Date', group_keys=False).apply(zscore_new_features)
df[new_features] = df[new_features].fillna(0)

# ---------- 6. Drop intermediate columns ----------
df = df.drop(columns=['RET_1M', 'RET_12M', 'HIGH_12M'], errors='ignore')

# ---------- 7. Drop rows with any remaining nulls in target ----------
df = df.dropna(subset=['FWD_RETURN'])

# ---------- Summary ----------
print(f"\nFinal shape: {df.shape}")
print(f"Null count: {df.isnull().sum().sum()}")
print(f"\nRegime distribution:")
print(df['REGIME'].value_counts())
print(f"\nNew feature stats:")
print(df[new_features].describe().round(3))
print(f"\nAll columns: {df.columns.tolist()}")

# Save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")