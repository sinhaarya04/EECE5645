"""
02_clean_data.py
Cleans the reshaped panel: drops nulls, removes bad columns,
computes forward 1-month return as target variable.
Input:  data/processed/panel_data.csv
Output: data/processed/panel_clean.csv
"""

import pandas as pd
import numpy as np
import os

# Paths
IN_PATH = os.path.join('data', 'processed', 'panel_data.csv')
OUT_PATH = os.path.join('data', 'processed', 'panel_clean.csv')

print("Loading panel data...")
df = pd.read_csv(IN_PATH)
print(f"Before cleaning: {df.shape}")

# 1. Drop rows where Date is null (empty rows from reshape)
df = df.dropna(subset=['Date'])
print(f"After dropping null dates: {df.shape}")

# 2. Drop high-null columns (>50% missing)
null_pct = df.isnull().mean()
high_null_cols = null_pct[null_pct > 0.5].index.tolist()
if high_null_cols:
    print(f"Dropping high-null columns: {high_null_cols}")
    df = df.drop(columns=high_null_cols)

# 3. Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# 4. Sort by Ticker and Date
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# 5. Compute forward 1-month return (target variable)
# For each ticker, shift the total return index back by 1 to get next month's return
df['FWD_RETURN'] = (
    df.groupby('Ticker')['TOT_RETURN_INDEX_GROSS_DVDS']
    .pct_change(periods=1)
    .shift(-1)
    .values
)

print(f"\nAfter cleaning: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nNull % per column:")
print(df.isnull().mean().round(3))
print(f"\nForward return stats:")
print(df['FWD_RETURN'].describe())
print(f"\nUnique tickers: {df['Ticker'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")