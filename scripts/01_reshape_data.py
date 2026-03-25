"""
01_reshape_data.py
Converts the raw Bloomberg wide-format export into a long panel format.
Input:  data/raw/DATAEECE5645.csv
Output: data/processed/panel_data.csv
"""

import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = os.path.join('data', 'raw', 'DATAEECE5645.csv')
OUT_PATH = os.path.join('data', 'processed', 'panel_data.csv')

print("Loading raw Bloomberg data...")
raw = pd.read_csv(RAW_PATH, header=None, low_memory=False)
print(f"Raw shape: {raw.shape}")

# Row 3: ticker names + field descriptions
# Row 4: "Dates" + field mnemonics
# Row 5+: actual data
ticker_row = raw.iloc[3].values
field_row = raw.iloc[4].values

# Parse tickers and their associated column positions
# Tickers contain "EQUITY", field cells don't
ticker_cols = {}
current_ticker = None

for i in range(1, len(ticker_row)):
    val = str(ticker_row[i])
    if 'EQUITY' in val:
        current_ticker = val.strip()
        ticker_cols[current_ticker] = []
    elif current_ticker and isinstance(field_row[i], str):
        ticker_cols[current_ticker].append((i, field_row[i]))

print(f"Found {len(ticker_cols)} tickers")
print(f"Fields per ticker: {len(list(ticker_cols.values())[0])}")
print(f"Sample tickers: {list(ticker_cols.keys())[:5]}")

# Build long panel
dates = raw.iloc[5:, 0].values
rows = []

for ticker, cols in ticker_cols.items():
    for date_idx in range(len(dates)):
        row = {'Date': dates[date_idx], 'Ticker': ticker}
        for col_idx, field_name in cols:
            row[field_name] = raw.iloc[5 + date_idx, col_idx]
        rows.append(row)

df = pd.DataFrame(rows)
print(f"\nPanel shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")