"""
20_ff_alpha.py
Fama-French 5-factor + momentum regression to test whether portfolio alpha
is genuine (not explained by common risk factors).

Downloads FF factors from Ken French's data library, reconstructs monthly
L/S returns for each model, and runs factor regressions.

Input:  data/predictions/*.csv
        Fama-French factors (downloaded)
Output: outputs/tables/ff_alpha_regression.csv
        outputs/figures/alpha_comparison.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

PRED_DIR = os.path.join('data', 'predictions')
os.makedirs(os.path.join('outputs', 'tables'), exist_ok=True)
os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)

MODEL_FILES = {
    'OLS': 'OLS_FamaMacBeth_predictions.csv',
    'Ridge': 'Ridge_predictions.csv',
    'Lasso': 'Lasso_predictions.csv',
    'NN': 'NN_64_32_predictions.csv',
}

MODEL_COLORS = {
    'OLS': '#3498db',
    'Ridge': '#2ecc71',
    'Lasso': '#e74c3c',
    'NN': '#9b59b6',
}

# ============================================================
# Helper: download and parse FF factor CSV from zip
# ============================================================
def download_ff_csv(url, label="factors"):
    """Download a zip file from Ken French's site and extract the CSV."""
    print(f"  Downloading {label} from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=30)
    zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        raw = zf.read(csv_name).decode('utf-8', errors='replace')
    return raw


def parse_ff5_csv(raw_text):
    """
    Parse the Fama-French 5-factor CSV.
    Skip header lines until we hit the first numeric YYYYMM date.
    Stop when we hit annual data or blank lines.
    Returns DataFrame with columns: Date, Mkt-RF, SMB, HML, RMW, CMA, RF.
    """
    lines = raw_text.strip().split('\n')
    data_lines = []
    started = False

    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 7 and not started:
            # Check if first field looks like YYYYMM (6 digits)
            if parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
                started = True
        if started:
            if len(parts) < 7:
                break
            if not parts[0].strip().isdigit() or len(parts[0].strip()) != 6:
                break
            data_lines.append(parts[:7])

    df = pd.DataFrame(data_lines, columns=['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0  # Convert from percent
    df['Date'] = df['Date'].astype(str)
    return df


def parse_momentum_csv(raw_text):
    """
    Parse the momentum factor CSV.
    Similar structure: skip headers, find YYYYMM rows.
    Returns DataFrame with columns: Date, UMD.
    """
    lines = raw_text.strip().split('\n')
    data_lines = []
    started = False

    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and not started:
            if parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
                started = True
        if started:
            if len(parts) < 2:
                break
            if not parts[0].strip().isdigit() or len(parts[0].strip()) != 6:
                break
            data_lines.append(parts[:2])

    df = pd.DataFrame(data_lines, columns=['Date', 'UMD'])
    df['UMD'] = pd.to_numeric(df['UMD'], errors='coerce') / 100.0
    df['Date'] = df['Date'].astype(str)
    return df


# ============================================================
# 1. Download Fama-French factors
# ============================================================
print("=" * 70)
print("DOWNLOADING FAMA-FRENCH FACTORS")
print("=" * 70)

ff_available = True
try:
    ff5_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
    mom_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'

    ff5_raw = download_ff_csv(ff5_url, "FF 5 factors")
    ff5 = parse_ff5_csv(ff5_raw)
    print(f"  FF5: {len(ff5)} months ({ff5['Date'].iloc[0]} to {ff5['Date'].iloc[-1]})")

    mom_raw = download_ff_csv(mom_url, "Momentum factor")
    mom = parse_momentum_csv(mom_raw)
    print(f"  Momentum: {len(mom)} months ({mom['Date'].iloc[0]} to {mom['Date'].iloc[-1]})")

    # Merge FF5 + momentum
    ff_factors = ff5.merge(mom, on='Date', how='inner')
    print(f"  Combined: {len(ff_factors)} months")

except Exception as e:
    print(f"\n  WARNING: Could not download Fama-French factors: {e}")
    print("  Falling back to intercept-only regression (alpha estimation without factor adjustment).")
    ff_available = False
    ff_factors = None

# ============================================================
# 2. Load predictions and construct monthly L/S returns
# ============================================================
print("\n" + "=" * 70)
print("CONSTRUCTING MONTHLY LONG-SHORT RETURNS")
print("=" * 70)

model_returns = {}
for name, fname in MODEL_FILES.items():
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found -- skipping {name}")
        continue

    preds = pd.read_csv(path)
    preds['Date'] = pd.to_datetime(preds['Date'])

    monthly_ls = []
    for date, group in preds.groupby('Date'):
        n_decile = 10
        if len(group) < n_decile * 2:
            continue
        group = group.sort_values('predicted_return', ascending=False)
        n = len(group) // n_decile
        long_ret = group.head(n)['actual_return'].mean()
        short_ret = group.tail(n)['actual_return'].mean()
        ls_ret = long_ret - short_ret

        # Convert date to YYYYMM string for merging with FF data
        yyyymm = date.strftime('%Y%m')
        monthly_ls.append({'Date': yyyymm, 'LS_Return': ls_ret})

    ret_df = pd.DataFrame(monthly_ls)
    model_returns[name] = ret_df
    print(f"  {name}: {len(ret_df)} months")

if not model_returns:
    raise FileNotFoundError("No prediction files found. Run scripts 05, 06, 09 first.")

# ============================================================
# 3. Factor regression for each model
# ============================================================
print("\n" + "=" * 70)
print("FAMA-FRENCH FACTOR REGRESSIONS")
print("=" * 70)

FACTOR_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']

regression_results = []

for model_name, ret_df in model_returns.items():
    print(f"\n--- {model_name} ---")

    if ff_available and ff_factors is not None:
        # Merge portfolio returns with FF factors
        merged = ret_df.merge(ff_factors, on='Date', how='inner')
        merged = merged.dropna()

        if len(merged) < 12:
            print(f"  WARNING: Only {len(merged)} overlapping months, skipping.")
            continue

        print(f"  Overlapping months: {len(merged)}")

        y = merged['LS_Return'].values
        X_factors = merged[FACTOR_COLS].values
        # Add intercept column
        X = np.column_stack([np.ones(len(y)), X_factors])

        # OLS via least squares
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

        y_hat = X @ beta
        resid = y - y_hat
        n_obs = len(y)
        k = X.shape[1]  # number of parameters (including intercept)

        # MSE and standard errors
        mse = np.sum(resid ** 2) / (n_obs - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(mse * np.diag(XtX_inv))

        # R-squared
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        alpha_monthly = beta[0]
        alpha_annual = alpha_monthly * 12
        alpha_se = se[0]
        alpha_tstat = alpha_monthly / alpha_se if alpha_se > 0 else 0.0
        alpha_pval = 2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n_obs - k))

        if alpha_pval < 0.01:
            sig = '***'
        elif alpha_pval < 0.05:
            sig = '**'
        elif alpha_pval < 0.10:
            sig = '*'
        else:
            sig = ''

        result_row = {
            'Model': model_name,
            'Alpha_Monthly': round(alpha_monthly, 6),
            'Alpha_Annual': round(alpha_annual, 4),
            'Alpha_tstat': round(alpha_tstat, 3),
            'Alpha_pval': round(alpha_pval, 4),
            'Alpha_sig': sig,
            'Alpha_SE': round(alpha_se, 6),
            'R_squared': round(r_squared, 4),
        }
        for i, fc in enumerate(FACTOR_COLS):
            result_row[f'{fc.replace("-", "")}_beta'] = round(beta[i + 1], 4)

        regression_results.append(result_row)

        # Print formatted results
        print(f"  Alpha (monthly):   {alpha_monthly:.6f}")
        print(f"  Alpha (annual):    {alpha_annual:.4f} ({alpha_annual:.2%})")
        print(f"  Alpha t-stat:      {alpha_tstat:.3f} {sig}")
        print(f"  Alpha p-value:     {alpha_pval:.4f}")
        print(f"  R-squared:         {r_squared:.4f}")
        print(f"  Factor loadings:")
        for i, fc in enumerate(FACTOR_COLS):
            fc_beta = beta[i + 1]
            fc_tstat = beta[i + 1] / se[i + 1] if se[i + 1] > 0 else 0
            fc_pval = 2 * (1 - stats.t.cdf(abs(fc_tstat), df=n_obs - k))
            fc_sig = '***' if fc_pval < 0.01 else ('**' if fc_pval < 0.05 else ('*' if fc_pval < 0.10 else ''))
            print(f"    {fc:>8s}: beta={fc_beta:+.4f}  t={fc_tstat:.3f} {fc_sig}")

    else:
        # Fallback: intercept-only regression (no FF factors)
        y = ret_df['LS_Return'].values
        n_obs = len(y)

        alpha_monthly = y.mean()
        alpha_annual = alpha_monthly * 12
        alpha_se = y.std() / np.sqrt(n_obs)
        alpha_tstat = alpha_monthly / alpha_se if alpha_se > 0 else 0.0
        alpha_pval = 2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n_obs - 1))

        if alpha_pval < 0.01:
            sig = '***'
        elif alpha_pval < 0.05:
            sig = '**'
        elif alpha_pval < 0.10:
            sig = '*'
        else:
            sig = ''

        result_row = {
            'Model': model_name,
            'Alpha_Monthly': round(alpha_monthly, 6),
            'Alpha_Annual': round(alpha_annual, 4),
            'Alpha_tstat': round(alpha_tstat, 3),
            'Alpha_pval': round(alpha_pval, 4),
            'Alpha_sig': sig,
            'Alpha_SE': round(alpha_se, 6),
            'R_squared': np.nan,
        }
        for fc in FACTOR_COLS:
            result_row[f'{fc.replace("-", "")}_beta'] = np.nan

        regression_results.append(result_row)

        print(f"  Alpha (monthly):   {alpha_monthly:.6f} (no FF adjustment)")
        print(f"  Alpha (annual):    {alpha_annual:.4f} ({alpha_annual:.2%})")
        print(f"  Alpha t-stat:      {alpha_tstat:.3f} {sig}")

# ============================================================
# 4. Save regression results
# ============================================================
results_df = pd.DataFrame(regression_results)

# Ensure consistent column order
base_cols = ['Model', 'Alpha_Monthly', 'Alpha_Annual', 'Alpha_tstat',
             'Alpha_pval', 'Alpha_sig', 'R_squared']
beta_cols = [f'{fc.replace("-", "")}_beta' for fc in FACTOR_COLS]
results_df = results_df[base_cols + ['Alpha_SE'] + beta_cols]

results_df.to_csv(os.path.join('outputs', 'tables', 'ff_alpha_regression.csv'), index=False)
print(f"\nSaved ff_alpha_regression.csv")

# ============================================================
# Print summary table
# ============================================================
print("\n" + "=" * 70)
print("FAMA-FRENCH ALPHA REGRESSION SUMMARY")
print("=" * 70)
print(f"{'Model':<10} {'Alpha(ann)':<14} {'t-stat':<10} {'p-val':<10} {'Sig':<6} {'R2':<8}")
print("-" * 58)
for _, row in results_df.iterrows():
    r2_str = f"{row['R_squared']:.4f}" if pd.notna(row['R_squared']) else 'N/A'
    print(f"{row['Model']:<10} {row['Alpha_Annual']:>+12.4f}  {row['Alpha_tstat']:>8.3f}  "
          f"{row['Alpha_pval']:>8.4f}  {row['Alpha_sig']:<6s} {r2_str:<8}")

# ============================================================
# 5. Alpha comparison bar chart
# ============================================================
print("\nGenerating alpha comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

models = results_df['Model'].values
alphas_annual = results_df['Alpha_Annual'].values
alpha_ses = results_df['Alpha_SE'].values
sigs = results_df['Alpha_sig'].values

# Error bars: alpha +/- 1.96 * SE * 12 (annualized)
error_bars = 1.96 * alpha_ses * 12

colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]

bars = ax.bar(models, alphas_annual, color=colors, edgecolor='black',
              linewidth=0.8, yerr=error_bars, capsize=6, error_kw={'linewidth': 1.2})

# Add significance stars above bars
for bar, alpha_val, sig, err in zip(bars, alphas_annual, sigs, error_bars):
    if sig:
        y_pos = alpha_val + err + 0.002 if alpha_val >= 0 else alpha_val - err - 0.005
        va = 'bottom' if alpha_val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos, sig,
                ha='center', va=va, fontsize=14, fontweight='bold', color='black')

# Add value labels on bars
for bar, alpha_val in zip(bars, alphas_annual):
    y_pos = alpha_val / 2 if abs(alpha_val) > 0.005 else alpha_val + 0.001
    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
            f'{alpha_val:.2%}', ha='center', va='center', fontsize=10, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Annualized Alpha')
ax.set_title('Fama-French 6-Factor Alpha by Model\n(Long-Short Portfolio, with 95% CI)')
ax.grid(axis='y', alpha=0.3)

# Format y-axis as percentage
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'alpha_comparison.png'), dpi=150)
print("Saved alpha_comparison.png")
plt.close()

# ============================================================
# Interpretation
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

sig_models = results_df[results_df['Alpha_sig'].str.len() > 0]['Model'].tolist()
nonsig_models = results_df[results_df['Alpha_sig'].str.len() == 0]['Model'].tolist()

if sig_models:
    print(f"Models with statistically significant alpha: {', '.join(sig_models)}")
    print("  -> These models generate returns NOT explained by common risk factors.")
else:
    print("No models show statistically significant alpha after FF adjustment.")

if nonsig_models:
    print(f"Models without significant alpha: {', '.join(nonsig_models)}")
    print("  -> Returns from these models may be explained by exposure to common factors.")

if ff_available:
    print("\nNote: Regressions use Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA)")
    print("      plus Momentum (UMD) -- a 6-factor model.")
else:
    print("\nNote: FF factors unavailable. Alphas are unadjusted (intercept-only).")

print("\nDone!")
