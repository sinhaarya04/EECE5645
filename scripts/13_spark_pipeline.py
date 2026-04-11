"""
13_spark_pipeline.py
Replicates the Fama-MacBeth cross-sectional regression pipeline using PySpark.
Demonstrates distributed data processing on the same panel dataset.
Runs cross-sectional OLS at each date in parallel via Spark UDFs.

Input:  data/processed/panel_features.csv
Output: outputs/tables/spark_fama_macbeth_results.csv
        outputs/figures/spark_factor_premia.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

IN_PATH = os.path.join('data', 'processed', 'panel_features.csv')
TABLE_OUT = os.path.join('outputs', 'tables', 'spark_fama_macbeth_results.csv')
FIG_OUT = os.path.join('outputs', 'figures', 'spark_factor_premia.png')

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
# Initialize Spark
# ============================================================
print("=" * 70)
print("INITIALIZING SPARK SESSION")
print("=" * 70)

spark = SparkSession.builder \
    .appName("EECE5645_FamaMacBeth") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# ============================================================
# Load data into Spark DataFrame
# ============================================================
print("\nLoading data into Spark...")
start_load = time.time()

pdf = pd.read_csv(IN_PATH)
pdf['Date'] = pd.to_datetime(pdf['Date'])
pdf['Date'] = pdf['Date'].dt.strftime('%Y-%m-%d')

# Drop rows with missing target or features
pdf = pdf.dropna(subset=FEATURES + [TARGET])

sdf = spark.createDataFrame(pdf)
sdf.cache()
row_count = sdf.count()
load_time = time.time() - start_load

print(f"Loaded {row_count} rows in {load_time:.2f}s")
print(f"Schema:")
sdf.printSchema()

# ============================================================
# Spark Fama-MacBeth: groupBy Date, run OLS per group
# ============================================================
print("\n" + "=" * 70)
print("SPARK FAMA-MACBETH REGRESSIONS")
print("=" * 70)

# Define output schema for the UDF
coef_fields = [StructField('Date', StringType(), False)]
coef_fields += [StructField(f, DoubleType(), True) for f in ['intercept'] + FEATURES]
coef_schema = StructType(coef_fields)


def fama_macbeth_udf(key, group_pdf):
    """
    Pandas UDF: runs OLS cross-sectional regression for one date.
    key: (Date,)
    group_pdf: pandas DataFrame for that date
    """
    date_val = key[0]
    sub = group_pdf[FEATURES + [TARGET]].dropna()

    if len(sub) < 30:
        return pd.DataFrame()

    X = sub[FEATURES].values
    y = sub[TARGET].values
    X_int = np.column_stack([np.ones(len(X)), X])

    try:
        beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        row = {'Date': date_val, 'intercept': float(beta[0])}
        for i, f in enumerate(FEATURES):
            row[f] = float(beta[i + 1])
        return pd.DataFrame([row])
    except np.linalg.LinAlgError:
        return pd.DataFrame()


print("Running cross-sectional regressions via Spark groupBy...")
start_fm = time.time()

coefs_sdf = sdf.groupBy('Date').applyInPandas(fama_macbeth_udf, schema=coef_schema)
coefs_pdf = coefs_sdf.toPandas()

fm_time = time.time() - start_fm
print(f"Completed {len(coefs_pdf)} cross-sections in {fm_time:.2f}s")

# ============================================================
# Analyze: time-series average of coefficients + t-stats
# ============================================================
print("\n" + "=" * 70)
print("SPARK FAMA-MACBETH RESULTS")
print("=" * 70)


def summarize_coefs(coefs_df, features, label="ALL"):
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
            'Significant': '***' if p_val < 0.01 else (
                '**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
        })
    return pd.DataFrame(results)


results_all = summarize_coefs(coefs_pdf, FEATURES, "ALL")
print(results_all.to_string(index=False))

# ============================================================
# Verify against pandas baseline (script 05)
# ============================================================
print("\n" + "=" * 70)
print("VERIFICATION: SPARK vs PANDAS COEFFICIENTS")
print("=" * 70)

pandas_table = os.path.join('outputs', 'tables', 'fama_macbeth_results.csv')
if os.path.exists(pandas_table):
    pandas_results = pd.read_csv(pandas_table)
    pandas_all = pandas_results[pandas_results['Regime'] == 'ALL'].copy()

    print(f"{'Factor':<35} {'Pandas':>10} {'Spark':>10} {'Diff':>10}")
    print("-" * 65)
    for _, row in results_all.iterrows():
        factor = row['Factor']
        spark_coef = row['Mean_Coef']
        pandas_row = pandas_all[pandas_all['Factor'] == factor]
        if len(pandas_row) > 0:
            pandas_coef = pandas_row['Mean_Coef'].values[0]
            diff = abs(spark_coef - pandas_coef)
            print(f"  {factor:<33} {pandas_coef:>10.6f} {spark_coef:>10.6f} {diff:>10.6f}")
else:
    print(f"WARNING: {pandas_table} not found. Run script 05 first.")

# ============================================================
# Spark aggregation: factor statistics
# ============================================================
print("\n" + "=" * 70)
print("SPARK AGGREGATIONS")
print("=" * 70)

print("\nMean feature values by regime (via Spark):")
feature_path = os.path.join('data', 'processed', 'panel_features.csv')
if 'REGIME' in sdf.columns:
    agg_exprs = [F.mean(F.col(f)).alias(f) for f in FEATURES[:5]]
    regime_stats = sdf.groupBy('REGIME').agg(*agg_exprs)
    regime_stats.show(truncate=False)

# ============================================================
# Save results
# ============================================================
results_all.to_csv(TABLE_OUT, index=False)
print(f"\nSaved results to {TABLE_OUT}")

# ============================================================
# Plot: factor premia (matching script 05 style)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
sig_mask = results_all['p_value'] < 0.05
colors = ['#2ecc71' if s else '#95a5a6' for s in sig_mask]

bars = ax.bar(results_all['Factor'], results_all['Mean_Coef'],
              color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Mean Monthly Coefficient')
ax.set_title('Spark Fama-MacBeth Factor Premia (Green = Significant at 5%)')
ax.set_xticklabels(results_all['Factor'], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
print(f"Saved figure to {FIG_OUT}")
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SPARK PIPELINE SUMMARY")
print("=" * 70)
print(f"Data load time:        {load_time:.2f}s")
print(f"Fama-MacBeth time:     {fm_time:.2f}s")
print(f"Total Spark time:      {load_time + fm_time:.2f}s")
print(f"Cross-sections:        {len(coefs_pdf)}")
print(f"Significant factors:   {(results_all['p_value'] < 0.05).sum()}/{len(FEATURES)}")

# Cleanup
spark.stop()
print("\nDone!")
