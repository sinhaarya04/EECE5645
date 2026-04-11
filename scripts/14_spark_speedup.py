"""
14_spark_speedup.py
Collects timing results from multiple spark runs and generates speedup plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

TABLE_DIR = 'outputs/tables'
FIG_DIR = 'outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(TABLE_DIR, 'spark_results_*cores.csv')))
if not files:
    print("No spark results found.")
    exit()

all_data = []
for f in files:
    df = pd.read_csv(f)
    all_data.append(df)

results = pd.concat(all_data, ignore_index=True)
print("Loaded results:")
print(results.to_string(index=False))

models = results['Model'].unique()
core_counts = sorted(results['Cores'].unique())
baseline_times = results[results['Cores'] == core_counts[0]].set_index('Model')['Time_sec']

speedup_rows = []
for _, row in results.iterrows():
    base = baseline_times.get(row['Model'], row['Time_sec'])
    speedup = base / row['Time_sec'] if row['Time_sec'] > 0 else 0
    speedup_rows.append({'Model': row['Model'], 'Cores': row['Cores'], 'Time_sec': row['Time_sec'], 'Speedup': speedup})

speedup_df = pd.DataFrame(speedup_rows)
speedup_df.to_csv(os.path.join(TABLE_DIR, 'speedup_summary.csv'), index=False)

print("\nSpeedup Summary:")
pivot = speedup_df.pivot_table(values='Speedup', index='Model', columns='Cores')
print(pivot.to_string(float_format='{:.2f}'.format))

fig, ax = plt.subplots(figsize=(12, 7))
colors = {'FamaMacBeth': '#3498db', 'Ridge': '#2ecc71', 'Lasso': '#e74c3c', 'NN_64_32': '#9b59b6'}
for model in models:
    model_data = speedup_df[speedup_df['Model'] == model]
    ax.plot(model_data['Cores'], model_data['Speedup'], 'o-', label=model, linewidth=2, markersize=8, color=colors.get(model, 'gray'))
ax.plot(core_counts, core_counts, 'k--', linewidth=1, label='Ideal Linear', alpha=0.5)
ax.set_xlabel('Number of Cores', fontsize=12)
ax.set_ylabel('Speedup (vs 1 core)', fontsize=12)
ax.set_title('Distributed ML Speedup Analysis', fontsize=14)
ax.set_xticks(core_counts)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'speedup_chart.png'), dpi=150)
print("\nSaved speedup_chart.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
for model in models:
    model_data = speedup_df[speedup_df['Model'] == model]
    ax.plot(model_data['Cores'], model_data['Time_sec'], 'o-', label=model, linewidth=2, markersize=8, color=colors.get(model, 'gray'))
ax.set_xlabel('Number of Cores', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Training Time vs Number of Cores', fontsize=14)
ax.set_xticks(core_counts)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'time_vs_cores.png'), dpi=150)
print("Saved time_vs_cores.png")
plt.close()

print("\nDone!")
