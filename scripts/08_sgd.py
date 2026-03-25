"""
08_sgd.py
Mini-batch Stochastic Gradient Descent for return prediction.
Compares convergence rates across different batch sizes.
Uses expanding-window evaluation like other models.

Input:  data/processed/panel_features.csv
Output: outputs/tables/sgd_results.csv
        outputs/figures/sgd_convergence.png
        outputs/figures/sgd_batch_comparison.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
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
# Manual mini-batch SGD implementation (for convergence tracking)
# ============================================================
def manual_sgd(X, y, batch_size, n_epochs=50, lr=0.001, alpha=0.01):
    """
    Mini-batch SGD with L2 regularization.
    Returns loss history per epoch for convergence analysis.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            
            # Forward pass
            y_pred = X_batch @ w + b
            error = y_pred - y_batch
            
            # Gradients (MSE + L2 regularization)
            grad_w = (2 / len(y_batch)) * (X_batch.T @ error) + 2 * alpha * w
            grad_b = (2 / len(y_batch)) * error.sum()
            
            # Update
            w -= lr * grad_w
            b -= lr * grad_b
            
            # Track loss
            batch_loss = np.mean(error ** 2) + alpha * np.sum(w ** 2)
            epoch_loss += batch_loss
            n_batches += 1
        
        loss_history.append(epoch_loss / n_batches)
    
    return w, b, loss_history


# ============================================================
# 1. Convergence analysis: compare batch sizes
# ============================================================
print("=" * 70)
print("SGD CONVERGENCE ANALYSIS")
print("=" * 70)

# Use a single large training set for convergence comparison
train_end = dates[MIN_TRAIN_MONTHS + 60]  # 10 years of training data
train = df[df['Date'] <= train_end][FEATURES + [TARGET]].dropna()
X_train = train[FEATURES].values
y_train = train[TARGET].values
print(f"Training set: {X_train.shape}")

BATCH_SIZES = [32, 64, 128, 256, 512, 1024]
N_EPOCHS = 50

convergence_results = {}

for bs in BATCH_SIZES:
    print(f"\nBatch size = {bs}:")
    start = time.time()
    w, b, losses = manual_sgd(X_train, y_train, batch_size=bs, n_epochs=N_EPOCHS)
    elapsed = time.time() - start
    
    # Compute predictions on training set
    y_pred = X_train @ w + b
    ic = stats.spearmanr(y_pred, y_train)[0]
    mse = np.mean((y_pred - y_train) ** 2)
    
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Train IC: {ic:.4f}")
    print(f"  Train MSE: {mse:.6f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Batches/epoch: {len(X_train) // bs}")
    
    convergence_results[bs] = {
        'losses': losses,
        'final_loss': losses[-1],
        'ic': ic,
        'mse': mse,
        'time': elapsed
    }

# ============================================================
# 2. Out-of-sample prediction with best batch size
# ============================================================
print("\n" + "=" * 70)
print("SGD OUT-OF-SAMPLE PREDICTION (EXPANDING WINDOW)")
print("=" * 70)

# Use sklearn SGDRegressor for expanding window (faster)
def run_sgd_expanding(df, dates, features, target, batch_sizes=[64, 256, 1024]):
    """Run expanding window SGD with different batch sizes using partial_fit."""
    results = {bs: [] for bs in batch_sizes}
    
    for bs in batch_sizes:
        ic_list = []
        start = time.time()
        
        for i in range(MIN_TRAIN_MONTHS, len(dates) - 1):
            train_dates = dates[:i]
            test_date = dates[i]
            
            train = df[df['Date'].isin(train_dates)][features + [target]].dropna()
            test = df[df['Date'] == test_date][features + [target]].dropna()
            
            if len(train) < 100 or len(test) < 20:
                continue
            
            X_train, y_train = train[features].values, train[target].values
            X_test, y_test = test[features].values, test[target].values
            
            # Train SGD with mini-batches
            model = SGDRegressor(
                loss='squared_error',
                penalty='l2',
                alpha=0.01,
                max_iter=1,  # we control epochs manually
                tol=None,
                learning_rate='constant',
                eta0=0.001,
                random_state=42
            )
            
            # Manual epochs with mini-batches
            n_epochs = 30
            for epoch in range(n_epochs):
                indices = np.random.permutation(len(X_train))
                for start_idx in range(0, len(X_train), bs):
                    end_idx = min(start_idx + bs, len(X_train))
                    batch_idx = indices[start_idx:end_idx]
                    model.partial_fit(X_train[batch_idx], y_train[batch_idx])
            
            y_pred = model.predict(X_test)
            ic = stats.spearmanr(y_pred, y_test)[0]
            ic_list.append({'Date': test_date, 'IC': ic})
        
        elapsed = time.time() - start
        ic_df = pd.DataFrame(ic_list)
        
        print(f"\nBatch size = {bs}:")
        print(f"  Mean IC: {ic_df['IC'].mean():.4f}")
        print(f"  IC Std: {ic_df['IC'].std():.4f}")
        print(f"  IC > 0 pct: {(ic_df['IC'] > 0).mean():.2%}")
        print(f"  IC t-stat: {ic_df['IC'].mean() / (ic_df['IC'].std() / np.sqrt(len(ic_df))):.3f}")
        print(f"  Time: {elapsed:.2f}s")
        
        results[bs] = {'ic_df': ic_df, 'time': elapsed}
    
    return results

sgd_results = run_sgd_expanding(df, dates, FEATURES, TARGET)

# ============================================================
# Save results
# ============================================================
summary_rows = []
for bs, res in sgd_results.items():
    ic_df = res['ic_df']
    summary_rows.append({
        'Batch_Size': bs,
        'Mean_IC': ic_df['IC'].mean(),
        'IC_Std': ic_df['IC'].std(),
        'IC_Hit_Rate': (ic_df['IC'] > 0).mean(),
        'Time_sec': res['time']
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join('outputs', 'tables', 'sgd_results.csv'), index=False)

# ============================================================
# Plot 1: Convergence curves by batch size
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
for bs in BATCH_SIZES:
    ax.plot(convergence_results[bs]['losses'], label=f'Batch={bs}', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE + L2)')
ax.set_title('SGD Convergence by Batch Size')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'sgd_convergence.png'), dpi=150)
print("\nSaved sgd_convergence.png")
plt.close()

# ============================================================
# Plot 2: Batch size vs IC and time
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

batch_sizes_oos = list(sgd_results.keys())
ics = [sgd_results[bs]['ic_df']['IC'].mean() for bs in batch_sizes_oos]
times = [sgd_results[bs]['time'] for bs in batch_sizes_oos]

ax1.bar([str(bs) for bs in batch_sizes_oos], ics, color='#3498db', edgecolor='black')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Mean IC')
ax1.set_title('Predictive Performance by Batch Size')
ax1.axhline(y=0.0623, color='red', linestyle='--', label='Ridge IC')
ax1.legend()

ax2.bar([str(bs) for bs in batch_sizes_oos], times, color='#e74c3c', edgecolor='black')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Training Time by Batch Size')

plt.tight_layout()
plt.savefig(os.path.join('outputs', 'figures', 'sgd_batch_comparison.png'), dpi=150)
print("Saved sgd_batch_comparison.png")
plt.close()

# ============================================================
# Final comparison
# ============================================================
print("\n" + "=" * 70)
print("FULL MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'IC/AUC':<10} {'Time (s)':<10}")
print("-" * 45)
print(f"{'OLS (Fama-MacBeth)':<25} {'0.0602':<10} {'0.24':<10}")
print(f"{'Ridge':<25} {'0.0623':<10} {'3.83':<10}")
print(f"{'Lasso':<25} {'0.0512':<10} {'3.45':<10}")
print(f"{'Logistic':<25} {'0.5440':<10} {'2.56':<10}")
for bs in batch_sizes_oos:
    ic = sgd_results[bs]['ic_df']['IC'].mean()
    t = sgd_results[bs]['time']
    print(f"{'SGD (bs=' + str(bs) + ')':<25} {ic:<10.4f} {t:<10.2f}")

print("\nDone!")