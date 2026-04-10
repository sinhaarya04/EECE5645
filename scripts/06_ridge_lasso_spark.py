"""
06_ridge_lasso_pyspark.py
Ridge (L2) and Lasso (L1) regression with expanding-window cross-validation.
Fully parallelized via PySpark — temporal ordering respected, no future leakage.
Benchmarks execution with 1, 2, 4, 8, 16 workers.

Input:  data/processed/panel_features.csv
Output: outputs/tables/regularized_results.csv
        outputs/tables/worker_scaling_benchmark.csv
        outputs/figures/lasso_coefficients.png
        outputs/figures/ridge_vs_lasso_ic.png
        outputs/figures/worker_scaling.png
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark import SparkConf

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
IN_PATH = os.path.join("data", "processed", "panel_features.csv")
os.makedirs(os.path.join("outputs", "tables"), exist_ok=True)
os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

FEATURES = [
    "PE_RATIO", "PX_TO_SALES_RATIO", "CURRENT_EV_TO_T12M_EBITDA",
    "PX_TO_FREE_CASH_FLOW", "EQY_DVD_YLD_12M", "CUR_MKT_CAP",
    "RETURN_ON_ASSET", "RETURN_COM_EQY", "GROSS_MARGIN", "OPER_MARGIN",
    "TOT_DEBT_TO_TOT_EQY", "CUR_RATIO", "BETA_RAW_OVERRIDABLE",
    "EARN_YLD", "VOLATILITY_90D", "VOLUME_AVG_30D",
    "MOMENTUM_12_1", "REVERSAL_3M", "HIGH_52W_RATIO",
]
TARGET = "FWD_RETURN"
MIN_TRAIN_MONTHS = 60
ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
WORKER_COUNTS = [1, 2, 4, 8, 16]


# ==================================================================
# Helper: build a local Spark session with a given number of cores
# ==================================================================
def get_spark(n_workers: int) -> SparkSession:
    """Return a *local* SparkSession using ``n_workers`` cores."""
    existing = SparkSession.getActiveSession()
    if existing is not None:
        existing.stop()

    conf = (
        SparkConf()
        .setMaster(f"local[{n_workers}]")
        .setAppName(f"RidgeLasso-{n_workers}w")
        .set("spark.driver.memory", "8g")
        .set("spark.sql.shuffle.partitions", str(max(n_workers * 2, 4)))
        .set("spark.default.parallelism", str(n_workers))
        .set("spark.ui.showConsoleProgress", "false")
        .set("spark.log.level", "WARN")
    )
    return SparkSession.builder.config(conf=conf).getOrCreate()


# ==================================================================
# Pre-group data by date for efficient slicing inside workers
# ==================================================================
def pregroup_data(pdf, features, target):
    """
    Convert the panel into a dict: date_str -> numpy arrays (X, y).
    This is much smaller to broadcast than the raw DataFrame and
    avoids repeated .loc[] filtering inside each worker.
    """
    cols = features + [target]
    grouped = {}
    for date_str, grp in pdf.groupby("_date_str"):
        sub = grp[cols].dropna()
        if len(sub) > 0:
            grouped[date_str] = (sub[features].values, sub[target].values)
    return grouped


# ==================================================================
# Core worker function — references broadcast vars, NOT task data
# ==================================================================
def make_fit_fold(bc_grouped, bc_dates, bc_features):
    """
    Return a closure that captures broadcast references.
    The task tuple is lightweight: (model_type, alpha, date_idx, mode).
    """
    def fit_fold(task):
        model_type, alpha, date_idx, mode = task

        grouped = bc_grouped.value
        dates_list = bc_dates.value
        features = bc_features.value

        # Build training arrays by concatenating all dates < date_idx
        test_date = dates_list[date_idx]
        train_dates = dates_list[:date_idx]

        X_parts, y_parts = [], []
        for d in train_dates:
            if d in grouped:
                X_parts.append(grouped[d][0])
                y_parts.append(grouped[d][1])

        if not X_parts:
            return None

        X_tr = np.concatenate(X_parts)
        y_tr = np.concatenate(y_parts)

        if test_date not in grouped:
            return None
        X_te, y_te = grouped[test_date]

        if len(X_tr) < 100 or len(X_te) < 20:
            return None

        ModelClass = Ridge if model_type == "ridge" else Lasso
        model = ModelClass(alpha=alpha, max_iter=10000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        ic = stats.spearmanr(y_pred, y_te)[0]

        result = {
            "model_type": model_type,
            "alpha": alpha,
            "date_idx": date_idx,
            "date": test_date,
            "ic": ic,
            "coefs": dict(zip(features, model.coef_.tolist())),
        }

        if mode == "oos":
            # Store predictions as lightweight arrays (not a DataFrame)
            result["y_pred"] = y_pred.tolist()
            result["y_true"] = y_te.tolist()
            result["X_test"] = X_te.tolist()

        return result

    return fit_fold


# ==================================================================
# Expanding-window CV — parallelised over (alpha × date)
# ==================================================================
def expanding_window_cv_spark(spark, bc_grouped, bc_dates, bc_features,
                              model_type, alphas, target, n_val=12):
    sc = spark.sparkContext
    dates_list = bc_dates.value

    val_start = MIN_TRAIN_MONTHS
    val_end = val_start + n_val

    # Lightweight task tuples: no data, just indices
    tasks = []
    for alpha in alphas:
        for t in range(val_start, min(val_end + 12, len(dates_list) - 1)):
            tasks.append((model_type, alpha, t, "cv"))

    worker_fn = make_fit_fold(bc_grouped, bc_dates, bc_features)
    rdd = sc.parallelize(tasks, numSlices=max(len(tasks) // 4, 4))
    results = rdd.map(worker_fn).filter(lambda x: x is not None).collect()

    # Aggregate: mean IC per alpha
    from collections import defaultdict
    ic_acc = defaultdict(list)
    for r in results:
        ic_acc[r["alpha"]].append(r["ic"])
    avg_ic = {a: np.mean(v) if v else -999 for a, v in ic_acc.items()}
    best_alpha = max(avg_ic, key=avg_ic.get)
    return best_alpha, avg_ic


# ==================================================================
# Out-of-sample expanding window — parallelised over dates
# ==================================================================
def run_expanding_window_spark(spark, bc_grouped, bc_dates, bc_features,
                               model_type, alpha, features, target):
    sc = spark.sparkContext
    dates_list = bc_dates.value

    tasks = [
        (model_type, alpha, i, "oos")
        for i in range(MIN_TRAIN_MONTHS, len(dates_list) - 1)
    ]

    worker_fn = make_fit_fold(bc_grouped, bc_dates, bc_features)
    rdd = sc.parallelize(tasks, numSlices=max(len(tasks) // 2, 4))
    results = rdd.map(worker_fn).filter(lambda x: x is not None).collect()

    # Sort by date index
    results.sort(key=lambda r: r["date_idx"])

    ic_list = [{"Date": r["date"], "IC": r["ic"]} for r in results]
    coef_list = [r["coefs"] for r in results]

    # Reconstruct prediction DataFrames
    pred_frames = []
    for r in results:
        p = pd.DataFrame(r["X_test"], columns=features)
        p[target] = r["y_true"]
        p["PREDICTED"] = r["y_pred"]
        p["Date"] = r["date"]
        pred_frames.append(p)

    ic_df = pd.DataFrame(ic_list)
    coef_df = pd.DataFrame(coef_list)
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    return ic_df, coef_df, pred_df


# ==================================================================
# Full pipeline for one (model_type, n_workers) combination
# ==================================================================
def run_pipeline(pdf, grouped, dates_list, model_type, n_workers):
    spark = get_spark(n_workers)
    sc = spark.sparkContext

    # Broadcast the pre-grouped data (compact numpy arrays)
    bc_grouped = sc.broadcast(grouped)
    bc_dates = sc.broadcast(dates_list)
    bc_features = sc.broadcast(FEATURES)

    t0 = time.time()

    # --- CV ---
    t_cv = time.time()
    best_alpha, cv_scores = expanding_window_cv_spark(
        spark, bc_grouped, bc_dates, bc_features,
        model_type, ALPHAS, TARGET
    )
    cv_elapsed = time.time() - t_cv

    # --- OOS ---
    t_oos = time.time()
    ic_df, coef_df, pred_df = run_expanding_window_spark(
        spark, bc_grouped, bc_dates, bc_features,
        model_type, best_alpha, FEATURES, TARGET
    )
    oos_elapsed = time.time() - t_oos

    total_elapsed = time.time() - t0

    # Clean up broadcasts
    bc_grouped.unpersist()
    bc_dates.unpersist()
    bc_features.unpersist()
    spark.stop()

    return (ic_df, coef_df, pred_df, best_alpha, cv_scores,
            cv_elapsed, oos_elapsed, total_elapsed)


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":

    print("Loading data...")
    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    dates_list = sorted(df["_date_str"].unique().tolist())

    # Pre-group once — shared across all runs
    print("Pre-grouping data by date...")
    grouped = pregroup_data(df, FEATURES, TARGET)
    print(f"  {len(grouped)} date groups, "
          f"~{sum(x[0].nbytes + x[1].nbytes for x in grouped.values()) / 1e6:.1f} MB total")

    timing_rows = []
    best_ridge = None
    best_lasso = None

    for n_workers in WORKER_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARKING WITH {n_workers} WORKER(S)")
        print(f"{'=' * 70}")

        # ---- Ridge ----
        print(f"\n--- Ridge (L2) | workers={n_workers} ---")
        (ridge_ic, ridge_coefs, ridge_preds, ridge_alpha,
         ridge_cv, ridge_cv_t, ridge_oos_t, ridge_tot) = \
            run_pipeline(df, grouped, dates_list, "ridge", n_workers)

        print(f"  Best alpha : {ridge_alpha}")
        print(f"  Mean IC    : {ridge_ic['IC'].mean():.4f}")
        print(f"  IC Std     : {ridge_ic['IC'].std():.4f}")
        print(f"  IC > 0 pct : {(ridge_ic['IC'] > 0).mean():.2%}")
        ic_tstat = ridge_ic["IC"].mean() / (ridge_ic["IC"].std() / np.sqrt(len(ridge_ic)))
        print(f"  IC t-stat  : {ic_tstat:.3f}")
        print(f"  CV time    : {ridge_cv_t:.2f}s | OOS time: {ridge_oos_t:.2f}s | Total: {ridge_tot:.2f}s")

        timing_rows.append({
            "Model": "Ridge", "Workers": n_workers,
            "CV_Time_sec": round(ridge_cv_t, 2),
            "OOS_Time_sec": round(ridge_oos_t, 2),
            "Total_Time_sec": round(ridge_tot, 2),
        })
        best_ridge = (ridge_ic, ridge_coefs, ridge_preds,
                      ridge_alpha, ridge_cv)

        # ---- Lasso ----
        print(f"\n--- Lasso (L1) | workers={n_workers} ---")
        (lasso_ic, lasso_coefs, lasso_preds, lasso_alpha,
         lasso_cv, lasso_cv_t, lasso_oos_t, lasso_tot) = \
            run_pipeline(df, grouped, dates_list, "lasso", n_workers)

        print(f"  Best alpha : {lasso_alpha}")
        print(f"  Mean IC    : {lasso_ic['IC'].mean():.4f}")
        print(f"  IC Std     : {lasso_ic['IC'].std():.4f}")
        print(f"  IC > 0 pct : {(lasso_ic['IC'] > 0).mean():.2%}")
        ic_tstat = lasso_ic["IC"].mean() / (lasso_ic["IC"].std() / np.sqrt(len(lasso_ic)))
        print(f"  IC t-stat  : {ic_tstat:.3f}")
        print(f"  CV time    : {lasso_cv_t:.2f}s | OOS time: {lasso_oos_t:.2f}s | Total: {lasso_tot:.2f}s")

        timing_rows.append({
            "Model": "Lasso", "Workers": n_workers,
            "CV_Time_sec": round(lasso_cv_t, 2),
            "OOS_Time_sec": round(lasso_oos_t, 2),
            "Total_Time_sec": round(lasso_tot, 2),
        })
        best_lasso = (lasso_ic, lasso_coefs, lasso_preds,
                      lasso_alpha, lasso_cv)

    # ==============================================================
    # Unpack best results (from final / highest worker run)
    # ==============================================================
    ridge_ic, ridge_coefs, ridge_preds, best_ridge_alpha, ridge_cv_scores = best_ridge
    lasso_ic, lasso_coefs, lasso_preds, best_lasso_alpha, lasso_cv_scores = best_lasso

    # ==============================================================
    # Lasso factor selection — final model
    # ==============================================================
    print("\n" + "=" * 70)
    print("LASSO FACTOR SELECTION (Final Model)")
    print("=" * 70)
    final_coefs = lasso_coefs.iloc[-1]
    for f in FEATURES:
        coef = final_coefs[f]
        status = "KEPT" if abs(coef) > 1e-6 else "DROPPED"
        print(f"  {f:35s} {coef:+.6f}  [{status}]")
    kept = sum(1 for f in FEATURES if abs(final_coefs[f]) > 1e-6)
    print(f"\nLasso kept {kept}/{len(FEATURES)} factors, dropped {len(FEATURES) - kept}")

    # ==============================================================
    # Model comparison table
    # ==============================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    comparison = pd.DataFrame({
        "Model": ["OLS (Fama-MacBeth)", "Ridge", "Lasso"],
        "Mean_IC": [0.0602, ridge_ic["IC"].mean(), lasso_ic["IC"].mean()],
        "IC_Std": [0.1107, ridge_ic["IC"].std(), lasso_ic["IC"].std()],
        "IC_Hit_Rate": [0.7095, (ridge_ic["IC"] > 0).mean(), (lasso_ic["IC"] > 0).mean()],
    })
    print(comparison.to_string(index=False))
    comparison.to_csv(
        os.path.join("outputs", "tables", "regularized_results.csv"), index=False
    )

    # ==============================================================
    # Worker scaling benchmark table
    # ==============================================================
    timing_df = pd.DataFrame(timing_rows)
    print("\n" + "=" * 70)
    print("WORKER SCALING BENCHMARK")
    print("=" * 70)
    print(timing_df.to_string(index=False))
    timing_df.to_csv(
        os.path.join("outputs", "tables", "worker_scaling_benchmark.csv"), index=False
    )

    # ==============================================================
    # Plot 1: Lasso coefficient path over time
    # ==============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    for f in FEATURES:
        vals = lasso_coefs[f].values
        if np.any(np.abs(vals) > 1e-6):
            ax.plot(vals, label=f, linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Month (expanding window)")
    ax.set_ylabel("Lasso Coefficient")
    ax.set_title("Lasso Coefficient Evolution Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", "lasso_coefficients.png"), dpi=150)
    print("\nSaved lasso_coefficients.png")
    plt.close()

    # ==============================================================
    # Plot 2: Rolling IC comparison (Ridge vs Lasso)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    window = 12
    ridge_ic["IC_rolling"] = ridge_ic["IC"].rolling(window).mean()
    lasso_ic["IC_rolling"] = lasso_ic["IC"].rolling(window).mean()
    ax.plot(ridge_ic["Date"], ridge_ic["IC_rolling"], label="Ridge", linewidth=1.5)
    ax.plot(lasso_ic["Date"], lasso_ic["IC_rolling"], label="Lasso", linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("12-Month Rolling IC")
    ax.set_title("Ridge vs Lasso: Rolling Information Coefficient")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", "ridge_vs_lasso_ic.png"), dpi=150)
    print("Saved ridge_vs_lasso_ic.png")
    plt.close()

    # ==============================================================
    # Plot 3: Worker scaling chart
    # ==============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, model_name in zip(axes, ["Ridge", "Lasso"]):
        sub = timing_df[timing_df["Model"] == model_name]
        ax.plot(sub["Workers"], sub["Total_Time_sec"], "o-", linewidth=2,
                label="Total", color="tab:blue")
        ax.plot(sub["Workers"], sub["CV_Time_sec"], "s--", linewidth=1.5,
                label="CV only", color="tab:orange")
        ax.plot(sub["Workers"], sub["OOS_Time_sec"], "^--", linewidth=1.5,
                label="OOS only", color="tab:green")

        # Ideal linear speed-up reference line
        base = sub["Total_Time_sec"].iloc[0]
        ideal = [base / w for w in sub["Workers"]]
        ax.plot(sub["Workers"], ideal, ":", color="grey", linewidth=1,
                label="Ideal linear")

        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"{model_name} — Scaling")
        ax.set_xticks(WORKER_COUNTS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", "worker_scaling.png"), dpi=150)
    print("Saved worker_scaling.png")
    plt.close()

    print("\nDone!")