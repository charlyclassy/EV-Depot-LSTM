#!/usr/bin/env python3
"""
EPA report plots: Previous-day baseline vs Random Forest vs XGBoost vs LSTM.

Expected input columns:
    timestamp, depot_id, demand_kw, observed

Expected file:
    /workspace/UKPN_OptimisePrime_Bexleyheath.csv
    or UKPN_OptimisePrime_Bexleyheath.csv beside this script.

Outputs are written to:
    report_figures_rf_xgb_lstm_final/

Install:
    pip install pandas numpy matplotlib scikit-learn xgboost tensorflow
"""

from __future__ import annotations

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

INPUT_CANDIDATES = [
    Path("/workspace/UKPN_OptimisePrime_Bexleyheath.csv"),
    Path("UKPN_OptimisePrime_Bexleyheath.csv"),
]
OUTPUT_DIR = Path("report_figures_rf_xgb_lstm_final")

DATA_START = pd.Timestamp("2021-07-08 00:00:00")
TRAIN_END = pd.Timestamp("2022-04-01 00:00:00")
VALID_END = pd.Timestamp("2022-05-10 00:00:00")
TEST_END = pd.Timestamp("2022-07-26 00:00:00")

PLANNING_CAPACITY_KW = 17.0
WARNING_THRESHOLD_KW = 0.8 * PLANNING_CAPACITY_KW
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 672
LSTM_LOOKBACK = 96
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 256

FEATURE_COLS = [
    "hour_sin", "hour_cos", "day_of_week", "weekend_flag", "month",
    "Lag_1", "Lag_4", "Lag_96", "Lag_672",
    "Rolling_mean_4", "Rolling_mean_24", "Rolling_mean_96", "Rolling_std_96",
]

SHORT_LABELS = {
    "Previous-day same-time": "Baseline",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
}


def locate_input() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find UKPN_OptimisePrime_Bexleyheath.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "demand_kw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["demand_kw"] = pd.to_numeric(df["demand_kw"], errors="coerce")
    df = df.dropna(subset=["timestamp", "demand_kw"]).copy()
    df["demand_kw"] = df["demand_kw"].clip(lower=0)

    df = (
        df.groupby("timestamp", as_index=False)["demand_kw"].mean()
        .set_index("timestamp").sort_index()
    )
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    df["demand_kw"] = df["demand_kw"].interpolate(method="time").ffill().bfill()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    hour_float = idx.hour + idx.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    out["day_of_week"] = idx.dayofweek.astype(int)
    out["weekend_flag"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month.astype(int)

    y = out["demand_kw"]
    out["Lag_1"] = y.shift(1)
    out["Lag_4"] = y.shift(4)
    out["Lag_96"] = y.shift(96)
    out["Lag_672"] = y.shift(672)
    shifted = y.shift(1)
    out["Rolling_mean_4"] = shifted.rolling(4).mean()
    out["Rolling_mean_24"] = shifted.rolling(24).mean()
    out["Rolling_mean_96"] = shifted.rolling(96).mean()
    out["Rolling_std_96"] = shifted.rolling(96).std()
    return out


def operational_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual_breach = y_true >= PLANNING_CAPACITY_KW
    predicted_breach = y_pred >= PLANNING_CAPACITY_KW

    tp = int(np.sum(actual_breach & predicted_breach))
    fp = int(np.sum((~actual_breach) & predicted_breach))
    fn = int(np.sum(actual_breach & (~predicted_breach)))
    tn = int(np.sum((~actual_breach) & (~predicted_breach)))

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else np.nan
    balanced = (recall + specificity) / 2 if np.isfinite(recall) and np.isfinite(specificity) else np.nan

    return {
        "MAE_kW": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Breach_Recall": recall,
        "False_Alarm_Ratio": far,
        "Breach_Precision": precision,
        "F1_Score": f1,
        "Specificity": specificity,
        "Balanced_Accuracy": balanced,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def build_lstm_sequences(scaled_values, index, target_times, lookback):
    position = pd.Series(np.arange(len(index)), index=index)
    X, times = [], []
    for ts in target_times:
        pos = int(position.loc[ts])
        if pos < lookback:
            continue
        X.append(scaled_values[pos-lookback:pos])
        times.append(ts)
    X = np.asarray(X, dtype=np.float32).reshape(-1, lookback, 1)
    return X, pd.DatetimeIndex(times)


def recursive_lstm_forecast(model, x_scaler, y_scaler, observed_history_kw, start_timestamp, steps=96):
    """Create a genuine recursive future forecast without future actual demand."""
    history_kw = list(np.asarray(observed_history_kw, dtype=float))
    forecasts = []
    for _ in range(steps):
        recent_kw = np.asarray(history_kw[-LSTM_LOOKBACK:], dtype=float).reshape(-1, 1)
        recent_scaled = x_scaler.transform(recent_kw).reshape(1, LSTM_LOOKBACK, 1)
        pred_scaled = model.predict(recent_scaled, verbose=0).reshape(-1, 1)
        pred_kw = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
        pred_kw = max(0.0, pred_kw)
        forecasts.append(pred_kw)
        history_kw.append(pred_kw)
    future_times = pd.date_range(start=start_timestamp, periods=steps, freq="15min")
    return future_times, np.asarray(forecasts, dtype=float)


def savefig(fig, name, bottom=None):
    if bottom is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(bottom=bottom)
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)


def confusion_plot(model_name, row, filename):
    matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted No Breach", "Predicted Breach"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No Breach", "Actual Breach"])
    ax.set_title(f"{model_name} — Capacity-Breach Confusion Matrix\nPlanning Capacity Threshold = {PLANNING_CAPACITY_KW:.1f} kW")
    names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max()/2 else "black"
            ax.text(j, i, f"{names[i][j]}\n{int(matrix[i,j]):,}", ha="center", va="center", fontweight="bold", fontsize=12, color=color)
    fig.colorbar(im, ax=ax, label="Number of 15-minute intervals")
    savefig(fig, filename)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("EPA comparison: Random Forest vs XGBoost vs LSTM")
    path = locate_input()
    print("Input:", path.resolve())

    raw = load_data(path)
    featured = add_features(raw)
    modelling = featured.loc[(featured.index >= DATA_START) & (featured.index < TEST_END)].dropna(subset=FEATURE_COLS + ["demand_kw"])

    train = modelling[(modelling.index >= DATA_START) & (modelling.index < TRAIN_END)].copy()
    valid = modelling[(modelling.index >= TRAIN_END) & (modelling.index < VALID_END)].copy()
    test = modelling[(modelling.index >= VALID_END) & (modelling.index < TEST_END)].copy()

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    print(f"Held-out rows: {len(test):,}")
    if len(test) != 7392:
        raise ValueError(f"Expected 7,392 held-out rows, got {len(test):,}")

    X_train, y_train = train[FEATURE_COLS], train["demand_kw"]
    X_valid, y_valid = valid[FEATURE_COLS], valid["demand_kw"]
    X_test, y_test = test[FEATURE_COLS], test["demand_kw"]
    dev_X = pd.concat([X_train, X_valid])
    dev_y = pd.concat([y_train, y_valid])

    baseline_pred = test["Lag_96"].to_numpy(float)

    print("\n[1/3] Random Forest")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        max_features="sqrt", random_state=SEED, n_jobs=-1,
    )
    t0 = time.perf_counter(); rf.fit(dev_X, dev_y); rf_seconds = time.perf_counter() - t0
    rf_pred = np.clip(rf.predict(X_test), 0, None)
    print(f"Training time: {rf_seconds:.2f}s")

    print("\n[2/3] XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=340, max_depth=7, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="reg:squarederror", tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )
    t0 = time.perf_counter(); xgb_model.fit(dev_X, dev_y, verbose=False); xgb_seconds = time.perf_counter() - t0
    xgb_pred = np.clip(xgb_model.predict(X_test), 0, None)
    print(f"Training time: {xgb_seconds:.2f}s")

    print("\n[3/3] LSTM")
    development_series = raw.loc[(raw.index >= DATA_START) & (raw.index < VALID_END), ["demand_kw"]]
    full_series = raw.loc[(raw.index >= DATA_START - pd.Timedelta(days=7)) & (raw.index < TEST_END), ["demand_kw"]]

    x_scaler = StandardScaler().fit(development_series[["demand_kw"]])
    y_scaler = StandardScaler().fit(development_series[["demand_kw"]])
    scaled_full = x_scaler.transform(full_series[["demand_kw"]]).ravel()

    X_lstm_train, t_train = build_lstm_sequences(scaled_full, full_series.index, train.index, LSTM_LOOKBACK)
    X_lstm_valid, t_valid = build_lstm_sequences(scaled_full, full_series.index, valid.index, LSTM_LOOKBACK)
    X_lstm_test, t_test = build_lstm_sequences(scaled_full, full_series.index, test.index, LSTM_LOOKBACK)
    if not t_test.equals(test.index):
        raise ValueError("LSTM test timestamps are not identical to common held-out timestamps")

    y_lstm_train = y_scaler.transform(raw.loc[t_train, ["demand_kw"]]).ravel().astype(np.float32)
    y_lstm_valid = y_scaler.transform(raw.loc[t_valid, ["demand_kw"]]).ravel().astype(np.float32)

    lstm = keras.Sequential([
        keras.layers.Input(shape=(LSTM_LOOKBACK, 1)),
        keras.layers.LSTM(
            48, activation="tanh", recurrent_activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
            recurrent_initializer=keras.initializers.Orthogonal(seed=SEED),
        ),
        keras.layers.Dense(24, activation="relu", kernel_initializer=keras.initializers.HeUniform(seed=SEED)),
        keras.layers.Dense(1),
    ])
    lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-5, restore_best_weights=True)
    t0 = time.perf_counter()
    history = lstm.fit(
        X_lstm_train, y_lstm_train,
        validation_data=(X_lstm_valid, y_lstm_valid),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        shuffle=False, verbose=1, callbacks=[early],
    )
    lstm_seconds = time.perf_counter() - t0
    lstm_scaled = lstm.predict(X_lstm_test, batch_size=LSTM_BATCH_SIZE, verbose=0)
    lstm_pred = np.clip(y_scaler.inverse_transform(lstm_scaled).ravel(), 0, None)
    print(f"Training time: {lstm_seconds:.2f}s; epochs: {len(history.history['loss'])}")

    actual = y_test.to_numpy(float)
    timestamps = test.index
    assert len(actual) == len(baseline_pred) == len(rf_pred) == len(xgb_pred) == len(lstm_pred)

    models = {
        "Previous-day same-time": baseline_pred,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "LSTM": lstm_pred,
    }
    roles = {
        "Previous-day same-time": "Operational baseline",
        "Random Forest": "Machine-learning alternative",
        "XGBoost": "Machine-learning alternative",
        "LSTM": "Deep-learning alternative",
    }
    train_times = {
        "Previous-day same-time": 0.0,
        "Random Forest": rf_seconds,
        "XGBoost": xgb_seconds,
        "LSTM": lstm_seconds,
    }

    rows = []
    for name, pred in models.items():
        row = {"Model": name, "Role": roles[name], "Training_Time_Seconds": train_times[name]}
        row.update(operational_metrics(actual, pred))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    pd.DataFrame({
        "timestamp": timestamps,
        "actual_kw": actual,
        "baseline_kw": baseline_pred,
        "random_forest_kw": rf_pred,
        "xgboost_kw": xgb_pred,
        "lstm_kw": lstm_pred,
    }).to_csv(OUTPUT_DIR / "heldout_predictions_all_models.csv", index=False)

    pd.DataFrame([{
        "Depot": "Bexleyheath",
        "Data_Source": "UK Power Networks Optimise Prime",
        "Sampling_Interval": "15 minutes",
        "Planning_Capacity_Threshold_kW": PLANNING_CAPACITY_KW,
        "Warning_Threshold_kW": WARNING_THRESHOLD_KW,
        "Training_Rows": len(train), "Validation_Rows": len(valid), "Heldout_Test_Rows": len(test),
        "Training_Start": train.index.min(), "Training_End": train.index.max(),
        "Validation_Start": valid.index.min(), "Validation_End": valid.index.max(),
        "Test_Start": test.index.min(), "Test_End": test.index.max(),
    }]).to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    trained = metrics[metrics["Model"] != "Previous-day same-time"].copy()
    criteria = [
        ("Lowest MAE", "MAE_kW", "min"),
        ("Lowest RMSE", "RMSE_kW", "min"),
        ("Highest R²", "R2", "max"),
        ("Highest breach recall", "Breach_Recall", "max"),
        ("Lowest false alarm ratio", "False_Alarm_Ratio", "min"),
        ("Highest F1-score", "F1_Score", "max"),
        ("Highest balanced accuracy", "Balanced_Accuracy", "max"),
        ("Fastest training", "Training_Time_Seconds", "min"),
    ]
    decisions = []
    for criterion, col, direction in criteria:
        idx = trained[col].idxmin() if direction == "min" else trained[col].idxmax()
        decisions.append({"Criterion": criterion, "Winner": metrics.loc[idx, "Model"], "Winning_Value": metrics.loc[idx, col]})
    decision = pd.DataFrame(decisions)
    decision.to_csv(OUTPUT_DIR / "model_selection_decision.csv", index=False)

    print("\nHELD-OUT RESULTS")
    print(metrics[["Model","MAE_kW","RMSE_kW","R2","Breach_Recall","False_Alarm_Ratio","F1_Score","Balanced_Accuracy","Training_Time_Seconds","TP","FP","FN"]].to_string(index=False))
    print("\nMETRIC WINNERS")
    print(decision.to_string(index=False))

    # 01 Historical demand
    hist = raw.loc[(raw.index >= pd.Timestamp("2021-07-01")) & (raw.index < TEST_END), "demand_kw"]
    peak_ts, peak_kw = hist.idxmax(), float(hist.max())
    fig, ax = plt.subplots(figsize=(14,5.5))
    ax.plot(hist.index, hist.values, lw=0.8, label="Observed Demand")
    ax.axhline(WARNING_THRESHOLD_KW, ls="--", lw=1.4, label=f"Warning Threshold ({WARNING_THRESHOLD_KW:.1f} kW)")
    ax.axhline(PLANNING_CAPACITY_KW, ls="--", lw=1.4, label=f"Planning Capacity Threshold ({PLANNING_CAPACITY_KW:.1f} kW)")
    ax.scatter([peak_ts],[peak_kw], marker="D", s=50, label=f"Observed Peak ({peak_kw:.1f} kW)")
    ax.set(title="Historical Observed Demand Profile", xlabel="Timestamp", ylabel="Demand (kW)")
    ax.grid(alpha=.25); ax.legend(ncol=2, loc="upper left")
    fig.text(.08,.015,f"Observed period: {hist.index.min():%d %b %Y} to {hist.index.max():%d %b %Y}. Peak: {peak_kw:.2f} kW at {peak_ts:%d %b %Y %H:%M}.", fontsize=9)
    savefig(fig,"01_historical_demand_profile.png",.11)

    # 02 Split timeline
    fig, ax = plt.subplots(figsize=(14,4.8))
    ax.axvspan(DATA_START,TRAIN_END,alpha=.18,label="Training")
    ax.axvspan(TRAIN_END,VALID_END,alpha=.18,label="Validation")
    ax.axvspan(VALID_END,TEST_END,alpha=.18,label="Held-out test")
    ax.axvline(TRAIN_END,ls="--",lw=1); ax.axvline(VALID_END,ls="--",lw=1)
    ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title("Chronological Data Split — Development Periods and Untouched Final Test"); ax.set_xlabel("Date")
    for a,b,label,count in [(DATA_START,TRAIN_END,"Training",len(train)),(TRAIN_END,VALID_END,"Validation",len(valid)),(VALID_END,TEST_END,"Held-out test",len(test))]:
        mid=a+(b-a)/2
        ax.text(mid,.52,f"{label}\n{count:,} rows",ha="center",va="center",fontweight="bold",bbox=dict(boxstyle="round,pad=.4",facecolor="white",alpha=.85))
    fig.text(.02,.015,"Training and validation are used for model development; the held-out test is not used for fitting.",fontsize=9)
    savefig(fig,"02_data_split_timeline.png",.13)

    labels = metrics["Model"].tolist(); short_labels = [SHORT_LABELS[m] for m in labels]; x = np.arange(len(labels)); width=.36

    # 03 Error comparison
    fig, ax = plt.subplots(figsize=(11.5,5.6))
    b1=ax.bar(x-width/2,metrics["MAE_kW"],width,label="MAE"); b2=ax.bar(x+width/2,metrics["RMSE_kW"],width,label="RMSE")
    bar_labels(ax,b1); bar_labels(ax,b2); ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=11); ax.set_ylabel("Error (kW) — lower is better"); ax.set_title("Forecast Error Comparison — Three Models vs Operational Baseline"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.text(.08,.015,"Baseline = previous-day same-time demand.",fontsize=10); savefig(fig,"03_error_comparison.png",.12)

    # 04 Operational comparison
    recall=metrics["Breach_Recall"]*100; far=metrics["False_Alarm_Ratio"]*100
    fig, ax = plt.subplots(figsize=(11.5,5.6))
    b1=ax.bar(x-width/2,recall,width,label="Breach Recall"); b2=ax.bar(x+width/2,far,width,label="False Alarm Ratio")
    bar_labels(ax,b1,"{:.1f}%"); bar_labels(ax,b2,"{:.1f}%"); ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=11); ax.set_ylabel("Percentage (%)"); ax.set_title("Operational Capacity-Breach Comparison"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.text(.08,.015,"Higher breach recall is better; lower false alarm ratio is better. Baseline = previous-day same-time.",fontsize=9); savefig(fig,"04_operational_comparison.png",.12)

    # 05 R2
    fig, ax = plt.subplots(figsize=(10.5,5.4)); bars=ax.bar(short_labels,metrics["R2"])
    for b,v in zip(bars,metrics["R2"]): ax.text(b.get_x()+b.get_width()/2,v+(0.015 if v>=0 else -0.04),f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top")
    ax.axhline(0,ls="--",lw=1); ax.set_ylabel("R² — higher is better"); ax.set_title("Held-Out R² Comparison Across Four Approaches"); ax.grid(axis="y",alpha=.25)
    fig.text(.08,.015,"Baseline = previous-day same-time demand.",fontsize=10); savefig(fig,"05_r2_comparison.png",.12)

    # 06 Breach counts
    w=.25; fig, ax=plt.subplots(figsize=(11.5,5.6))
    b1=ax.bar(x-w,metrics["TP"],w,label="True Positive (TP)"); b2=ax.bar(x,metrics["FP"],w,label="False Positive (FP)"); b3=ax.bar(x+w,metrics["FN"],w,label="False Negative (FN)")
    bar_labels(ax,b1,"{:.0f}"); bar_labels(ax,b2,"{:.0f}"); bar_labels(ax,b3,"{:.0f}"); ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=11); ax.set_ylabel("Held-out interval count"); ax.set_title("Capacity-Breach Detection Counts Across Four Approaches"); ax.legend(ncol=3); ax.grid(axis="y",alpha=.25)
    fig.text(.08,.015,"TP = actual breach correctly identified; FP = false alarm; FN = actual breach missed. Baseline = previous-day same-time.",fontsize=9)
    savefig(fig,"06_breach_counts.png",.12)

    # 07 F1 lollipop
    f1=metrics["F1_Score"]*100; fig,ax=plt.subplots(figsize=(10.5,5.4))
    for i,v in enumerate(f1): ax.vlines(i,0,v,lw=2); ax.scatter(i,v,s=110,zorder=3); ax.text(i,v+2,f"{v:.1f}%",ha="center")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=11); ax.set_ylim(0,100); ax.set_ylabel("F1-score (%) — higher is better"); ax.set_title("Capacity-Breach F1-Score Across Four Approaches"); ax.grid(axis="y",alpha=.25)
    savefig(fig,"07_f1_score.png")

    # 08 Balanced accuracy
    ba=metrics["Balanced_Accuracy"]*100; ypos=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(10.5,5.4))
    for y,v in zip(ypos,ba): ax.hlines(y,50,v,lw=2); ax.scatter(v,y,s=110,zorder=3); ax.text(v+.8,y,f"{v:.1f}%",va="center")
    ax.set_yticks(ypos); ax.set_yticklabels(short_labels, fontsize=11); ax.set_xlim(45,100); ax.invert_yaxis(); ax.set_xlabel("Balanced accuracy (%) — higher is better"); ax.set_title("Balanced Accuracy for Capacity-Breach Detection"); ax.grid(axis="x",alpha=.25)
    savefig(fig,"08_balanced_accuracy.png")

    # 09 Training time
    tm=metrics[metrics["Model"]!="Previous-day same-time"]
    fig,ax=plt.subplots(figsize=(9.5,5.2)); bars=ax.bar(tm["Model"],tm["Training_Time_Seconds"]); bar_labels(ax,bars,"{:.1f}s"); ax.set_ylabel("Training time (seconds) — lower is better"); ax.set_title("Model Training-Time Comparison"); ax.grid(axis="y",alpha=.25)
    fig.text(.08,.015,"Training time is measured on the current machine and is included as an operational trade-off.",fontsize=9)
    savefig(fig,"09_training_time.png",.12)

    # 10 7-day all-model detail
    mask=timestamps >= timestamps.max()-pd.Timedelta(days=7)
    fig,ax=plt.subplots(figsize=(13.5,6.1)); ax.plot(timestamps[mask],actual[mask],lw=2,label="Actual demand"); ax.plot(timestamps[mask],rf_pred[mask],lw=1.4,label="Random Forest"); ax.plot(timestamps[mask],xgb_pred[mask],lw=1.4,label="XGBoost"); ax.plot(timestamps[mask],lstm_pred[mask],lw=1.4,label="LSTM"); ax.plot(timestamps[mask],baseline_pred[mask],lw=1.1,ls="--",label="Previous-day baseline")
    ax.axhline(WARNING_THRESHOLD_KW,ls="--",lw=1,label=f"Warning threshold ({WARNING_THRESHOLD_KW:.1f} kW)"); ax.axhline(PLANNING_CAPACITY_KW,ls="--",lw=1,label=f"Planning capacity threshold ({PLANNING_CAPACITY_KW:.1f} kW)")
    ax.set_title("Actual Demand vs All Forecasting Approaches — 7-Day Held-Out Detail"); ax.set_ylabel("Demand (kW)"); ax.legend(ncol=3,loc="upper center"); ax.grid(alpha=.25); fig.autofmt_xdate()
    fig.text(.08,.018,"Note: All lines use identical held-out timestamps. The 7-day window is shown only for readability; evaluation metrics use the full held-out test period.",fontsize=9)
    savefig(fig,"10_actual_vs_all_models_7day.png",.18)

    # 11 Residual boxplot
    residuals=[actual-baseline_pred,actual-rf_pred,actual-xgb_pred,actual-lstm_pred]
    fig,ax=plt.subplots(figsize=(10.5,5.6)); ax.boxplot(residuals,tick_labels=short_labels,showmeans=True,showfliers=False); ax.axhline(0,ls="--",lw=1); ax.set_ylabel("Residual (Actual − Predicted) kW"); ax.set_title("Held-Out Residual Distribution Across Four Approaches"); ax.grid(axis="y",alpha=.25)
    savefig(fig,"11_residual_boxplot.png")

    # 12 Metric winner scorecard
    trained_names=tm["Model"].tolist(); score=np.zeros((len(criteria),len(trained_names)))
    for r,row in decision.iterrows(): score[r,trained_names.index(row["Winner"])]=1
    fig,ax=plt.subplots(figsize=(9.5,6.4)); ax.imshow(score,cmap="Blues",vmin=0,vmax=1,aspect="auto"); ax.set_xticks(np.arange(len(trained_names))); ax.set_xticklabels(trained_names); ax.set_yticks(np.arange(len(criteria))); ax.set_yticklabels([c[0] for c in criteria]); ax.set_title("Metric-Winner Trade-Off Scorecard"); ax.set_xlabel("Forecasting model")
    for r in range(score.shape[0]):
        for c in range(score.shape[1]): ax.text(c,r,"WIN" if score[r,c] else "—",ha="center",va="center",fontweight="bold" if score[r,c] else "normal")
    fig.text(.08,.012,"Metric winners are shown above. Final model selection also considers training cost, maintainability, retraining requirements and operational suitability.",fontsize=9)
    savefig(fig,"12_metric_winner_scorecard.png",.14)

    # 13/14 Feature importance
    for model_name, imp, filename in [
        ("Random Forest", rf.feature_importances_, "13_random_forest_feature_importance.png"),
        ("XGBoost", xgb_model.feature_importances_, "14_xgboost_feature_importance.png"),
    ]:
        s=pd.Series(imp,index=FEATURE_COLS).sort_values(); fig,ax=plt.subplots(figsize=(9.5,6)); ax.barh(s.index,s.values); ax.set_xlabel("Relative importance"); ax.set_ylabel("Feature"); ax.set_title(f"{model_name} — Feature Importance"); ax.grid(axis="x",alpha=.25); savefig(fig,filename)

    confusion_plot("Random Forest", metrics.loc[metrics["Model"]=="Random Forest"].iloc[0], "15_random_forest_confusion_matrix.png")
    confusion_plot("XGBoost", metrics.loc[metrics["Model"]=="XGBoost"].iloc[0], "16_xgboost_confusion_matrix.png")
    confusion_plot("LSTM", metrics.loc[metrics["Model"]=="LSTM"].iloc[0], "17_lstm_confusion_matrix.png")

    # 18 LSTM actual vs predicted scatter
    fig,ax=plt.subplots(figsize=(8.5,8.0))
    ax.scatter(actual,lstm_pred,alpha=.35,s=16)
    upper=max(float(actual.max()),float(lstm_pred.max()))
    ax.plot([0,upper],[0,upper],ls="--",lw=1.5,label="Ideal: Predicted = Actual")
    ax.set_xlim(0,upper); ax.set_ylim(0,upper)
    ax.set_xlabel("Actual Demand (kW)"); ax.set_ylabel("Predicted Demand (kW)")
    ax.set_title("LSTM — Actual vs Predicted Demand"); ax.legend(loc="upper left"); ax.grid(alpha=.25)
    savefig(fig,"18_lstm_actual_vs_predicted_scatter.png")

    # 19 LSTM residual histogram — central 98% detail
    lstm_residuals=actual-lstm_pred
    q_low=float(np.quantile(lstm_residuals,.01)); q_high=float(np.quantile(lstm_residuals,.99))
    visible=(lstm_residuals>=q_low)&(lstm_residuals<=q_high)
    visible_res=lstm_residuals[visible]
    mean_res=float(np.mean(lstm_residuals)); median_res=float(np.median(lstm_residuals))
    fig,ax=plt.subplots(figsize=(10.5,6.4))
    ax.hist(visible_res,bins=45,alpha=.85,label="Central 98% of residuals")
    ax.axvline(0,ls="--",lw=1.5,label="Zero error")
    ax.axvline(mean_res,ls=":",lw=1.7,label=f"Mean: {mean_res:.2f} kW")
    ax.axvline(median_res,ls="-.",lw=1.7,label=f"Median: {median_res:.2f} kW")
    ax.text(.02,.96,f"Visible rows: {visible.sum():,} of {len(lstm_residuals):,}\nOutliers outside view: {len(lstm_residuals)-visible.sum():,}",transform=ax.transAxes,ha="left",va="top",fontsize=10,bbox=dict(boxstyle="round,pad=.35",facecolor="white",alpha=.9))
    ax.set_xlabel("Residual (Actual − Predicted) kW"); ax.set_ylabel("Frequency")
    ax.set_title("LSTM Held-Out Residual Distribution — Central 98% Detail"); ax.legend(loc="upper right"); ax.grid(axis="y",alpha=.25)
    savefig(fig,"19_lstm_residual_distribution_central_98.png")

    # 20 Full held-out actual vs LSTM prediction
    fig,ax=plt.subplots(figsize=(15,6.4))
    ax.plot(timestamps,actual,lw=1.2,label="Actual Demand")
    ax.plot(timestamps,lstm_pred,lw=1.0,label="LSTM Prediction")
    ax.plot(timestamps,baseline_pred,lw=.9,alpha=.8,label="Previous-Day Baseline")
    ax.axhline(WARNING_THRESHOLD_KW,ls="--",lw=1.2,label=f"Warning Threshold ({WARNING_THRESHOLD_KW:.1f} kW)")
    ax.axhline(PLANNING_CAPACITY_KW,ls="--",lw=1.2,label=f"Planning Capacity Threshold ({PLANNING_CAPACITY_KW:.1f} kW)")
    ax.set_title("Held-Out Test: Actual vs LSTM Predicted Demand"); ax.set_xlabel("Timestamp"); ax.set_ylabel("Demand (kW)")
    ax.legend(ncol=3,loc="upper left"); ax.grid(alpha=.25)
    savefig(fig,"20_lstm_heldout_actual_vs_predicted_full.png")

    # 21 LSTM training vs validation loss
    train_loss=history.history["loss"]; val_loss=history.history["val_loss"]; epochs=np.arange(1,len(train_loss)+1)
    best_epoch=int(np.argmin(val_loss)+1); best_val=float(np.min(val_loss))
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    ax.plot(epochs,train_loss,marker="o",lw=1.5,label="Training loss")
    ax.plot(epochs,val_loss,marker="o",lw=1.5,label="Validation loss")
    ax.axvline(best_epoch,ls="--",lw=1,label=f"Best validation epoch = {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Squared Error loss"); ax.set_title("LSTM Training and Validation Loss")
    ax.legend(); ax.grid(alpha=.25)
    fig.text(.08,.015,f"Minimum validation loss = {best_val:.6f}. Early stopping restores the best validation weights.",fontsize=9)
    savefig(fig,"21_lstm_training_validation_loss.png",.12)

    # 22 Genuine 24-hour recursive LSTM forecast
    observed_history=raw.loc[raw.index<TEST_END,"demand_kw"]
    future_times,future_lstm=recursive_lstm_forecast(lstm,x_scaler,y_scaler,observed_history.values,TEST_END,96)
    pd.DataFrame({"timestamp":future_times,"lstm_forecast_kw":future_lstm}).to_csv(OUTPUT_DIR/"lstm_24h_recursive_forecast.csv",index=False)
    context=raw.loc[(raw.index>=TEST_END-pd.Timedelta(days=2))&(raw.index<TEST_END),"demand_kw"]
    fig,ax=plt.subplots(figsize=(14,6.2))
    ax.plot(context.index,context.values,lw=1.7,label="Observed Demand")
    ax.plot(future_times,future_lstm,ls="--",lw=1.8,label="24-Hour LSTM Recursive Forecast")
    ax.axhline(WARNING_THRESHOLD_KW,ls="--",lw=1.2,label=f"Warning Threshold ({WARNING_THRESHOLD_KW:.1f} kW)")
    ax.axhline(PLANNING_CAPACITY_KW,ls="--",lw=1.2,label=f"Planning Capacity Threshold ({PLANNING_CAPACITY_KW:.1f} kW)")
    ax.axvline(TEST_END,ls=":",lw=1.5,label="Forecast Start")
    ax.set_title("Observed Demand and 24-Hour Recursive LSTM Forecast"); ax.set_xlabel("Timestamp"); ax.set_ylabel("Demand (kW)")
    ax.legend(ncol=2,loc="upper left"); ax.grid(alpha=.25)
    fig.text(.08,.015,"The future forecast is recursive: each 15-minute prediction is fed into the next LSTM input window. No future actual demand is used.",fontsize=9)
    savefig(fig,"22_lstm_24h_recursive_forecast.png",.13)

    print("\nDONE")
    print("Saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
