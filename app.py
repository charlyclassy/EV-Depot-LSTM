import streamlit as st
import sys
import platform
from datetime import datetime
import pandas as pd
import json
import hashlib
import numpy as np
import plotly.graph_objects as go
import re
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def _model_version_id(depot_name: str, metadata: dict) -> str:
    """Deterministic traceability ID derived from depot and model metadata."""
    seed = "|".join([
        str(depot_name),
        str(metadata.get("model_type", "LSTM")),
        str(metadata.get("training_date", "")),
        str(metadata.get("train_period", "")),
        str(metadata.get("validation_period", "")),
    ])
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8].upper()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(depot_name)).strip("-").upper()
    return f"LSTM-{slug}-{digest}"


def _consecutive_index_runs(indices):
    """Return inclusive consecutive-index runs as (start_index, end_index)."""
    values = np.asarray(indices, dtype=int)
    if values.size == 0:
        return []
    runs = []
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _format_duration(minutes: int) -> str:
    minutes = int(max(0, minutes))
    if minutes == 0:
        return "0 minutes"
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours} h {mins} min"
    if hours:
        return f"{hours} h"
    return f"{mins} min"


def _operational_recommendation(
    predicted_peak_kw: float,
    warning_kw: float,
    planning_kw: float,
    first_breach_time=None,
    first_breach_end=None,
) -> str:
    """Decision-support wording only; no automatic control action is implied."""
    if predicted_peak_kw >= planning_kw:
        excess = max(0.0, predicted_peak_kw - planning_kw)
        window = ""
        if first_breach_time is not None and first_breach_end is not None:
            window = (
                f" during the forecast breach window "
                f"({pd.Timestamp(first_breach_time):%d %b %H:%M}–"
                f"{pd.Timestamp(first_breach_end):%H:%M})"
            )
        return (
            f"Review whether flexible charging can be staggered or shifted{window}. "
            f"The forecast peak is {excess:.1f} kW above the planning capacity threshold. "
            "Any charging adjustment should be confirmed against operational requirements."
        )
    if predicted_peak_kw >= warning_kw:
        margin = max(0.0, planning_kw - predicted_peak_kw)
        return (
            "Review the charging schedule for avoidable coincidence during the forecast peak. "
            f"The forecast remains {margin:.1f} kW below the planning capacity threshold, "
            "so early load staggering may preserve additional headroom."
        )
    headroom = max(0.0, planning_kw - predicted_peak_kw)
    return (
        f"No capacity intervention is indicated by the current forecast. "
        f"Forecast headroom to the planning capacity threshold is approximately {headroom:.1f} kW. "
        "Continue routine monitoring."
    )


def _build_operational_pdf(report_rows, title: str) -> bytes:
    """Create a concise traceable PDF report in memory."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=title,
        author="Optivolt Solutions Limited",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "OptivoltTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10,
    )
    sub_style = ParagraphStyle(
        "OptivoltSub",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "OptivoltBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    story = [
        Paragraph("Optivolt EV Depot Intelligence Portal", title_style),
        Paragraph(title, styles["Heading2"]),
        Paragraph(
            "Forecast status is based on the historical project dataset and does not represent live grid telemetry.",
            sub_style,
        ),
        Spacer(1, 4),
    ]

    table_data = [
        [Paragraph("<b>Report Field</b>", body_style), Paragraph("<b>Value</b>", body_style)]
    ]
    for field, value in report_rows:
        table_data.append([
            Paragraph(str(field), body_style),
            Paragraph(str(value).replace("&", "&amp;"), body_style),
        ])

    table = Table(table_data, colWidths=[62 * mm, 112 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2A43")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#F8FAFC"), colors.HexColor("#EFF6FF")
        ]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Planning Capacity Threshold is an application planning value and is not presented as a verified contracted grid connection limit.",
        body_style,
    ))
    doc.build(story)
    return buffer.getvalue()


st.set_page_config(page_title="Optivolt EV Depot Intelligence Portal", page_icon="⚡", layout="wide")

st.markdown("""
<style>
/* Keep full model-performance metric names visible on summary cards */
div[data-testid="stMetricLabel"] p {
    font-size: 0.82rem !important;
    line-height: 1.15 !important;
    white-space: normal !important;
}
</style>
""", unsafe_allow_html=True)

# ===================================================================== #
# REAL UK POWER NETWORKS OPTIMISE PRIME DATA                             #
# ===================================================================== #
REAL_DEPOTS = [
    "Bexleyheath", "Dartford", "Islington", "Mount Pleasant",
    "Orpington", "Premier Park", "Whitechapel", "Camden", "Victoria",
]

# These are default PLANNING thresholds derived from the training-period
# demand profile. They are not claimed to be contracted grid connection limits.
DEPOT_CAPACITY_KW = {
    "Bexleyheath": 17.0,
    "Dartford": 45.0,
    "Islington": 13.0,
    "Mount Pleasant": 260.0,
    "Orpington": 14.0,
    "Premier Park": 70.0,
    "Whitechapel": 55.0,
    "Camden": 21.0,
    "Victoria": 29.0,
}
WARNING_ALPHA = 0.80
SAMPLING_MINUTES = 15

# Fixed chronological project periods for the real Optimise Prime data.
DATA_START = pd.Timestamp("2021-07-08 00:00:00")
TRAIN_END = pd.Timestamp("2022-04-01")
VALID_END = pd.Timestamp("2022-05-10")
TEST_END = pd.Timestamp("2022-07-26")
STEPS_PER_HOUR = 60 // SAMPLING_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# Selected final model configuration
LSTM_LOOKBACK = STEPS_PER_DAY  # 96 x 15-minute intervals = 24 hours
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 256
MODEL_SEED = 42

np.random.seed(MODEL_SEED)
tf.keras.utils.set_random_seed(MODEL_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

FINAL_METRICS_FILE = Path("model_comparison_metrics.csv")
FINAL_SELECTION_FILE = Path("model_selection_decision.csv")
FINAL_PREDICTIONS_FILE = Path("heldout_predictions_all_models.csv")
FINAL_FUTURE_FILE = Path("lstm_24h_recursive_forecast.csv")
MODEL_DIR = Path("models")

# The app supports either one combined processed file or the nine
# individual real depot CSVs placed at the repository root.
BUNDLED_DATA_CANDIDATES = [
    Path("data/processed/depot_demand.csv"),
    Path("UKPN_OptimisePrime_9_Depots_Processed.csv"),
]

INDIVIDUAL_DEPOT_FILES = {
    "Bexleyheath": Path("UKPN_OptimisePrime_Bexleyheath.csv"),
    "Camden": Path("UKPN_OptimisePrime_Camden.csv"),
    "Dartford": Path("UKPN_OptimisePrime_Dartford.csv"),
    "Islington": Path("UKPN_OptimisePrime_Islington.csv"),
    "Mount Pleasant": Path("UKPN_OptimisePrime_Mount_Pleasant.csv"),
    "Orpington": Path("UKPN_OptimisePrime_Orpington.csv"),
    "Premier Park": Path("UKPN_OptimisePrime_Premier_Park.csv"),
    "Victoria": Path("UKPN_OptimisePrime_Victoria.csv"),
    "Whitechapel": Path("UKPN_OptimisePrime_Whitechapel.csv"),
}


@st.cache_data(show_spinner=False)
def load_real_depot_data(depot_name: str) -> pd.DataFrame:
    """Load one genuine Optimise Prime depot."""
    # Prefer a combined dataset if one exists.
    for path in BUNDLED_DATA_CANDIDATES:
        if path.exists():
            raw = pd.read_csv(path, parse_dates=["timestamp"], dtype={"depot_id": str})
            required = {"timestamp", "depot_id", "demand_kw"}
            missing = required.difference(raw.columns)
            if missing:
                raise ValueError(f"Real depot file is missing columns: {sorted(missing)}")

            df = raw.loc[
                raw["depot_id"].astype(str) == depot_name,
                ["timestamp", "demand_kw"],
            ].copy()
            if not df.empty:
                df = df.rename(columns={"timestamp": "Timestamp", "demand_kw": "Demand_kW"})
                return df.sort_values("Timestamp").reset_index(drop=True)

    # Otherwise load the matching individual depot CSV from repo root.
    path = INDIVIDUAL_DEPOT_FILES.get(depot_name)
    if path is None:
        raise ValueError(f"Unknown depot: {depot_name}")
    if not path.exists():
        raise FileNotFoundError(
            f"Real data file for {depot_name} was not found. Expected: {path}"
        )

    raw = pd.read_csv(path, parse_dates=["timestamp"], dtype={"depot_id": str})
    required = {"timestamp", "demand_kw"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = raw[["timestamp", "demand_kw"]].copy()
    df = df.rename(columns={"timestamp": "Timestamp", "demand_kw": "Demand_kW"})
    return df.sort_values("Timestamp").reset_index(drop=True)


def _standardise_input(data_df: pd.DataFrame) -> pd.DataFrame:
    df = data_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp" not in df.columns or "Demand_kW" not in df.columns:
        time_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        demand_candidates = [c for c in df.columns if "demand" in c.lower() or "load" in c.lower() or "kw" in c.lower()]
        if not time_candidates or not demand_candidates:
            raise ValueError("The data needs a timestamp column and a demand/load kW column.")
        df = df[[time_candidates[0], demand_candidates[0]]].rename(
            columns={time_candidates[0]: "Timestamp", demand_candidates[0]: "Demand_kW"}
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Demand_kW"] = pd.to_numeric(df["Demand_kW"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Demand_kW"])
    df = df.groupby("Timestamp", as_index=False)["Demand_kW"].mean().sort_values("Timestamp")
    df["Demand_kW"] = df["Demand_kW"].clip(lower=0)

    # Harmonise uploaded data to the real model's 15-minute grid.
    df = df.set_index("Timestamp").resample(f"{SAMPLING_MINUTES}min").mean()
    df["Demand_kW"] = df["Demand_kW"].interpolate(limit=4, limit_direction="forward").ffill()
    df = df.dropna(subset=["Demand_kW"])
    return df



def validate_uploaded_telemetry(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate and standardise an uploaded depot telemetry file.

    Required logical fields:
      - timestamp/date/time
      - demand/load/kW

    The returned dataframe is harmonised to the model's 15-minute grid.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The uploaded file is empty.")

    original = raw_df.copy()
    original.columns = [str(c).strip() for c in original.columns]

    time_candidates = [
        c for c in original.columns
        if c == "Timestamp" or "time" in c.lower() or "date" in c.lower()
    ]
    demand_candidates = [
        c for c in original.columns
        if c == "Demand_kW"
        or "demand" in c.lower()
        or "load" in c.lower()
        or "kw" in c.lower()
    ]

    if not time_candidates:
        raise ValueError(
            "No timestamp/date/time column was detected. "
            "Add a timestamp column and upload the file again."
        )
    if not demand_candidates:
        raise ValueError(
            "No demand/load kW column was detected. "
            "Add a demand or load kW column and upload the file again."
        )

    raw_rows = len(original)
    time_col = time_candidates[0]
    demand_col = demand_candidates[0]

    work = original[[time_col, demand_col]].rename(
        columns={time_col: "Timestamp", demand_col: "Demand_kW"}
    )
    work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
    work["Demand_kW"] = pd.to_numeric(work["Demand_kW"], errors="coerce")

    missing_timestamps = int(work["Timestamp"].isna().sum())
    missing_demand_values = int(work["Demand_kW"].isna().sum())
    invalid_rows = int(work[["Timestamp", "Demand_kW"]].isna().any(axis=1).sum())
    negative_values = int((work["Demand_kW"] < 0).fillna(False).sum())

    valid_timestamp_series = work["Timestamp"].dropna().sort_values()
    duplicate_timestamps = int(valid_timestamp_series.duplicated().sum())
    if len(valid_timestamp_series) >= 3:
        sampling_deltas = valid_timestamp_series.diff().dropna()
        sampling_consistency_pct = float(
            100.0 * (sampling_deltas == pd.Timedelta(minutes=15)).mean()
        )
    else:
        sampling_consistency_pct = 0.0

    standardised = _standardise_input(original)

    if standardised.empty:
        raise ValueError("No usable demand observations remain after validation.")

    coverage_days = (
        standardised.index.max() - standardised.index.min()
    ).total_seconds() / 86400.0

    # A one-week lag is used by the feature set. Require enough history for
    # feature construction plus meaningful chronological train/validation/test splits.
    minimum_days = int(st.session_state.get("min_upload_history_days", 28))
    if coverage_days < minimum_days:
        raise ValueError(
            f"At least {minimum_days} days of telemetry are required. "
            f"The uploaded file covers approximately {coverage_days:.1f} days."
        )

    report = {
        "timestamp_column": time_col,
        "demand_column": demand_col,
        "raw_rows": raw_rows,
        "missing_timestamps": missing_timestamps,
        "missing_demand_values": missing_demand_values,
        "duplicate_timestamps": duplicate_timestamps,
        "sampling_consistency_pct": sampling_consistency_pct,
        "invalid_rows_removed": invalid_rows,
        "negative_values_clipped": negative_values,
        "standardised_rows": len(standardised),
        "start": standardised.index.min(),
        "end": standardised.index.max(),
        "coverage_days": coverage_days,
        "sampling_interval": "15 minutes",
    }
    return standardised, report


def derive_uploaded_planning_threshold(standardised_df: pd.DataFrame) -> float:
    """
    Derive a default planning threshold for a newly uploaded depot from the
    development portion only: 75th percentile of daily peak demand.
    """
    n = len(standardised_df)
    development_end = max(1, int(n * 0.85))
    development = standardised_df.iloc[:development_end]["Demand_kW"]
    daily_peaks = development.resample("1D").max().dropna()
    if daily_peaks.empty:
        return float(max(1.0, development.quantile(0.75)))
    return float(max(1.0, daily_peaks.quantile(0.75)))

def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causal time-series features: every demand feature uses past observations only."""
    feat = df.copy()
    hour_decimal = feat.index.hour + feat.index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24.0)
    feat["day_of_week"] = feat.index.dayofweek
    feat["weekend_flag"] = (feat.index.dayofweek >= 5).astype(int)
    feat["month"] = feat.index.month

    feat["Lag_1"] = feat["Demand_kW"].shift(1)
    feat["Lag_4"] = feat["Demand_kW"].shift(STEPS_PER_HOUR)
    feat["Lag_96"] = feat["Demand_kW"].shift(STEPS_PER_DAY)
    feat["Lag_672"] = feat["Demand_kW"].shift(STEPS_PER_WEEK)
    past = feat["Demand_kW"].shift(1)
    feat["Rolling_mean_4"] = past.rolling(STEPS_PER_HOUR).mean()
    feat["Rolling_mean_24"] = past.rolling(6 * STEPS_PER_HOUR).mean()
    feat["Rolling_mean_96"] = past.rolling(STEPS_PER_DAY).mean()
    feat["Rolling_std_96"] = past.rolling(STEPS_PER_DAY).std()

    features = [
        "hour_sin", "hour_cos", "day_of_week", "weekend_flag", "month",
        "Lag_1", "Lag_4", "Lag_96", "Lag_672",
        "Rolling_mean_4", "Rolling_mean_24", "Rolling_mean_96", "Rolling_std_96",
    ]
    return feat.dropna(subset=features + ["Demand_kW"]), features


def _make_lstm_sequences(scaled_values, index, target_times):
    """Build 96-step LSTM inputs ending immediately before each target timestamp."""
    positions = pd.Series(np.arange(len(index)), index=index)
    X, times = [], []
    for ts in pd.DatetimeIndex(target_times):
        if ts not in positions.index:
            continue
        pos = int(positions.loc[ts])
        if pos < LSTM_LOOKBACK:
            continue
        X.append(scaled_values[pos - LSTM_LOOKBACK:pos])
        times.append(ts)
    if not X:
        raise ValueError("No valid LSTM sequences could be constructed from this dataset.")
    return np.asarray(X, dtype=np.float32).reshape(-1, LSTM_LOOKBACK, 1), pd.DatetimeIndex(times)


def _new_lstm_model(learning_rate: float = 0.001):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(MODEL_SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(LSTM_LOOKBACK, 1)),
        keras.layers.LSTM(
            48,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=MODEL_SEED),
            recurrent_initializer=keras.initializers.Orthogonal(seed=MODEL_SEED),
        ),
        keras.layers.Dense(
            24,
            activation="relu",
            kernel_initializer=keras.initializers.HeUniform(seed=MODEL_SEED),
        ),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)), loss="mse")
    return model


def _recursive_forecast(model, history: pd.Series, scaler_bundle: dict, horizon_hours: int):
    """Genuine multi-step LSTM forecast. Each prediction becomes part of the next input."""
    x_scaler = scaler_bundle["x_scaler"]
    y_scaler = scaler_bundle["y_scaler"]
    history_values = list(history.astype(float).sort_index().values)
    start = history.index.max() + pd.Timedelta(minutes=SAMPLING_MINUTES)
    steps = int(horizon_hours * STEPS_PER_HOUR)
    future_times = pd.date_range(start=start, periods=steps, freq=f"{SAMPLING_MINUTES}min")
    preds = []
    for _ in range(steps):
        window = np.asarray(history_values[-LSTM_LOOKBACK:], dtype=float).reshape(-1, 1)
        window_scaled = x_scaler.transform(window).reshape(1, LSTM_LOOKBACK, 1)
        pred_scaled = model.predict(window_scaled, verbose=0).reshape(-1, 1)
        pred = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
        pred = max(0.0, pred)
        preds.append(pred)
        history_values.append(pred)
    return pd.DatetimeIndex(future_times), np.asarray(preds, dtype=float)


def _compute_metrics(y_true, y_pred, capacity_kw: float):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = round(float(mean_absolute_error(y_true, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2)
    actual_breach = y_true >= capacity_kw
    pred_breach = y_pred >= capacity_kw
    tp = int(np.sum(actual_breach & pred_breach))
    fp = int(np.sum((~actual_breach) & pred_breach))
    fn = int(np.sum(actual_breach & (~pred_breach)))
    recall = f"{100 * tp / (tp + fn):.2f}%" if (tp + fn) else "N/A"
    far = f"{100 * fp / (tp + fp):.2f}%" if (tp + fp) else "N/A"
    return mae, rmse, recall, far


def _load_final_bexleyheath_evidence(horizon_hours: int):
    """Use the final comparison CSV outputs so normal Bexleyheath navigation is immediate."""
    if not (FINAL_METRICS_FILE.exists() and FINAL_PREDICTIONS_FILE.exists() and FINAL_FUTURE_FILE.exists()):
        return None
    try:
        metrics = pd.read_csv(FINAL_METRICS_FILE)
        preds = pd.read_csv(FINAL_PREDICTIONS_FILE, parse_dates=["timestamp"])
        future = pd.read_csv(FINAL_FUTURE_FILE, parse_dates=["timestamp"])
        lstm_row = metrics.loc[metrics["Model"] == "LSTM"].iloc[0]
        base_row = metrics.loc[metrics["Model"] == "Previous-day same-time"].iloc[0]
        raw = _standardise_input(load_real_depot_data("Bexleyheath"))
        steps = min(len(future), int(horizon_hours * STEPS_PER_HOUR))
        future = future.head(steps)
        meta = {
            "model_type": "LSTM Depot Demand Forecast Model",
            "training_date": "Final comparison run",
            "dataset_source": "UK Power Networks / Optimise Prime WS2 depot trial",
            "split_strategy": "Fixed Optimise Prime project periods",
            "selected_depot": "Bexleyheath",
            "total_rows": len(raw),
            "sampling_interval": "15 minutes",
            "target_variable": "Demand_kW",
            "features_used": ["Previous 96 demand intervals (24-hour LSTM sequence)"],
        "training_epochs_requested": training_epochs if "training_epochs" in locals() else LSTM_EPOCHS,
        "training_batch_size": training_batch_size if "training_batch_size" in locals() else LSTM_BATCH_SIZE,
        "early_stopping_patience": training_patience if "training_patience" in locals() else 5,
        "early_stopping_min_delta": training_min_delta if "training_min_delta" in locals() else 1e-5,
        "learning_rate": training_learning_rate if "training_learning_rate" in locals() else 0.001,
            "train_period": "2021-07-08 to 2022-03-31",
            "validation_period": "2022-04-01 to 2022-05-09",
            "final_fit_period": "Training with validation used for early stopping",
            "test_period": "2022-05-10 to 2022-07-25",
            "test_observations": int(len(preds)),
            "metric_scope": "LSTM and baseline metrics use the same held-out test rows",
            "forecast_horizon_hours": int(horizon_hours),
            "synthetic_data_used": False,
            "metrics_xgb": {
                "mae": round(float(lstm_row["MAE_kW"]), 2),
                "rmse": round(float(lstm_row["RMSE_kW"]), 2),
                "recall": f"{100*float(lstm_row['Breach_Recall']):.2f}%",
                "far": f"{100*float(lstm_row['False_Alarm_Ratio']):.2f}%",
            },
            "metrics_baseline": {
                "mae": round(float(base_row["MAE_kW"]), 2),
                "rmse": round(float(base_row["RMSE_kW"]), 2),
                "recall": f"{100*float(base_row['Breach_Recall']):.2f}%",
                "far": f"{100*float(base_row['False_Alarm_Ratio']):.2f}%",
            },
            "final_comparison_metrics": metrics.to_dict(orient="records"),
        }
        if FINAL_SELECTION_FILE.exists():
            meta["model_selection_decision"] = pd.read_csv(FINAL_SELECTION_FILE).to_dict(orient="records")
        return {
            "metadata": meta,
            "history": raw["Demand_kW"].copy(),
            "y_test": pd.Series(preds["actual_kw"].to_numpy(float), index=pd.DatetimeIndex(preds["timestamp"])),
            "xgb_preds": preds["lstm_kw"].to_numpy(float),  # legacy internal key retained for UI compatibility
            "baseline_preds": preds["baseline_kw"].to_numpy(float),
            "errors": preds["actual_kw"].to_numpy(float) - preds["lstm_kw"].to_numpy(float),
            "future_times": pd.DatetimeIndex(future["timestamp"]),
            "future_forecast": future["lstm_forecast_kw"].to_numpy(float),
            "model": None,
            "feature_cols": None,
        }
    except Exception:
        return None



def _model_slug(depot_name: str) -> str:
    return depot_name.lower().replace("&", "and").replace(" ", "_").replace("-", "_")


def _restore_standard_scaler(mean_value: float, scale_value: float) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray([float(mean_value)], dtype=float)
    scaler.scale_ = np.asarray([float(scale_value)], dtype=float)
    scaler.var_ = np.asarray([float(scale_value) ** 2], dtype=float)
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = np.int64(1)
    return scaler


@st.cache_resource(show_spinner=False)
def _load_saved_builtin_model(depot_name: str):
    """Load a previously trained built-in depot model without fitting."""
    slug = _model_slug(depot_name)
    model_path = MODEL_DIR / f"{slug}.keras"
    scaler_path = MODEL_DIR / f"{slug}_scalers.npz"
    metadata_path = MODEL_DIR / f"{slug}_metadata.json"

    missing = [p.name for p in (model_path, scaler_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"LSTM model artifacts for {depot_name} are not available: {', '.join(missing)}. "
            "Run the supplied built-in model training workflow once, commit the generated "
            "models folder to GitHub, then reboot the app."
        )

    model = keras.models.load_model(model_path, compile=False)
    scalers = np.load(scaler_path)
    x_scaler = _restore_standard_scaler(
        float(scalers["x_mean"][0]), float(scalers["x_scale"][0])
    )
    y_scaler = _restore_standard_scaler(
        float(scalers["y_mean"][0]), float(scalers["y_scale"][0])
    )

    saved_metadata = {}
    if metadata_path.exists():
        try:
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            saved_metadata = {}

    return model, {"x_scaler": x_scaler, "y_scaler": y_scaler}, saved_metadata


def _load_pretrained_builtin_pipeline(
    depot_name: str,
    capacity_kw: float,
    horizon_hours: int,
):
    """Run inference/evaluation from a depot-specific LSTM model; never call fit()."""
    source_df = load_real_depot_data(depot_name)
    df = _standardise_input(source_df)

    train_times = df.index[(df.index >= DATA_START) & (df.index < TRAIN_END)]
    valid_times = df.index[(df.index >= TRAIN_END) & (df.index < VALID_END)]
    test_times = df.index[(df.index >= VALID_END) & (df.index < TEST_END)]
    if not len(train_times) or not len(valid_times) or not len(test_times):
        raise ValueError(
            f"{depot_name} does not contain the expected chronological project periods."
        )

    model, scaler_bundle, saved_metadata = _load_saved_builtin_model(depot_name)
    x_scaler = scaler_bundle["x_scaler"]
    y_scaler = scaler_bundle["y_scaler"]

    scaled_all = x_scaler.transform(df[["Demand_kW"]]).ravel()
    X_test, t_test = _make_lstm_sequences(scaled_all, df.index, test_times)
    y_test = df.loc[t_test, "Demand_kW"].astype(float)

    pred_scaled = model.predict(
        X_test, batch_size=LSTM_BATCH_SIZE, verbose=0
    ).reshape(-1, 1)
    lstm_preds = np.clip(
        y_scaler.inverse_transform(pred_scaled).ravel(), 0, None
    )

    baseline_preds = np.asarray(
        [
            float(df.loc[ts - pd.Timedelta(days=1), "Demand_kW"])
            if ts - pd.Timedelta(days=1) in df.index
            else np.nan
            for ts in t_test
        ],
        dtype=float,
    )
    valid_mask = np.isfinite(baseline_preds)
    t_test = t_test[valid_mask]
    y_test = pd.Series(y_test.to_numpy()[valid_mask], index=t_test)
    lstm_preds = lstm_preds[valid_mask]
    baseline_preds = baseline_preds[valid_mask]
    errors = y_test.to_numpy() - lstm_preds

    mae_lstm, rmse_lstm, recall_lstm, far_lstm = _compute_metrics(
        y_test, lstm_preds, capacity_kw
    )
    mae_base, rmse_base, recall_base, far_base = _compute_metrics(
        y_test, baseline_preds, capacity_kw
    )

    history_series = df["Demand_kW"].copy()
    future_times, future_forecast = _recursive_forecast(
        model, history_series, scaler_bundle, horizon_hours
    )

    trained_at = saved_metadata.get("trained_at", "Model artifact")
    summary_metadata = {
        "model_type": "LSTM Depot Demand Forecast Model",
        "training_date": trained_at,
        "dataset_source": "UK Power Networks / Optimise Prime WS2 depot trial",
        "split_strategy": "Fixed Optimise Prime project periods",
        "selected_depot": depot_name,
        "total_rows": len(df),
        "sampling_interval": "15 minutes",
        "target_variable": "Demand_kW",
        "features_used": ["Previous 96 demand intervals (24-hour LSTM sequence)"],
        "train_period": f"{train_times[0]:%Y-%m-%d} to {train_times[-1]:%Y-%m-%d}",
        "validation_period": f"{valid_times[0]:%Y-%m-%d} to {valid_times[-1]:%Y-%m-%d}",
        "final_fit_period": "Training set with validation used for early stopping",
        "test_period": f"{t_test[0]:%Y-%m-%d} to {t_test[-1]:%Y-%m-%d}",
        "test_observations": int(len(t_test)),
        "metric_scope": "LSTM and baseline metrics use the same held-out test rows",
        "forecast_horizon_hours": horizon_hours,
        "synthetic_data_used": False,
        "metrics_xgb": {
            "mae": mae_lstm,
            "rmse": rmse_lstm,
            "recall": recall_lstm,
            "far": far_lstm,
        },
        "metrics_baseline": {
            "mae": mae_base,
            "rmse": rmse_base,
            "recall": recall_base,
            "far": far_base,
        },
    }

    return {
        "metadata": summary_metadata,
        "history": history_series,
        "y_test": y_test,
        "xgb_preds": lstm_preds,
        "baseline_preds": baseline_preds,
        "errors": errors,
        "future_times": future_times,
        "future_forecast": future_forecast,
        "model": model,
        "feature_cols": scaler_bundle,
    }


def _execute_production_ml_pipeline_core(
    depot_name: str,
    capacity_kw: float,
    horizon_hours: int,
    uploaded_df=None,
    _progress_callback=None,
):
    """Train/evaluate the selected LSTM on real depot demand; no synthetic demand is created."""
    def report_progress(percent: int, message: str):
        if _progress_callback is not None:
            _progress_callback(int(percent), message)

    report_progress(5, "Loading depot telemetry")
    source_df = uploaded_df.copy() if uploaded_df is not None else load_real_depot_data(depot_name)
    report_progress(12, "Standardising timestamps and demand to the 15-minute model grid")
    df = _standardise_input(source_df)
    is_uploaded = uploaded_df is not None

    report_progress(25, "Creating chronological training, validation and held-out test periods")
    if is_uploaded:
        available = df.index[LSTM_LOOKBACK:]
        n = len(available)
        train_end_idx = int(n * 0.70)
        valid_end_idx = int(n * 0.85)
        train_times = available[:train_end_idx]
        valid_times = available[train_end_idx:valid_end_idx]
        test_times = available[valid_end_idx:]
        split_strategy = "Relative chronological split: 70% train / 15% validation / 15% held-out test"
    else:
        train_times = df.index[(df.index >= DATA_START) & (df.index < TRAIN_END)]
        valid_times = df.index[(df.index >= TRAIN_END) & (df.index < VALID_END)]
        test_times = df.index[(df.index >= VALID_END) & (df.index < TEST_END)]
        split_strategy = "Fixed Optimise Prime project periods"

    if not len(train_times) or not len(valid_times) or not len(test_times):
        raise ValueError("The dataset does not contain enough chronological observations for the LSTM split.")

    development_end = pd.DatetimeIndex(valid_times).max()
    development = df.loc[df.index <= development_end, ["Demand_kW"]]
    x_scaler = StandardScaler().fit(development[["Demand_kW"]])
    y_scaler = StandardScaler().fit(development[["Demand_kW"]])
    scaled_all = x_scaler.transform(df[["Demand_kW"]]).ravel()

    X_train, t_train = _make_lstm_sequences(scaled_all, df.index, train_times)
    X_valid, t_valid = _make_lstm_sequences(scaled_all, df.index, valid_times)
    X_test, t_test = _make_lstm_sequences(scaled_all, df.index, test_times)
    y_train = y_scaler.transform(df.loc[t_train, ["Demand_kW"]]).ravel().astype(np.float32)
    y_valid = y_scaler.transform(df.loc[t_valid, ["Demand_kW"]]).ravel().astype(np.float32)
    y_test = df.loc[t_test, "Demand_kW"].astype(float)

    report_progress(40, "Training LSTM with early stopping")

    if is_uploaded:
        training_epochs = int(st.session_state.get("retrain_epochs", LSTM_EPOCHS))
        training_batch_size = int(st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE))
        training_patience = int(st.session_state.get("retrain_patience", 5))
        training_min_delta = float(st.session_state.get("retrain_min_delta", 1e-5))
        training_learning_rate = float(st.session_state.get("retrain_learning_rate", 0.001))
    else:
        training_epochs = LSTM_EPOCHS
        training_batch_size = LSTM_BATCH_SIZE
        training_patience = 5
        training_min_delta = 1e-5
        training_learning_rate = 0.001

    model = _new_lstm_model(learning_rate=training_learning_rate)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=training_patience,
        min_delta=training_min_delta,
        restore_best_weights=True,
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=training_epochs,
        batch_size=training_batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[early_stop],
    )

    report_progress(76, "LSTM training complete — generating held-out predictions")
    pred_scaled = model.predict(
        X_test,
        batch_size=training_batch_size,
        verbose=0,
    ).reshape(-1, 1)
    lstm_preds = np.clip(y_scaler.inverse_transform(pred_scaled).ravel(), 0, None)
    baseline_preds = np.asarray([
        float(df.loc[ts - pd.Timedelta(days=1), "Demand_kW"])
        if ts - pd.Timedelta(days=1) in df.index else np.nan
        for ts in t_test
    ], dtype=float)
    valid_mask = np.isfinite(baseline_preds)
    t_test = t_test[valid_mask]
    y_test = pd.Series(y_test.to_numpy()[valid_mask], index=t_test)
    lstm_preds = lstm_preds[valid_mask]
    baseline_preds = baseline_preds[valid_mask]
    errors = y_test.to_numpy() - lstm_preds

    mae_lstm, rmse_lstm, recall_lstm, far_lstm = _compute_metrics(y_test, lstm_preds, capacity_kw)
    mae_base, rmse_base, recall_base, far_base = _compute_metrics(y_test, baseline_preds, capacity_kw)

    report_progress(90, f"Generating the {horizon_hours}-hour recursive LSTM demand forecast")
    history_series = df["Demand_kW"].copy()
    scaler_bundle = {"x_scaler": x_scaler, "y_scaler": y_scaler}
    future_times, future_forecast = _recursive_forecast(model, history_series, scaler_bundle, horizon_hours)
    report_progress(100, "LSTM retraining and evaluation completed")

    summary_metadata = {
        "model_type": "LSTM Depot Demand Forecast Model",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_source": ("Uploaded depot telemetry" if is_uploaded else "UK Power Networks / Optimise Prime WS2 depot trial"),
        "split_strategy": split_strategy,
        "selected_depot": depot_name,
        "total_rows": len(df),
        "sampling_interval": "15 minutes",
        "target_variable": "Demand_kW",
        "features_used": ["Previous 96 demand intervals (24-hour LSTM sequence)"],
        "train_period": f"{t_train[0]:%Y-%m-%d} to {t_train[-1]:%Y-%m-%d}",
        "validation_period": f"{t_valid[0]:%Y-%m-%d} to {t_valid[-1]:%Y-%m-%d}",
        "final_fit_period": "Training set with validation used for early stopping",
        "test_period": f"{t_test[0]:%Y-%m-%d} to {t_test[-1]:%Y-%m-%d}",
        "test_observations": int(len(t_test)),
        "metric_scope": "LSTM and baseline metrics use the same held-out test rows",
        "forecast_horizon_hours": horizon_hours,
        "synthetic_data_used": False,
        "metrics_xgb": {"mae": mae_lstm, "rmse": rmse_lstm, "recall": recall_lstm, "far": far_lstm},
        "metrics_baseline": {"mae": mae_base, "rmse": rmse_base, "recall": recall_base, "far": far_base},
    }

    return {
        "metadata": summary_metadata,
        "history": history_series,
        "y_test": y_test,
        "xgb_preds": lstm_preds,  # legacy internal key retained so original UI layout remains unchanged
        "baseline_preds": baseline_preds,
        "errors": errors,
        "future_times": future_times,
        "future_forecast": future_forecast,
        "model": model,
        "feature_cols": scaler_bundle,
    }


@st.cache_resource(show_spinner=False)
def execute_production_ml_pipeline(
    depot_name: str,
    capacity_kw: float,
    horizon_hours: int,
    uploaded_df=None,
):
    """
    Normal dashboard path.

    Built-in depots load depot-specific LSTM model artifacts and never train when
    the user changes navigation or selects a depot. Training remains an explicit
    action for uploaded data.
    """
    if uploaded_df is None and depot_name in REAL_DEPOTS:
        return _load_pretrained_builtin_pipeline(
            depot_name, capacity_kw, horizon_hours
        )

    return _execute_production_ml_pipeline_core(
        depot_name, capacity_kw, horizon_hours, uploaded_df, _progress_callback=None
    )


def execute_production_ml_pipeline_with_progress(
    depot_name: str,
    capacity_kw: float,
    horizon_hours: int,
    uploaded_df,
    progress_callback,
):
    return _execute_production_ml_pipeline_core(
        depot_name, capacity_kw, horizon_hours, uploaded_df, _progress_callback=progress_callback
    )


def _ensure_project_lstm_for_unseen_eval(pipeline):
    """Return an already loaded model for unseen-data evaluation; never fit here."""
    if pipeline.get("model") is not None and pipeline.get("feature_cols") is not None:
        return pipeline["model"], pipeline["feature_cols"]

    active_depot = st.session_state.get("current_depot", "Bexleyheath")
    if active_depot not in REAL_DEPOTS:
        active_depot = "Bexleyheath"
    model, scaler_bundle, _ = _load_saved_builtin_model(active_depot)
    return model, scaler_bundle


def evaluate_existing_model_on_uploaded_data(
    current_model,
    expected_feature_cols,
    raw_uploaded_df: pd.DataFrame,
    capacity_kw: float,
) -> dict:
    """Evaluate the fitted LSTM on uploaded telemetry without fitting to the uploaded observations."""
    if current_model is None or not isinstance(expected_feature_cols, dict):
        raise ValueError("The existing LSTM must be prepared before unseen-data evaluation.")
    x_scaler = expected_feature_cols["x_scaler"]
    y_scaler = expected_feature_cols["y_scaler"]
    df = _standardise_input(raw_uploaded_df)
    scaled = x_scaler.transform(df[["Demand_kW"]]).ravel()
    target_times = df.index[LSTM_LOOKBACK:]
    X, times = _make_lstm_sequences(scaled, df.index, target_times)
    pred_scaled = current_model.predict(X, batch_size=LSTM_BATCH_SIZE, verbose=0).reshape(-1, 1)
    preds = np.clip(y_scaler.inverse_transform(pred_scaled).ravel(), 0, None)
    actual = df.loc[times, "Demand_kW"].to_numpy(float)
    baseline = np.asarray([
        float(df.loc[ts - pd.Timedelta(days=1), "Demand_kW"])
        if ts - pd.Timedelta(days=1) in df.index else np.nan
        for ts in times
    ], dtype=float)
    mask = np.isfinite(baseline)
    times = times[mask]
    actual, preds, baseline = actual[mask], preds[mask], baseline[mask]

    def metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        actual_breach = y_true >= capacity_kw
        pred_breach = y_pred >= capacity_kw
        tp = int(np.sum(actual_breach & pred_breach))
        fp = int(np.sum((~actual_breach) & pred_breach))
        fn = int(np.sum(actual_breach & (~pred_breach)))
        recall = (100.0 * tp / (tp + fn)) if (tp + fn) else None
        far = (100.0 * fp / (tp + fp)) if (tp + fp) else None
        return {"mae": mae, "rmse": rmse, "r2": r2, "recall": recall, "far": far, "tp": tp, "fp": fp, "fn": fn}

    return {
        "start": times.min(), "end": times.max(), "rows": len(times),
        "timestamps": times.to_numpy(), "actual": actual, "predictions": preds,
        "baseline": baseline, "xgb_metrics": metrics(actual, preds),
        "baseline_metrics": metrics(actual, baseline),
    }


def regression_metrics_for_comparison(y_true, y_pred) -> dict:
    """Metrics used for before-vs-after retraining comparison on identical rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ===================================================================== #
# PERSISTED UPLOADED-DEPOT REGISTRY                                     #
# ===================================================================== #
APP_STATE_DIR = Path(".app_state")
UPLOADED_DEPOTS_DIR = APP_STATE_DIR / "uploaded_depots"
UPLOADED_DEPOTS_REGISTRY_FILE = APP_STATE_DIR / "uploaded_depots_registry.json"


def _safe_uploaded_depot_id(depot_name: str) -> str:
    """Stable filesystem-safe identifier for an uploaded depot."""
    return hashlib.sha256(depot_name.strip().lower().encode("utf-8")).hexdigest()[:16]


def load_uploaded_depots_registry() -> dict:
    """
    Load metadata for all persisted uploaded depots.
    Built-in Optimise Prime depots are never stored in this registry.
    """
    if not UPLOADED_DEPOTS_REGISTRY_FILE.exists():
        return {}

    try:
        registry = json.loads(
            UPLOADED_DEPOTS_REGISTRY_FILE.read_text(encoding="utf-8")
        )
        if not isinstance(registry, dict):
            return {}

        valid = {}
        for depot_name, meta in registry.items():
            if (
                isinstance(meta, dict)
                and depot_name not in REAL_DEPOTS
                and meta.get("data_file")
                and Path(meta["data_file"]).exists()
            ):
                valid[depot_name] = meta
        return valid
    except Exception:
        return {}


def _write_uploaded_depots_registry(registry: dict):
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_DEPOTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_DEPOTS_REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def save_uploaded_depot_state(
    depot_name: str,
    capacity_kw: float,
    raw_df: pd.DataFrame,
    filename: str,
    upload_id: str,
):
    """
    Add or update one uploaded depot without affecting any other uploaded depot.
    """
    clean_name = depot_name.strip()
    if not clean_name:
        raise ValueError("Uploaded depot name cannot be blank.")
    if clean_name in REAL_DEPOTS:
        raise ValueError(
            f"{clean_name} is a protected built-in Optimise Prime depot name. "
            "Choose a different name for uploaded telemetry."
        )

    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_DEPOTS_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_uploaded_depots_registry()
    depot_id = _safe_uploaded_depot_id(clean_name)
    data_path = UPLOADED_DEPOTS_DIR / f"{depot_id}.csv"
    raw_df.to_csv(data_path, index=False)

    registry[clean_name] = {
        "depot_name": clean_name,
        "capacity_kw": float(capacity_kw),
        "warning_kw": round(WARNING_ALPHA * float(capacity_kw), 1),
        "filename": filename,
        "upload_id": upload_id,
        "data_file": str(data_path),
    }
    _write_uploaded_depots_registry(registry)
    return registry


def load_uploaded_depot_dataframe(depot_name: str, registry: dict | None = None):
    """Load telemetry for one uploaded depot from the persistent registry."""
    registry = registry if registry is not None else load_uploaded_depots_registry()
    meta = registry.get(depot_name)
    if not meta:
        return None

    data_path = Path(meta.get("data_file", ""))
    if not data_path.exists():
        return None

    try:
        df = pd.read_csv(data_path)
        return None if df.empty else df
    except Exception:
        return None


def remove_persisted_uploaded_depot(depot_name: str) -> dict:
    """
    Remove exactly one uploaded depot. Built-in depots cannot be removed here.
    Returns the updated registry.
    """
    if depot_name in REAL_DEPOTS:
        raise ValueError("Built-in Optimise Prime depots cannot be removed.")

    registry = load_uploaded_depots_registry()
    meta = registry.pop(depot_name, None)

    if meta:
        try:
            data_path = Path(meta.get("data_file", ""))
            if data_path.exists():
                data_path.unlink()
        except Exception:
            pass

    _write_uploaded_depots_registry(registry)
    return registry


def find_uploaded_depot_by_upload_id(upload_id: str, registry: dict | None = None):
    registry = registry if registry is not None else load_uploaded_depots_registry()
    for depot_name, meta in registry.items():
        if meta.get("upload_id") == upload_id:
            return depot_name, meta
    return None, None


_persisted_uploaded_depots = load_uploaded_depots_registry()


def _optivolt_plotly_chart(fig, *args, **kwargs):
    """Display Plotly figures using the administrator-selected chart height."""
    try:
        preferred_height = int(st.session_state.get("preferred_chart_height", 450))
        fig.update_layout(height=preferred_height)
    except Exception:
        pass
    return st.plotly_chart(fig, *args, **kwargs)



def _format_admin_value(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _record_admin_changes(section, changes):
    """Append human-readable Settings/Admin changes to the session audit trail."""
    if not changes:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_depot = st.session_state.get("current_depot", "Not available")

    log = list(st.session_state.get("admin_change_log", []))
    for setting_name, old_value, new_value in changes:
        if _format_admin_value(old_value) == _format_admin_value(new_value):
            continue
        log.append(
            {
                "Timestamp": timestamp,
                "Section": section,
                "Setting": setting_name,
                "Previous Value": _format_admin_value(old_value),
                "New Value": _format_admin_value(new_value),
                "Active Depot": active_depot,
            }
        )

    # Keep the latest 500 entries in the current application session.
    st.session_state["admin_change_log"] = log[-500:]


# ===================================================================== #
# SESSION STATE / UI                                                     #
# ===================================================================== #
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "current_depot" not in st.session_state: st.session_state["current_depot"] = "Bexleyheath"
if "org_name" not in st.session_state: st.session_state["org_name"] = "Optivolt Solutions Limited"
if "grid_limit" not in st.session_state: st.session_state["grid_limit"] = DEPOT_CAPACITY_KW.get(st.session_state["current_depot"], 17.0)
if "warn_threshold" not in st.session_state: st.session_state["warn_threshold"] = round(WARNING_ALPHA * st.session_state["grid_limit"], 1)
if "forecast_horizon" not in st.session_state: st.session_state["forecast_horizon"] = 24
if "portal_title" not in st.session_state: st.session_state["portal_title"] = "Optivolt EV Depot Intelligence Portal"
if "depot_threshold_overrides" not in st.session_state:
    st.session_state["depot_threshold_overrides"] = {}
if "preferred_chart_height" not in st.session_state: st.session_state["preferred_chart_height"] = 450
if "display_density" not in st.session_state: st.session_state["display_density"] = "Compact"
if "display_theme" not in st.session_state: st.session_state["display_theme"] = "Optivolt Blue / Teal"
if "min_upload_history_days" not in st.session_state: st.session_state["min_upload_history_days"] = 28
if "retrain_epochs" not in st.session_state: st.session_state["retrain_epochs"] = LSTM_EPOCHS
if "retrain_batch_size" not in st.session_state: st.session_state["retrain_batch_size"] = LSTM_BATCH_SIZE
if "retrain_patience" not in st.session_state: st.session_state["retrain_patience"] = 5
if "retrain_min_delta" not in st.session_state: st.session_state["retrain_min_delta"] = 1e-5
if "retrain_learning_rate" not in st.session_state: st.session_state["retrain_learning_rate"] = 0.001

if "pending_admin_settings" not in st.session_state:
    st.session_state["pending_admin_settings"] = None
if "settings_flash_message" not in st.session_state:
    st.session_state["settings_flash_message"] = None
if "admin_change_log" not in st.session_state:
    st.session_state["admin_change_log"] = []

# Apply Settings/Admin changes BEFORE sidebar widgets are created.
# This prevents an older widget value from immediately overwriting a
# newly applied administrator setting on the next Streamlit rerun.
_pending_admin = st.session_state.get("pending_admin_settings")
if _pending_admin:
    for _key, _value in _pending_admin.items():
        st.session_state[_key] = _value

    # Keep the sidebar forecast selector synchronized with the administrator
    # forecast-horizon setting.
    if "forecast_horizon" in _pending_admin:
        st.session_state["forecast_horizon_selector"] = int(
            _pending_admin["forecast_horizon"]
        )

    # Remove stale Settings/Admin widget values so the controls are rebuilt
    # from the newly applied source-of-truth values.
    for _widget_key in [
        "admin_org_name",
        "admin_portal_title",
        "admin_capacity_kw",
        "admin_warning_percent",
        "admin_default_horizon",
        "admin_display_density",
        "admin_chart_height",
        "admin_display_theme",
        "admin_min_upload_days",
        "admin_retrain_epochs",
        "admin_retrain_batch",
        "admin_retrain_patience",
        "admin_retrain_lr",
        "admin_retrain_delta",
    ]:
        st.session_state.pop(_widget_key, None)

    st.session_state["pending_admin_settings"] = None

# Registry of ALL user-uploaded depots. Each depot is independently removable.
if "uploaded_depots_registry" not in st.session_state:
    st.session_state["uploaded_depots_registry"] = _persisted_uploaded_depots.copy()

# The following fields represent the currently active or currently processed upload.
if "uploaded_df" not in st.session_state: st.session_state["uploaded_df"] = None
if "upload_meta" not in st.session_state: st.session_state["upload_meta"] = None
if "uploaded_depot_name" not in st.session_state: st.session_state["uploaded_depot_name"] = None
if "uploaded_capacity_kw" not in st.session_state: st.session_state["uploaded_capacity_kw"] = None
if "last_activated_upload_id" not in st.session_state: st.session_state["last_activated_upload_id"] = None
if "upload_activation_notice" not in st.session_state: st.session_state["upload_activation_notice"] = None
if "last_evaluated_upload_key" not in st.session_state: st.session_state["last_evaluated_upload_key"] = None
if "uploaded_evaluation_summary" not in st.session_state: st.session_state["uploaded_evaluation_summary"] = None
if "trained_upload_id" not in st.session_state: st.session_state["trained_upload_id"] = None
if "trained_upload_filename" not in st.session_state: st.session_state["trained_upload_filename"] = None
if "active_uploaded_pipeline" not in st.session_state:
    st.session_state["active_uploaded_pipeline"] = None
if "active_uploaded_pipeline_depot" not in st.session_state:
    st.session_state["active_uploaded_pipeline_depot"] = None

if "retraining_completion_summary" not in st.session_state: st.session_state["retraining_completion_summary"] = None
if "data_source_status" not in st.session_state: st.session_state["data_source_status"] = "Real UKPN Optimise Prime bundled data"

st.markdown("""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif !important; background-color: #F8FAFC !important; color: #1E293B !important; }
 [data-testid="stSidebar"] { background-color: #0F172A !important; }
 [data-testid="stSidebar"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #E2E8F0 !important; }
 [data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 600 !important; color: #0F172A !important; }
 [data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 500 !important; color: #64748B !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
 [data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; padding: 10px 14px !important; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; }
 .stButton>button { background-color: #2563EB !important; color: white !important; border-radius: 6px !important; font-weight: 500 !important; border: none !important; }
 .commercial-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #FFFFFF; border-top: 1px solid #E2E8F0; padding: 6px 20px; text-align: right; font-size: 11px; color: #64748B; z-index: 999; }
 .status-badge { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; }
 .badge-safe { background-color: #DCFCE7; color: #15803D; }
 .badge-warning { background-color: #FEF9C3; color: #A16207; }
 .badge-danger { background-color: #FEE2E2; color: #B91C1C; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   OPTIVOLT PREMIUM ENTERPRISE DASHBOARD — VISUAL OVERRIDES ONLY
   ================================================================ */
:root {
    --ov-navy:#071426;
    --ov-navy2:#0A1D33;
    --ov-blue:#2563EB;
    --ov-cyan:#06B6D4;
    --ov-teal:#14B8A6;
    --ov-slate:#0F172A;
    --ov-muted:#64748B;
    --ov-border:#DCE5EF;
    --ov-bg:#F4F7FB;
    --ov-white:#FFFFFF;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 86% 2%, rgba(6,182,212,.08), transparent 22rem),
        radial-gradient(circle at 27% 5%, rgba(37,99,235,.055), transparent 27rem),
        var(--ov-bg) !important;
}
[data-testid="stHeader"] {
    background: rgba(244,247,251,.86) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(220,229,239,.78);
}
.block-container {
    max-width: 1500px !important;
    padding-top: 2rem !important;
    padding-bottom: 4.2rem !important;
}
h1 {
    color: var(--ov-slate) !important;
    font-weight: 850 !important;
    font-size: clamp(2rem,3.1vw,3rem) !important;
    line-height: 1.05 !important;
    letter-spacing: -.035em !important;
}
h2, h3 {
    color: var(--ov-slate) !important;
    font-weight: 760 !important;
    letter-spacing: -.02em !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(20,184,166,.18), transparent 18rem),
        linear-gradient(180deg,#071426 0%,#0A1D33 55%,#0B213A 100%) !important;
    border-right: 1px solid rgba(148,163,184,.13) !important;
    box-shadow: 14px 0 40px rgba(15,23,42,.08);
}
[data-testid="stSidebar"] h3 { color:#F8FAFC !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color:#D8E4EF !important; }
[data-testid="stSidebar"] hr {
    border-color:rgba(148,163,184,.16) !important;
    margin:1.15rem 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:rgba(255,255,255,.075) !important;
    border:1px solid rgba(148,163,184,.24) !important;
    border-radius:12px !important;
    min-height:46px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color:rgba(45,212,191,.58) !important;
    background:rgba(255,255,255,.105) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap:.32rem !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    border:1px solid transparent !important;
    border-radius:11px !important;
    padding:.46rem .58rem !important;
    margin-bottom:.1rem !important;
    transition:all .18s ease !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:rgba(255,255,255,.075) !important;
    border-color:rgba(148,163,184,.16) !important;
    transform:translateX(2px);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(90deg,rgba(37,99,235,.27),rgba(20,184,166,.14)) !important;
    border-color:rgba(45,212,191,.36) !important;
    box-shadow:inset 3px 0 0 #2DD4BF,0 8px 20px rgba(2,6,23,.12);
}

/* Cards, metrics, bordered containers */
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background:rgba(255,255,255,.97) !important;
    border:1px solid rgba(203,213,225,.72) !important;
    border-radius:16px !important;
    box-shadow:0 9px 30px rgba(15,23,42,.055) !important;
}
[data-testid="stMetric"] {
    padding:1rem 1.05rem !important;
    min-height:108px !important;
    transition:transform .18s ease,box-shadow .18s ease !important;
}
[data-testid="stMetric"]:hover {
    transform:translateY(-2px);
    box-shadow:0 14px 35px rgba(15,23,42,.085) !important;
}
[data-testid="stMetricValue"] {
    color:#0F172A !important;
    font-size:1.55rem !important;
    font-weight:850 !important;
    letter-spacing:-.025em !important;
}
[data-testid="stMetricLabel"] {
    color:#64748B !important;
    font-size:.71rem !important;
    font-weight:800 !important;
    text-transform:uppercase !important;
    letter-spacing:.075em !important;
}
div[data-testid="stMetricLabel"] p {
    font-size:.71rem !important;
    line-height:1.25 !important;
    white-space:normal !important;
}

/* Plotly */
[data-testid="stPlotlyChart"] {
    background:rgba(255,255,255,.985);
    border:1px solid rgba(203,213,225,.72);
    border-radius:18px;
    padding:.42rem .48rem .2rem;
    box-shadow:0 10px 34px rgba(15,23,42,.055);
    overflow:hidden;
    margin-bottom:.8rem;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border:0 !important;
    border-radius:11px !important;
    min-height:42px !important;
    font-weight:780 !important;
    color:#FFFFFF !important;
    background:linear-gradient(135deg,#2563EB 0%,#0EA5E9 52%,#14B8A6 100%) !important;
    box-shadow:0 8px 22px rgba(37,99,235,.20) !important;
    transition:all .16s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform:translateY(-1px);
    box-shadow:0 12px 27px rgba(37,99,235,.27) !important;
}
.stButton > button:disabled {
    opacity:.5 !important;
    transform:none !important;
    box-shadow:none !important;
}

/* Forms and tabs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stFileUploader"] section,
textarea { border-radius:11px !important; }

[data-baseweb="tab-list"] {
    gap:.35rem !important;
    border-bottom:1px solid #E2E8F0 !important;
}
[data-baseweb="tab"] {
    border-radius:10px 10px 0 0 !important;
    font-weight:700 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background:rgba(37,99,235,.075) !important;
}

/* Dataframes / alerts */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius:14px !important;
    overflow:hidden !important;
    border:1px solid rgba(203,213,225,.75) !important;
    box-shadow:0 8px 24px rgba(15,23,42,.04);
}
[data-testid="stAlert"] {
    border-radius:13px !important;
    box-shadow:0 6px 22px rgba(15,23,42,.035);
}

/* Progress */
[data-testid="stProgress"] > div > div > div {
    background:linear-gradient(90deg,#2563EB,#06B6D4,#14B8A6) !important;
}

/* Premium reusable blocks */
.ov-hero {
    position:relative;
    overflow:hidden;
    background:
        radial-gradient(circle at 92% 10%,rgba(45,212,191,.24),transparent 16rem),
        linear-gradient(120deg,#0A1D33 0%,#123B65 55%,#0E7490 100%);
    color:white;
    border:1px solid rgba(148,163,184,.17);
    border-radius:22px;
    padding: 14px 16px;
    margin:.4rem 0 1.25rem;
    box-shadow:0 18px 48px rgba(15,23,42,.16);
}
.ov-kicker {
    color:#99F6E4;
    font-size:.75rem;
    font-weight:850;
    letter-spacing:.12em;
    text-transform:uppercase;
    margin-bottom:.45rem;
}
.ov-title {
    color:#FFFFFF;
    font-size:clamp(1.75rem,2.7vw,2.5rem);
    font-weight:880;
    letter-spacing:-.035em;
    line-height:1.05;
    margin-bottom:.55rem;
}
.ov-subtitle {
    color:#DCEAF5;
    font-size:.96rem;
    max-width:900px;
    margin:0;
}
.ov-summary {
    background:rgba(255,255,255,.98);
    border:1px solid rgba(203,213,225,.72);
    border-radius:17px;
    padding:17px 19px;
    margin-bottom:1.1rem;
    box-shadow:0 9px 30px rgba(15,23,42,.05);
}
.ov-summary-grid {
    display:grid;
    grid-template-columns:1.2fr .9fr 1.15fr;
    gap:14px;
}
.ov-label {
    color:#64748B;
    font-size:.68rem;
    font-weight:850;
    letter-spacing:.08em;
    text-transform:uppercase;
    margin-bottom:.42rem;
}
.ov-value {
    color:#0F172A;
    font-size:.98rem;
    font-weight:780;
}
.status-badge {
    display:inline-flex !important;
    align-items:center;
    padding:5px 10px !important;
    border-radius:999px !important;
    font-size:.73rem !important;
    font-weight:850 !important;
}
.badge-safe { background:#DCFCE7 !important; color:#166534 !important; border:1px solid #BBF7D0; }
.badge-warning { background:#FEF3C7 !important; color:#92400E !important; border:1px solid #FDE68A; }
.badge-danger { background:#FEE2E2 !important; color:#991B1B !important; border:1px solid #FECACA; }

/* Existing home KPI cards upgraded */
.home-metric-grid {
    gap:16px !important;
    margin:16px 0 28px !important;
}
.home-metric-card {
    position:relative !important;
    overflow:hidden !important;
    background:linear-gradient(180deg,#FFFFFF 0%,#FBFDFF 100%) !important;
    border:1px solid rgba(203,213,225,.75) !important;
    border-radius:17px !important;
    padding:18px 19px 17px !important;
    min-height:116px !important;
    box-shadow:0 9px 28px rgba(15,23,42,.055) !important;
    transition:all .18s ease !important;
}
.home-metric-card::before {
    content:"";
    position:absolute;
    left:0; top:0; right:0;
    height:4px;
    background:linear-gradient(90deg,#2563EB,#06B6D4,#14B8A6);
}
.home-metric-card:hover {
    transform:translateY(-2px);
    box-shadow:0 14px 34px rgba(15,23,42,.08) !important;
}
.home-metric-label {
    color:#64748B !important;
    font-size:.69rem !important;
    font-weight:850 !important;
    letter-spacing:.08em !important;
    margin-bottom:11px !important;
}
.home-metric-value {
    color:#0F172A !important;
    font-size:1.66rem !important;
    font-weight:880 !important;
    letter-spacing:-.035em !important;
}

/* Footer */
.commercial-footer {
    background:rgba(7,20,38,.95) !important;
    color:#CBD5E1 !important;
    border-top:1px solid rgba(148,163,184,.18) !important;
    padding:7px 20px !important;
    backdrop-filter:blur(12px);
    box-shadow:0 -8px 30px rgba(15,23,42,.08);
}

/* Responsive */
@media(max-width:1000px) {
    .ov-summary-grid { grid-template-columns:1fr; }
}
@media(max-width:620px) {
    .block-container { padding-left:.9rem !important; padding-right:.9rem !important; }
    .ov-hero { padding:20px 18px; border-radius:17px; }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* ================================================================
   COMPACT SPACING OVERRIDES
   Keeps the premium theme but removes excess vertical whitespace.
   ================================================================ */

/* Main page density */
.block-container {
    max-width: 1500px !important;
    padding-top: 1.15rem !important;
    padding-bottom: 2.4rem !important;
}

/* Headings */
h1 {
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.0 !important;
}
h2 {
    margin-top: 0.75rem !important;
    margin-bottom: 0.35rem !important;
}
h3 {
    margin-top: 0.55rem !important;
    margin-bottom: 0.3rem !important;
}

/* General paragraph/caption spacing */
p {
    margin-top: 0.18rem !important;
    margin-bottom: 0.42rem !important;
}
[data-testid="stCaptionContainer"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.45rem !important;
}

/* Reduce Streamlit block gaps */
[data-testid="stVerticalBlock"] {
    gap: 0.55rem !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
}

/* Premium hero tighter */
.ov-hero {
    padding: 18px 22px !important;
    margin: 0.15rem 0 0.75rem !important;
    border-radius: 18px !important;
}
.ov-kicker {
    margin-bottom: 0.25rem !important;
}
.ov-title {
    margin-bottom: 0.3rem !important;
    font-size: clamp(1.65rem, 2.35vw, 2.2rem) !important;
}
.ov-subtitle {
    font-size: 0.9rem !important;
    line-height: 1.4 !important;
}

/* Summary card tighter */
.ov-summary {
    padding: 13px 16px !important;
    margin-bottom: 0.7rem !important;
    border-radius: 14px !important;
}
.ov-summary-grid {
    gap: 10px !important;
}
.ov-label {
    margin-bottom: 0.22rem !important;
}

/* KPI cards tighter */
.home-metric-grid {
    gap: 12px !important;
    margin: 10px 0 16px !important;
}
.home-metric-card {
    padding: 14px 16px 13px !important;
    min-height: 92px !important;
    border-radius: 14px !important;
}
.home-metric-label {
    margin-bottom: 7px !important;
}
.home-metric-value {
    font-size: 1.48rem !important;
}

/* Streamlit metric cards tighter */
[data-testid="stMetric"] {
    padding: 0.75rem 0.9rem !important;
    min-height: 88px !important;
    border-radius: 14px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.42rem !important;
}
[data-testid="stMetricLabel"] {
    margin-bottom: 0.15rem !important;
}

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding-top: 0.55rem !important;
    padding-bottom: 0.55rem !important;
}

/* Plotly chart spacing */
[data-testid="stPlotlyChart"] {
    margin-top: 0.15rem !important;
    margin-bottom: 0.45rem !important;
    padding: 0.25rem 0.35rem 0.12rem !important;
    border-radius: 15px !important;
}

/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    margin-top: 0.2rem !important;
    margin-bottom: 0.45rem !important;
    border-radius: 12px !important;
}

/* Alerts */
[data-testid="stAlert"] {
    margin-top: 0.25rem !important;
    margin-bottom: 0.45rem !important;
    padding-top: 0.65rem !important;
    padding-bottom: 0.65rem !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    margin-bottom: 0.4rem !important;
}

/* Dividers */
hr {
    margin-top: 0.65rem !important;
    margin-bottom: 0.65rem !important;
}

/* Sidebar also a little tighter */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.85rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.18rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding-top: 0.36rem !important;
    padding-bottom: 0.36rem !important;
}

/* Mobile */
@media(max-width:620px) {
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.8rem !important;
    }
    .ov-hero {
        padding: 15px 16px !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   GLOBAL DENSITY REFINEMENT
   Applies across all pages to reduce unnecessary whitespace while
   retaining readability and visual hierarchy.
   ================================================================ */

/* Overall page spacing */
.block-container {
    padding-top: 0.9rem !important;
    padding-bottom: 1.6rem !important;
}

/* Main vertical layout */
[data-testid="stVerticalBlock"] {
    gap: 0.38rem !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.6rem !important;
}

/* Headings and text */
h1 {
    margin-top: 0 !important;
    margin-bottom: 0.22rem !important;
}
h2 {
    margin-top: 0.48rem !important;
    margin-bottom: 0.24rem !important;
}
h3 {
    margin-top: 0.34rem !important;
    margin-bottom: 0.18rem !important;
}
h4 {
    margin-top: 0.28rem !important;
    margin-bottom: 0.16rem !important;
}
p {
    margin-top: 0.08rem !important;
    margin-bottom: 0.26rem !important;
    line-height: 1.45 !important;
}
[data-testid="stCaptionContainer"] {
    margin-top: 0.04rem !important;
    margin-bottom: 0.22rem !important;
}

/* Markdown blocks */
[data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}

/* Metric/KPI cards */
[data-testid="stMetric"] {
    min-height: 78px !important;
    padding: 0.62rem 0.78rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.32rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.66rem !important;
    margin-bottom: 0.08rem !important;
}

/* Home custom KPI cards */
.home-metric-grid {
    gap: 10px !important;
    margin: 7px 0 10px !important;
}
.home-metric-card {
    min-height: 82px !important;
    padding: 12px 14px 11px !important;
}
.home-metric-label {
    margin-bottom: 4px !important;
}
.home-metric-value {
    font-size: 1.36rem !important;
}

/* Custom hero/summary sections */
.ov-hero {
    padding: 14px 18px !important;
    margin: 0 0 0.45rem !important;
}
.ov-kicker {
    margin-bottom: 0.12rem !important;
}
.ov-title {
    margin-bottom: 0.16rem !important;
}
.ov-subtitle {
    line-height: 1.35 !important;
}
.ov-summary {
    padding: 10px 13px !important;
    margin-bottom: 0.42rem !important;
}
.ov-summary-grid {
    gap: 8px !important;
}
.ov-label {
    margin-bottom: 0.12rem !important;
}

/* Generic bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    margin-top: 0.08rem !important;
    margin-bottom: 0.18rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding-top: 0.42rem !important;
    padding-bottom: 0.42rem !important;
}

/* Tables and charts */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    margin-top: 0.08rem !important;
    margin-bottom: 0.24rem !important;
}
[data-testid="stPlotlyChart"] {
    margin-top: 0.06rem !important;
    margin-bottom: 0.28rem !important;
    padding: 0.16rem 0.24rem 0.08rem !important;
}

/* Alerts/info/success/warning */
[data-testid="stAlert"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.24rem !important;
    padding-top: 0.52rem !important;
    padding-bottom: 0.52rem !important;
}

/* Buttons and controls */
.stButton,
.stDownloadButton {
    margin-top: 0.08rem !important;
    margin-bottom: 0.12rem !important;
}
.stButton > button,
.stDownloadButton > button {
    min-height: 38px !important;
}

/* Inputs */
[data-testid="stTextInput"],
[data-testid="stNumberInput"],
[data-testid="stSelectbox"],
[data-testid="stFileUploader"] {
    margin-bottom: 0.18rem !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    margin-bottom: 0.18rem !important;
}
[data-baseweb="tab"] {
    padding-top: 0.52rem !important;
    padding-bottom: 0.52rem !important;
}

/* Expanders */
[data-testid="stExpander"] {
    margin-top: 0.08rem !important;
    margin-bottom: 0.18rem !important;
}

/* Dividers */
hr {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* Sidebar density */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.62rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.08rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding-top: 0.3rem !important;
    padding-bottom: 0.3rem !important;
}

/* Reduce excess whitespace around st.columns groups */
[data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Mobile */
@media(max-width:620px) {
    .block-container {
        padding-top: 0.65rem !important;
        padding-bottom: 1.2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   COLOUR-RICH ENTERPRISE THEME
   Adds visual colour depth without making the dashboard look flashy.
   ================================================================ */

/* Richer page canvas */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(135deg, rgba(37,99,235,.06), rgba(20,184,166,.05) 35%, rgba(99,102,241,.04) 70%),
        #EEF4FA !important;
}

/* Section banding */
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(odd) {
    border-radius: 16px;
}

/* Bordered containers get subtle tinted surfaces */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(241,247,253,.98)) !important;
    border: 1px solid rgba(148,163,184,.32) !important;
}

/* Metric cards alternate richer tints */
[data-testid="stMetric"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,.97), rgba(235,245,255,.95)) !important;
    border-top: 4px solid #3B82F6 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,.97), rgba(235,250,247,.96)) !important;
    border-top-color: #14B8A6 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,.97), rgba(243,240,255,.96)) !important;
    border-top-color: #8B5CF6 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,.97), rgba(255,247,232,.97)) !important;
    border-top-color: #F59E0B !important;
}

/* Home KPI cards each get a richer surface */
.home-metric-card:nth-child(1) {
    background: linear-gradient(145deg,#FFFFFF,#EAF3FF) !important;
}
.home-metric-card:nth-child(2) {
    background: linear-gradient(145deg,#FFFFFF,#E9FBF6) !important;
}
.home-metric-card:nth-child(3) {
    background: linear-gradient(145deg,#FFFFFF,#F3EEFF) !important;
}
.home-metric-card:nth-child(4) {
    background: linear-gradient(145deg,#FFFFFF,#FFF5E5) !important;
}

/* Model comparison score cards */
.perf-card {
    background: linear-gradient(145deg,#FFFFFF,#EEF5FF) !important;
    border: 1px solid #C7DBF7 !important;
    box-shadow: 0 8px 24px rgba(37,99,235,.08) !important;
}
.perf-card-grid .perf-card:nth-child(2) {
    background: linear-gradient(145deg,#FFFFFF,#EAFBF6) !important;
    border-color:#BDEBDD !important;
}
.perf-card-grid .perf-card:nth-child(3) {
    background: linear-gradient(145deg,#FFFFFF,#F4EEFF) !important;
    border-color:#DCCCFB !important;
}

/* Comparison/result information boxes */
[data-testid="stAlert"] {
    background: linear-gradient(90deg, rgba(236,253,245,.98), rgba(239,246,255,.98)) !important;
    border-left: 5px solid #14B8A6 !important;
}

/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #FFFFFF !important;
    box-shadow: 0 10px 28px rgba(37,99,235,.06) !important;
}

/* Plotly panels on soft blue glass */
[data-testid="stPlotlyChart"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,.98), rgba(244,249,255,.98)) !important;
    border-color: rgba(96,165,250,.28) !important;
}

/* Sidebar selected item stronger */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(37,99,235,.38), rgba(20,184,166,.22)) !important;
    border-color: rgba(45,212,191,.52) !important;
}

/* Select and input surfaces */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background: #F8FBFF !important;
}

/* Section titles get a subtle accent line */
h2, h3 {
    position: relative;
}
h2::after {
    content:"";
    display:block;
    width:54px;
    height:3px;
    margin-top:6px;
    border-radius:999px;
    background:linear-gradient(90deg,#2563EB,#14B8A6);
}

/* Colour-coded model comparison headline cards */
.model-lead-blue {
    background: linear-gradient(135deg,#EAF3FF,#F8FBFF);
    border:1px solid #BFDBFE;
}
.model-lead-green {
    background: linear-gradient(135deg,#EAFBF6,#F8FFFC);
    border:1px solid #BBF7D0;
}
.model-lead-purple {
    background: linear-gradient(135deg,#F3EEFF,#FBF9FF);
    border:1px solid #DDD6FE;
}

/* Footer slightly richer */
.commercial-footer {
    background: linear-gradient(90deg,#071426,#0A2440) !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   LEGIBILITY + COLOUR REFINEMENT
   Fixes truncated metric text and reduces plain-white surfaces.
   ================================================================ */

/* Richer page canvas */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 82% 4%, rgba(14,165,233,.10), transparent 24rem),
        radial-gradient(circle at 22% 8%, rgba(20,184,166,.08), transparent 28rem),
        linear-gradient(135deg, #EEF5FC 0%, #EAF7F6 46%, #F3F0FB 100%) !important;
}

/* Metric cards: slightly smaller type, more room, no ellipsis */
[data-testid="stMetric"] {
    min-height: 108px !important;
    padding: 0.78rem 0.88rem !important;
    border-radius: 16px !important;
    background: linear-gradient(145deg, #F9FCFF 0%, #EAF3FF 100%) !important;
    border: 1px solid rgba(96,165,250,.34) !important;
    border-top: 4px solid #3B82F6 !important;
    overflow: visible !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
    background: linear-gradient(145deg, #FBFFFE 0%, #E8F8F4 100%) !important;
    border-color: rgba(20,184,166,.30) !important;
    border-top-color: #14B8A6 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
    background: linear-gradient(145deg, #FDFBFF 0%, #F1ECFF 100%) !important;
    border-color: rgba(139,92,246,.27) !important;
    border-top-color: #8B5CF6 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFDFC 0%, #FFF3E1 100%) !important;
    border-color: rgba(245,158,11,.28) !important;
    border-top-color: #F59E0B !important;
}

/* Labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    font-size: 0.66rem !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    letter-spacing: .055em !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
}

/* Metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] p {
    font-size: 1.15rem !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    max-width: 100% !important;
}

/* Long metric text specifically gets more breathing room */
[data-testid="stMetricValue"] {
    min-height: 2.5rem !important;
}

/* Tables: less stark white */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: linear-gradient(180deg, #FBFDFF 0%, #F1F7FD 100%) !important;
    border: 1px solid rgba(96,165,250,.24) !important;
    box-shadow: 0 10px 28px rgba(37,99,235,.06) !important;
}

/* Bordered containers use tinted surfaces */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(180deg, rgba(250,253,255,.98), rgba(237,246,252,.98)) !important;
    border: 1px solid rgba(125,166,208,.27) !important;
}

/* Chart panels */
[data-testid="stPlotlyChart"] {
    background:
        linear-gradient(180deg, rgba(251,254,255,.99), rgba(239,247,253,.99)) !important;
    border: 1px solid rgba(59,130,246,.22) !important;
}

/* Alerts use stronger but restrained colour */
[data-testid="stAlert"] {
    background: linear-gradient(90deg, rgba(236,253,245,.98), rgba(239,246,255,.98)) !important;
}

/* Section headings */
h2::after {
    content:"";
    display:block;
    width:62px;
    height:3px;
    margin-top:5px;
    border-radius:999px;
    background:linear-gradient(90deg,#2563EB,#14B8A6);
}

/* Table text slightly smaller for better fit */
[data-testid="stTable"] table,
[data-testid="stDataFrame"] {
    font-size: 0.86rem !important;
}

/* On narrower screens, shrink metric text further */
@media (max-width: 1100px) {
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] p {
        font-size: 1.02rem !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        font-size: 0.61rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   WRAP-SAFE INFO CARDS
   Replaces long-text st.metric cards that can truncate with ellipses.
   ================================================================ */
.ov-info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 8px 0 10px 0;
}
.ov-info-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 8px 0 10px 0;
}
.ov-info-card {
    min-width: 0;
    background: linear-gradient(145deg, #F9FCFF 0%, #EAF3FF 100%);
    border: 1px solid rgba(59,130,246,.26);
    border-top: 4px solid #3B82F6;
    border-radius: 16px;
    padding: 15px 17px;
    box-shadow: 0 9px 26px rgba(15,23,42,.06);
}
.ov-info-card:nth-child(2) {
    background: linear-gradient(145deg, #FBFFFE 0%, #E8F8F4 100%);
    border-color: rgba(20,184,166,.28);
    border-top-color: #14B8A6;
}
.ov-info-card:nth-child(3) {
    background: linear-gradient(145deg, #FDFBFF 0%, #F1ECFF 100%);
    border-color: rgba(139,92,246,.24);
    border-top-color: #8B5CF6;
}
.ov-info-card:nth-child(4) {
    background: linear-gradient(145deg, #FFFDFC 0%, #FFF3E1 100%);
    border-color: rgba(245,158,11,.26);
    border-top-color: #F59E0B;
}
.ov-info-label {
    color: #64748B;
    font-size: .68rem;
    line-height: 1.18;
    font-weight: 800;
    letter-spacing: .07em;
    text-transform: uppercase;
    margin-bottom: 8px;
    white-space: normal;
    overflow-wrap: anywhere;
}
.ov-info-value {
    color: #0F172A;
    font-size: 1.00rem;
    line-height: 1.24;
    font-weight: 760;
    letter-spacing: -.02em;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
}
.ov-info-value-small {
    font-size: 0.90rem;
    line-height: 1.26;
    font-weight: 740;
}
@media (max-width: 1100px) {
    .ov-info-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
    .ov-info-grid,
    .ov-info-grid-4 { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   FINAL MODEL COMPARISON TABLE
   Blue header, restrained alternating rows, pale-teal selected LSTM row.
   ================================================================ */
.ov-model-table-wrap {
    width: 100%;
    overflow-x: auto;
    border-radius: 14px;
    border: 1px solid rgba(59,130,246,.24);
    box-shadow: 0 10px 28px rgba(37,99,235,.07);
    margin: 8px 0 12px 0;
    background: #F8FBFF;
}

.ov-model-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: auto;
    font-size: 0.86rem;
    color: #0F172A;
}

.ov-model-table thead th {
    background: linear-gradient(180deg, #1E4F7A 0%, #173E63 100%);
    color: #FFFFFF;
    font-weight: 800;
    text-align: left;
    padding: 11px 12px;
    border-right: 1px solid rgba(255,255,255,.12);
    white-space: nowrap;
}

.ov-model-table thead th:first-child {
    border-top-left-radius: 13px;
}
.ov-model-table thead th:last-child {
    border-top-right-radius: 13px;
    border-right: 0;
}

.ov-model-table tbody td {
    padding: 10px 12px;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    background: rgba(255,255,255,.96);
    vertical-align: middle;
}

.ov-model-table tbody tr:nth-child(even):not(.selected-model-row) td {
    background: #F3F8FD;
}

.ov-model-table tbody tr.selected-model-row td {
    background: linear-gradient(90deg, #DDF7F1 0%, #E9FBF7 100%);
    font-weight: 800;
    color: #0F172A;
    border-top: 1px solid #9DE2D2;
    border-bottom: 1px solid #9DE2D2;
}

.ov-model-table tbody tr.selected-model-row td:first-child {
    box-shadow: inset 4px 0 0 #14B8A6;
}

.ov-model-table tbody td:last-child,
.ov-model-table thead th:last-child {
    border-right: 0;
}

.ov-model-table tbody tr:last-child td:first-child {
    border-bottom-left-radius: 13px;
}
.ov-model-table tbody tr:last-child td:last-child {
    border-bottom-right-radius: 13px;
}

@media(max-width:900px) {
    .ov-model-table {
        font-size: 0.78rem;
    }
    .ov-model-table thead th,
    .ov-model-table tbody td {
        padding: 9px 9px;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ================================================================
   SIDEBAR CONTROL VISIBILITY + CONSISTENT COLOURED TABLES
   ================================================================ */

/* Sidebar control labels remain bright on navy */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #E6EEF7 !important;
    font-weight: 700 !important;
}

/* Select boxes: light surface + dark, readable selected value */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: linear-gradient(180deg, #FFFFFF 0%, #F1F7FD 100%) !important;
    border: 1px solid #8CB9E8 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    box-shadow: 0 5px 16px rgba(2, 6, 23, .16) !important;
}

/* Override earlier sidebar-wide light text rules INSIDE the white select */
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #0F2742 !important;
    -webkit-text-fill-color: #0F2742 !important;
    font-weight: 750 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #2563EB !important;
    color: #2563EB !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #2F80ED !important;
    box-shadow: 0 0 0 3px rgba(47, 128, 237, .15), 0 6px 18px rgba(2,6,23,.16) !important;
}

/* Dropdown menu is rendered outside sidebar, so style the popover too */
div[data-baseweb="popover"] ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1px solid #B8D2EC !important;
    box-shadow: 0 12px 28px rgba(15,23,42,.18) !important;
}
div[data-baseweb="popover"] [role="option"] {
    color: #0F2742 !important;
    background: #FFFFFF !important;
    font-weight: 600 !important;
}
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background: #E8F3FF !important;
    color: #123F6B !important;
}

/* Small horizon confirmation pill */
.ov-sidebar-status-pill {
    display: inline-block;
    margin: 4px 0 10px 0;
    padding: 6px 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(37,99,235,.28), rgba(20,184,166,.22));
    border: 1px solid rgba(94,234,212,.35);
    color: #DDF9F4;
    font-size: .74rem;
    line-height: 1.2;
    letter-spacing: .01em;
}

/* Architecture comparison table */
.ov-architecture-table-wrap {
    width: 100%;
    overflow-x: auto;
    border-radius: 14px;
    border: 1px solid rgba(59,130,246,.24);
    box-shadow: 0 10px 26px rgba(37,99,235,.07);
    margin: 8px 0 12px 0;
}

.ov-architecture-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: .87rem;
    color: #0F172A;
}

.ov-architecture-table thead th {
    padding: 11px 12px;
    text-align: left;
    color: #FFFFFF;
    background: linear-gradient(180deg, #1E4F7A 0%, #173E63 100%);
    font-weight: 800;
    border-right: 1px solid rgba(255,255,255,.13);
    white-space: normal;
}

.ov-architecture-table thead th:first-child {
    border-top-left-radius: 13px;
}
.ov-architecture-table thead th:last-child {
    border-top-right-radius: 13px;
    border-right: 0;
}

/* Metric name column */
.ov-architecture-table tbody td:first-child {
    background: #EAF3FF;
    color: #274C72;
    font-weight: 720;
    width: 32%;
}

/* Baseline column */
.ov-architecture-table tbody td:nth-child(2) {
    background: #F5F9FD;
}

/* Selected LSTM column */
.ov-architecture-table tbody td:nth-child(3) {
    background: #E8F8F4;
    font-weight: 760;
    color: #123D36;
}

.ov-architecture-table tbody td {
    padding: 10px 12px;
    border-right: 1px solid #DCE6EF;
    border-bottom: 1px solid #DCE6EF;
    vertical-align: middle;
}
.ov-architecture-table tbody td:last-child {
    border-right: 0;
}

/* Similar standard tables get a subtle blue header where Streamlit exposes table cells */
[data-testid="stTable"] thead th {
    background: #1E4F7A !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) td {
    background: #F3F8FD !important;
}

/* Inputs on main pages: stronger contrast too */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #F8FBFF !important;
    color: #0F2742 !important;
    -webkit-text-fill-color: #0F2742 !important;
    border-color: #B8D2EC !important;
}

/* Keep report/other dataframe panels softly tinted, not stark white */
[data-testid="stDataFrame"] {
    background: linear-gradient(180deg, #F9FCFF 0%, #EEF5FB 100%) !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* High-contrast sidebar selectors */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F8FBFF !important;
    border: 1px solid #78A7D6 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #0B2540 !important;
    -webkit-text-fill-color: #0B2540 !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #2563EB !important;
    fill: #2563EB !important;
}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] * {
    color: #0B2540 !important;
    -webkit-text-fill-color: #0B2540 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Apply administrator-selected display preferences.
_display_density = st.session_state.get("display_density", "Compact")
_display_theme = st.session_state.get("display_theme", "Optivolt Blue / Teal")

if _display_density == "Compact":
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: .75rem !important;
            padding-bottom: 1.35rem !important;
            max-width: 1450px !important;
        }
        [data-testid="stVerticalBlock"] { gap: .34rem !important; }
        [data-testid="stMetric"] { padding: .35rem .45rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            max-width: 1250px !important;
        }
        [data-testid="stVerticalBlock"] { gap: .95rem !important; }
        [data-testid="stMetric"] { padding: .8rem .8rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

if _display_theme == "High Contrast Navy":
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg,#D9E7F4 0%,#E4EEF7 48%,#DDEFEA 100%) !important;
        }
        .block-container h1,
        .block-container h2,
        .block-container h3 {
            color:#071A2E !important;
        }
        div[data-testid="stMetric"],
        [data-testid="stDataFrame"],
        .ov-info-card,
        .ov-card,
        .ov-model-table-wrap,
        .ov-architecture-table-wrap {
            border-color:#6B8FB4 !important;
            box-shadow:0 8px 20px rgba(7,26,46,.12) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#06172A 0%,#0A2340 100%) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg,#F2F8FD 0%,#EDF7F8 58%,#F5F8FC 100%) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.markdown("""
<style>
/* ================================================================
   GLOBAL STATUS COLOUR SYSTEM
   green = safe | amber = warning | red = breach | blue = info |
   teal = selected / model-positive
   ================================================================ */
.ov-status-strip {
    display:grid;
    grid-template-columns:repeat(5,minmax(0,1fr));
    gap:10px;
    margin:10px 0 18px 0;
}
.ov-status-card {
    border-radius:12px;
    padding:11px 13px;
    border:1px solid #D7E3EC;
    min-height:84px;
    box-shadow:0 5px 14px rgba(15,23,42,.05);
}
.ov-status-card .label {
    font-size:.70rem;
    text-transform:uppercase;
    letter-spacing:.06em;
    font-weight:800;
    margin-bottom:6px;
}
.ov-status-card .value {
    font-size:1.08rem;
    line-height:1.15;
    font-weight:850;
}
.ov-safe { background:#EAF8F0; border-color:#A9DEC0; color:#17643A; }
.ov-warning { background:#FFF6DD; border-color:#F4D487; color:#805B09; }
.ov-danger { background:#FDECEC; border-color:#F3B4B4; color:#9B1C1C; }
.ov-info { background:#EAF3FF; border-color:#B9D5F3; color:#174F7A; }
.ov-model { background:#E7F8F4; border-color:#A9E2D4; color:#145A50; }

.ov-explanation-panel {
    border-radius:14px;
    padding:14px 16px;
    border:1px solid #B9D5F3;
    background:linear-gradient(135deg,#EEF6FF,#E9F8F5);
    margin:10px 0 16px 0;
}
.ov-explanation-panel .title {
    color:#164D75;
    font-weight:850;
    font-size:.92rem;
    margin-bottom:5px;
}
.ov-explanation-panel .body {
    color:#29475F;
    line-height:1.55;
    font-size:.90rem;
}

.ov-quality-grid {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px;
    margin:10px 0 14px 0;
}
.ov-quality-card {
    background:#F8FBFF;
    border:1px solid #C9DDED;
    border-radius:12px;
    padding:10px 12px;
}
.ov-quality-card .q-label {
    color:#56728A;
    text-transform:uppercase;
    font-size:.67rem;
    font-weight:800;
    letter-spacing:.05em;
}
.ov-quality-card .q-value {
    color:#102A43;
    font-size:1rem;
    font-weight:850;
    margin-top:4px;
}
@media(max-width:950px) {
    .ov-status-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .ov-quality-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
</style>
""", unsafe_allow_html=True)

if not st.session_state["authenticated"]:
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ov-hero" style="text-align:center; margin-bottom:22px;">
            <div class="ov-kicker">Optivolt Solutions Limited</div>
            <div class="ov-title">⚡ {st.session_state["portal_title"]}</div>
            <p class="ov-subtitle" style="margin:0 auto;">
                Predictive demand forecasting and capacity planning for electric vehicle depot operations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h4 style='color:#1E293B;'>Sign in to your account</h4>", unsafe_allow_html=True)
            st.text_input("Username or corporate email", value="operations@optivolt.co.uk")
            st.text_input("Password", type="password", value="••••••••••••")
            if st.button("Authenticate Workspace Access", use_container_width=True):
                st.session_state["authenticated"] = True
                st.rerun()
        st.stop()

with st.sidebar:
    st.markdown(
        f"""
        <div style="
            margin:2px 0 16px 0;
            padding:15px 15px 14px;
            border:1px solid rgba(148,163,184,.16);
            border-radius:15px;
            background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(20,184,166,.08));
            box-shadow:0 12px 28px rgba(2,6,23,.12);">
            <div style="color:#5EEAD4;font-size:.68rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px;">
                {st.session_state.get('portal_title', 'Optivolt EV Depot Intelligence Portal')}
            </div>
            <div style="color:#FFFFFF;font-size:1.02rem;font-weight:800;line-height:1.2;margin-bottom:6px;">
                {st.session_state['org_name']}
            </div>
            <div style="color:#BFD0E2;font-size:.76rem;line-height:1.35;">
                UK Power Networks Optimise Prime
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Built-in project depots are protected; uploaded depots are appended.
    uploaded_registry = st.session_state.get("uploaded_depots_registry", {})
    uploaded_depot_names = sorted(uploaded_registry.keys())
    depot_options = REAL_DEPOTS.copy() + [
        name for name in uploaded_depot_names if name not in REAL_DEPOTS
    ]

    if st.session_state["current_depot"] not in depot_options:
        st.session_state["current_depot"] = "Bexleyheath"

    current_index = depot_options.index(st.session_state["current_depot"])

    selected_depot = st.selectbox(
        "Select Active Depot Node:",
        depot_options,
        index=current_index,
    )

    if selected_depot in uploaded_registry:
        st.caption(f"Uploaded depot active: {selected_depot}")

    horizon_options = [2, 4, 6, 12, 24]
    previous_horizon = int(st.session_state.get("forecast_horizon", 24))
    if previous_horizon not in horizon_options:
        previous_horizon = 24
        st.session_state["forecast_horizon"] = 24

    # Initialise or re-synchronise the selector before it is rendered.
    if "forecast_horizon_selector" not in st.session_state:
        st.session_state["forecast_horizon_selector"] = previous_horizon
    elif int(st.session_state["forecast_horizon_selector"]) not in horizon_options:
        st.session_state["forecast_horizon_selector"] = previous_horizon

    selected_horizon = st.selectbox(
        "Forecast Horizon:",
        horizon_options,
        format_func=lambda h: f"{h} hours",
        help="Choose how many hours of the recursive LSTM forecast to display.",
        key="forecast_horizon_selector",
    )

    selected_horizon = int(selected_horizon)
    if selected_horizon != int(st.session_state.get("forecast_horizon", 24)):
        st.session_state["forecast_horizon"] = selected_horizon
        # The model itself is not retrained; only the recursive forecast
        # horizon is regenerated.
        execute_production_ml_pipeline.clear()

    st.markdown(
        f"""
        <div class="ov-sidebar-status-pill">
            Active horizon: <strong>{st.session_state['forecast_horizon']} hours</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_depot != st.session_state["current_depot"]:
        st.session_state["current_depot"] = selected_depot

    if selected_depot in DEPOT_CAPACITY_KW:
        st.session_state["uploaded_df"] = None
        st.session_state["uploaded_depot_name"] = None
        st.session_state["uploaded_capacity_kw"] = None

        depot_override = st.session_state.get("depot_threshold_overrides", {}).get(
            selected_depot
        )

        if depot_override:
            st.session_state["grid_limit"] = float(depot_override["capacity_kw"])
            st.session_state["warn_threshold"] = float(depot_override["warning_kw"])
        else:
            st.session_state["grid_limit"] = float(DEPOT_CAPACITY_KW[selected_depot])
            st.session_state["warn_threshold"] = round(
                WARNING_ALPHA * float(DEPOT_CAPACITY_KW[selected_depot]), 1
            )
    elif selected_depot in uploaded_registry:
        selected_meta = uploaded_registry[selected_depot]
        selected_uploaded_df = load_uploaded_depot_dataframe(
            selected_depot,
            uploaded_registry,
        )
        if selected_uploaded_df is None:
            st.error(
                f"Stored telemetry for {selected_depot} could not be loaded."
            )
            st.stop()

        st.session_state["uploaded_df"] = selected_uploaded_df
        st.session_state["uploaded_depot_name"] = selected_depot
        st.session_state["uploaded_capacity_kw"] = float(
            selected_meta["capacity_kw"]
        )
        st.session_state["grid_limit"] = float(
            selected_meta["capacity_kw"]
        )
        st.session_state["warn_threshold"] = float(
            selected_meta.get(
                "warning_kw",
                round(WARNING_ALPHA * float(selected_meta["capacity_kw"]), 1),
            )
        )
    st.markdown("---")
    nav_selection = st.radio("Navigation Menu:", ["🏠 Home / Overview", "🏢 Depots Setup", "📊 Upload Depot Data", "📈 Demand Forecast", "🚗 Scenario Analysis", "⚙️ Model Performance", "🚨 System Alerts", "📋 Reports", "📜 Archive History", "⚙️ Settings / Admin"])

try:
    with st.spinner(f"Preparing {st.session_state['current_depot']} forecast from real Optimise Prime data..."):
        active_uploaded_df = (
            st.session_state.get("uploaded_df")
            if st.session_state["current_depot"]
            in st.session_state.get("uploaded_depots_registry", {})
            else None
        )

        active_pipeline_in_session = (
            st.session_state.get("active_uploaded_pipeline")
            if (
                active_uploaded_df is not None
                and st.session_state.get("active_uploaded_pipeline_depot")
                == st.session_state["current_depot"]
            )
            else None
        )

        if active_pipeline_in_session is not None:
            pipeline = dict(active_pipeline_in_session)
            requested_horizon = int(st.session_state["forecast_horizon"])
            current_pipeline_horizon = int(
                pipeline.get("metadata", {}).get("forecast_horizon_hours", requested_horizon)
            )
            if current_pipeline_horizon != requested_horizon:
                refreshed_times, refreshed_forecast = _recursive_forecast(
                    pipeline["model"],
                    pipeline["history"],
                    pipeline["feature_cols"],
                    requested_horizon,
                )
                pipeline["future_times"] = refreshed_times
                pipeline["future_forecast"] = refreshed_forecast
                pipeline["metadata"] = dict(pipeline["metadata"])
                pipeline["metadata"]["forecast_horizon_hours"] = requested_horizon
        else:
            pipeline = execute_production_ml_pipeline(
                st.session_state["current_depot"],
                float(st.session_state["grid_limit"]),
                int(st.session_state["forecast_horizon"]),
                active_uploaded_df,
            )
except Exception as exc:
    st.error(str(exc))
    st.info("Real Optimise Prime data is loaded from the depot CSV files included in this GitHub repository.")
    st.stop()

model_meta = pipeline["metadata"]
history_series = pipeline["history"]
y_test = pipeline["y_test"]
xgb_predictions = pipeline["xgb_preds"]
baseline_predictions = pipeline["baseline_preds"]
future_times = pipeline["future_times"]
horizon_forecast = pipeline["future_forecast"]

current_demand = float(history_series.iloc[-1])  # latest observed historical demand
predicted_peak = float(np.max(horizon_forecast))

# Empirical forecast uncertainty from untouched held-out residuals.
heldout_residuals = np.asarray(y_test.values, dtype=float) - np.asarray(xgb_predictions, dtype=float)
finite_residuals = heldout_residuals[np.isfinite(heldout_residuals)]
if finite_residuals.size >= 20:
    residual_q10, residual_q90 = np.quantile(finite_residuals, [0.10, 0.90])
else:
    fallback_rmse = float(model_meta.get("metrics_xgb", {}).get("rmse", 0.0) or 0.0)
    residual_q10, residual_q90 = -fallback_rmse, fallback_rmse

forecast_lower = np.clip(np.asarray(horizon_forecast, dtype=float) + residual_q10, 0, None)
forecast_upper = np.clip(np.asarray(horizon_forecast, dtype=float) + residual_q90, 0, None)
peak_index = int(np.argmax(horizon_forecast))
forecast_peak_lower = float(forecast_lower[peak_index])
forecast_peak_upper = float(forecast_upper[peak_index])

model_version = _model_version_id(st.session_state["current_depot"], model_meta)

model_mae = float(model_meta.get("metrics_xgb", {}).get("mae", np.nan))
model_rmse = float(model_meta.get("metrics_xgb", {}).get("rmse", np.nan))
baseline_mae = float(model_meta.get("metrics_baseline", {}).get("mae", np.nan))
baseline_rmse = float(model_meta.get("metrics_baseline", {}).get("rmse", np.nan))
if np.isfinite(model_mae) and np.isfinite(model_rmse) and np.isfinite(baseline_mae) and np.isfinite(baseline_rmse):
    if model_mae < baseline_mae and model_rmse < baseline_rmse:
        model_health_status = "Performing above baseline"
    elif model_mae < baseline_mae or model_rmse < baseline_rmse:
        model_health_status = "Mixed relative to baseline"
    else:
        model_health_status = "Review recommended"
else:
    model_health_status = "Performance status unavailable"

try:
    model_r2 = float(r2_score(np.asarray(y_test.values, dtype=float), np.asarray(xgb_predictions, dtype=float)))
except Exception:
    model_r2 = float("nan")

future_times_index = pd.DatetimeIndex(pd.to_datetime(future_times))
breaches = np.where(horizon_forecast >= st.session_state["grid_limit"])[0]
warnings = np.where(horizon_forecast >= st.session_state["warn_threshold"])[0]

breach_runs = _consecutive_index_runs(breaches)
total_breach_minutes = int(len(breaches) * SAMPLING_MINUTES)
if breach_runs:
    first_breach_start_idx, first_breach_end_idx = breach_runs[0]
    first_breach_start_time = future_times_index[first_breach_start_idx]
    first_breach_end_time = future_times_index[first_breach_end_idx] + pd.Timedelta(minutes=SAMPLING_MINUTES)
    first_breach_duration_minutes = int(
        (first_breach_end_idx - first_breach_start_idx + 1) * SAMPLING_MINUTES
    )
else:
    first_breach_start_time = None
    first_breach_end_time = None
    first_breach_duration_minutes = 0

operational_recommendation = _operational_recommendation(
    predicted_peak,
    float(st.session_state["warn_threshold"]),
    float(st.session_state["grid_limit"]),
    first_breach_start_time,
    first_breach_end_time,
)

if len(breaches) > 0:
    op_status, status_badge = "CRITICAL RISK", "badge-danger"
    breach_minutes = int((breaches[0] + 1) * SAMPLING_MINUTES)
    breach_time = f"+{breach_minutes / 60:.2f} Hours"
    first_warning = warnings[0] if len(warnings) > 0 else breaches[0]
    lead_minutes = max(0, (breaches[0] - first_warning) * SAMPLING_MINUTES)
    lead_time = f"{lead_minutes / 60:.2f} Hours"
elif len(warnings) > 0:
    op_status, status_badge = "WARNING THRESHOLD FORECAST", "badge-warning"
    breach_time = "No capacity breach predicted"
    lead_time = "N/A"
else:
    op_status, status_badge = "OPTIMAL SAFE", "badge-safe"
    breach_time = "No capacity breach predicted"
    lead_time = "No breach predicted"

if nav_selection == "🏠 Home / Overview":
    st.markdown(
        f"""
        <div class="ov-hero">
            <div class="ov-kicker">Depot Operations Intelligence</div>
            <div class="ov-title">🏢 {st.session_state['current_depot']} Operations Dashboard</div>
            <p class="ov-subtitle">
                Demand forecasting, capacity risk visibility and operational planning powered by
                <strong>{model_meta['model_type']}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Forecast status is based on the historical project dataset and does not represent live grid telemetry.")

    st.markdown(
        f"""
        <div class="ov-summary">
            <div class="ov-summary-grid">
                <div>
                    <div class="ov-label">Organisation</div>
                    <div class="ov-value">{st.session_state['org_name']}</div>
                </div>
                <div>
                    <div class="ov-label">Operational Status</div>
                    <div class="ov-value"><span class="status-badge {status_badge}">{op_status}</span></div>
                </div>
                <div>
                    <div class="ov-label">Expected Capacity Breach</div>
                    <div class="ov-value">{breach_time}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    current_demand_text = f"{round(current_demand, 1)} kW"
    predicted_peak_text = f"{round(predicted_peak, 1)} kW"
    planning_threshold_text = f"{float(st.session_state['grid_limit']):.1f} kW"
    warning_lead_text = str(lead_time)

    st.markdown(
        f"""
        <style>
        .home-metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 18px;
            margin: 18px 0 28px 0;
        }}
        .home-metric-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 20px;
            min-height: 108px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .home-metric-label {{
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.18;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            margin-bottom: 8px;
        }}
        .home-metric-value {{
            color: #0F172A;
            font-size: 1.58rem;
            font-weight: 700;
            line-height: 1.12;
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        @media (max-width: 950px) {{
            .home-metric-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 560px) {{
            .home-metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>

        <div class="home-metric-grid">
            <div class="home-metric-card">
                <div class="home-metric-label">Latest Observed Demand</div>
                <div class="home-metric-value">{current_demand_text}</div>
            </div>
            <div class="home-metric-card">
                <div class="home-metric-label">Predicted Peak Demand</div>
                <div class="home-metric-value">{predicted_peak_text}</div>
            </div>
            <div class="home-metric-card">
                <div class="home-metric-label">Planning Capacity Threshold</div>
                <div class="home-metric-value">{planning_threshold_text}</div>
            </div>
            <div class="home-metric-card">
                <div class="home-metric-label">Warning Lead Time</div>
                <div class="home-metric-value">{warning_lead_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])

    def build_home_history_chart(series: pd.Series, xaxis_title: str, peak_label: str = "Historical Peak", show_peak: bool = True) -> go.Figure:
        series = series.dropna().sort_index()
        hist_x = series.index
        hist_y = series.to_numpy(dtype=float)
        peak_time = series.idxmax()
        peak_value = float(series.max())

        fig_home = go.Figure()
        fig_home.add_trace(
            go.Scatter(
                x=hist_x,
                y=hist_y,
                name="Actual Demand",
                mode="lines",
                line=dict(color="#2563EB", width=2.4),
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual demand: %{y:.2f} kW<extra></extra>",
            )
        )
        fig_home.add_trace(
            go.Scatter(
                x=[hist_x[0], hist_x[-1]],
                y=[warning_threshold, warning_threshold],
                name="Warning Threshold",
                mode="lines",
                line=dict(color="#F59E0B", width=1.8, dash="dash"),
                hovertemplate=f"Warning threshold: {warning_threshold:.1f} kW<extra></extra>",
            )
        )
        fig_home.add_trace(
            go.Scatter(
                x=[hist_x[0], hist_x[-1]],
                y=[planning_threshold, planning_threshold],
                name="Planning Capacity Threshold",
                mode="lines",
                line=dict(color="#EF4444", width=1.8, dash="dash"),
                hovertemplate=f"Planning capacity threshold: {planning_threshold:.1f} kW<extra></extra>",
            )
        )
        if show_peak:
            fig_home.add_trace(
                go.Scatter(
                    x=[peak_time],
                    y=[peak_value],
                    name=peak_label,
                    mode="markers",
                    marker=dict(size=8, symbol="diamond", color="#7C3AED"),
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>Peak demand: %{y:.2f} kW<extra></extra>",
                )
            )
        fig_home.update_layout(
            template="plotly_white",
            height=430,
            xaxis_title=xaxis_title,
            yaxis_title="Demand (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            margin=dict(l=25, r=25, t=70, b=45),
            hovermode="x unified",
        )
        fig_home.update_yaxes(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
        )
        fig_home.update_xaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.12)",
        )
        return fig_home

    st.subheader("Historical Headroom Footprint Profile")
    st.caption(
        "Show both a short operational view and the full multi-month held-out test profile "
        "for the selected depot."
    )

    recent_lookback = min(192, len(history_series))
    recent_series = history_series.iloc[-recent_lookback:]
    full_test_series = y_test.copy()

    recent_start_label = recent_series.index.min().strftime("%d %b %Y")
    recent_end_label = recent_series.index.max().strftime("%d %b %Y")
    heldout_start_label = full_test_series.index.min().strftime("%d %b %Y")
    heldout_end_label = full_test_series.index.max().strftime("%d %b %Y")

    st.markdown(
        f"**Recent operational view — {recent_start_label} to {recent_end_label}**"
    )
    recent_max_value = float(recent_series.max())
    recent_max_time = recent_series.idxmax()

    st.caption(
        f"Latest {recent_lookback} fifteen-minute intervals for "
        f"{st.session_state['current_depot']}. "
        f"Recent-window peak: {recent_max_value:.2f} kW "
        f"on {recent_max_time:%d %b %Y at %H:%M}."
    )
    _optivolt_plotly_chart(
        build_home_history_chart(
            recent_series,
            xaxis_title="Time",
            peak_label="Recent-window peak",
            show_peak=True,
        ),
        use_container_width=True,
    )

    st.markdown(
        f"**Full held-out test view — {heldout_start_label} to {heldout_end_label}**"
    )
    st.caption(
        f"Full untouched held-out test period used for evaluation "
        f"({model_meta['test_observations']:,} observations). "
        f"This period includes the recent operational window shown above."
    )
    heldout_peak_value = float(full_test_series.max())
    heldout_peak_time = full_test_series.idxmax()

    st.caption(
        f"Held-out-period peak: {heldout_peak_value:.2f} kW "
        f"on {heldout_peak_time:%d %b %Y at %H:%M}."
    )
    _optivolt_plotly_chart(
        build_home_history_chart(
            full_test_series,
            xaxis_title="Date",
            peak_label="Held-out-period peak",
            show_peak=True,
        ),
        use_container_width=True,
    )


    st.subheader("Operational Outlook")
    st.markdown(
        f"""
        <div class="ov-info-grid-4">
            <div class="ov-info-card">
                <div class="ov-info-label">Model Health</div>
                <div class="ov-info-value ov-info-value-small">{model_health_status.replace("Performing above baseline", "Above baseline")}</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Model Version</div>
                <div class="ov-info-value ov-info-value-small">{model_version}</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Forecast Horizon</div>
                <div class="ov-info-value">{int(st.session_state["forecast_horizon"])} hours</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Forecast Breach Duration</div>
                <div class="ov-info-value">{_format_duration(total_breach_minutes) if total_breach_minutes else "None"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if len(breaches):
        st.warning(operational_recommendation)
    elif len(warnings):
        st.info(operational_recommendation)
    else:
        st.success(operational_recommendation)

elif nav_selection == "🏢 Depots Setup":
    st.title("🏢 Depot Configuration Registry")
    st.caption(
        "Configure planning thresholds for the currently selected depot. "
        "The organisation shown below is the company that owns and operates this application."
    )

    with st.container(border=True):
        st.text_input(
            "Selected Depot",
            value=st.session_state["current_depot"],
            disabled=True,
        )

        st.text_input(
            "Organisation",
            value=st.session_state["org_name"],
            disabled=True,
        )

        st.session_state["grid_limit"] = st.number_input(
            "Planning Capacity Threshold (kW)",
            value=float(st.session_state["grid_limit"]),
            min_value=0.0,
            step=1.0,
        )

        st.session_state["warn_threshold"] = st.number_input(
            "Warning Threshold (kW)",
            value=float(st.session_state["warn_threshold"]),
            min_value=0.0,
            step=1.0,
        )

        if st.button("Save Depot Configuration"):
            active_depot = st.session_state["current_depot"]
            current_capacity = float(st.session_state["grid_limit"])
            current_warning = float(st.session_state["warn_threshold"])

            if active_depot in DEPOT_CAPACITY_KW:
                overrides = dict(st.session_state.get("depot_threshold_overrides", {}))
                overrides[active_depot] = {
                    "capacity_kw": current_capacity,
                    "warning_kw": current_warning,
                    "warning_percent": (
                        int(round(100.0 * current_warning / current_capacity))
                        if current_capacity > 0 else 80
                    ),
                }
                st.session_state["depot_threshold_overrides"] = overrides

            elif active_depot in st.session_state.get("uploaded_depots_registry", {}):
                registry = dict(st.session_state.get("uploaded_depots_registry", {}))
                if active_depot in registry:
                    registry[active_depot] = dict(registry[active_depot])
                    registry[active_depot]["capacity_kw"] = current_capacity
                    registry[active_depot]["warning_kw"] = current_warning
                    st.session_state["uploaded_depots_registry"] = registry

            execute_production_ml_pipeline.clear()
            st.toast(
                f"{active_depot} configuration updated successfully."
            )
            st.rerun()

elif nav_selection == "📊 Upload Depot Data":
    st.title("📊 Upload Depot Data")

    saved_registry = st.session_state.get("uploaded_depots_registry", {})

    if saved_registry:
        with st.container(border=True):
            st.markdown("### Uploaded Depot Records")
            st.caption(
                "Only user-uploaded depots can be removed. The nine built-in "
                "Optimise Prime depots are protected."
            )

            registry_rows = []
            for depot_name, meta in sorted(saved_registry.items()):
                registry_rows.append(
                    {
                        "Uploaded Depot": depot_name,
                        "Planning Capacity (kW)": float(meta["capacity_kw"]),
                        "Source File": meta.get("filename", "N/A"),
                    }
                )
            st.dataframe(
                pd.DataFrame(registry_rows),
                use_container_width=True,
                hide_index=True,
            )

            depot_to_remove = st.selectbox(
                "Choose an uploaded depot to remove",
                sorted(saved_registry.keys()),
                key="uploaded_depot_remove_selector",
            )

            confirm_remove = st.checkbox(
                f"I confirm that I want to remove {depot_to_remove}.",
                key="confirm_remove_uploaded_depot",
            )

            if st.button(
                f"Remove {depot_to_remove}",
                use_container_width=True,
                disabled=not confirm_remove,
            ):
                updated_registry = remove_persisted_uploaded_depot(
                    depot_to_remove
                )
                st.session_state["uploaded_depots_registry"] = updated_registry

                # If the removed depot is active, return to a protected built-in depot.
                if st.session_state["current_depot"] == depot_to_remove:
                    fallback_depot = "Bexleyheath"
                    st.session_state["current_depot"] = fallback_depot
                    st.session_state["uploaded_df"] = None
                    st.session_state["uploaded_depot_name"] = None
                    st.session_state["uploaded_capacity_kw"] = None
                    st.session_state["grid_limit"] = DEPOT_CAPACITY_KW[fallback_depot]
                    st.session_state["warn_threshold"] = round(
                        WARNING_ALPHA * DEPOT_CAPACITY_KW[fallback_depot],
                        1,
                    )

                if st.session_state.get("active_uploaded_pipeline_depot") == depot_to_remove:
                    st.session_state["active_uploaded_pipeline"] = None
                    st.session_state["active_uploaded_pipeline_depot"] = None

                # Clear transient results only when they belong to the removed depot.
                completion = st.session_state.get("retraining_completion_summary")
                if completion and completion.get("depot") == depot_to_remove:
                    st.session_state["retraining_completion_summary"] = None

                execute_production_ml_pipeline.clear()
                st.rerun()

    st.markdown("### Import depot telemetry data")
    st.caption(
        "Upload telemetry for a new depot or newer observations. "
        "Evaluate the currently fitted model on the unseen data first; retrain only "
        "when adaptation is required."
    )
    st.markdown(
        "**Supported formats:** CSV, XLSX  •  **Maximum file size:** 200 MB"
    )

    uploaded_depot_label = st.text_input(
        "Uploaded depot name",
        value="New Depot",
        help=(
            "Enter a unique name for a new uploaded depot. If you deliberately use "
            "the name of an existing uploaded depot, its stored telemetry will be updated."
        ),
    )

    uploaded_file = st.file_uploader(
        "Upload depot telemetry file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            upload_id = hashlib.sha256(file_bytes).hexdigest()

            if uploaded_file.name.lower().endswith(".csv"):
                raw_uploaded_df = pd.read_csv(uploaded_file)
            else:
                raw_uploaded_df = pd.read_excel(uploaded_file)

            standardised_upload, validation = validate_uploaded_telemetry(
                raw_uploaded_df
            )

            st.success("File validation passed.")

            st.markdown("#### Data Quality Status")
            quality_status = (
                "PASS"
                if validation["sampling_consistency_pct"] >= 95.0
                and validation["missing_timestamps"] == 0
                and validation["missing_demand_values"] == 0
                else "PASS WITH CLEANING"
            )
            st.markdown(
                f"""
                <div class="ov-quality-grid">
                    <div class="ov-quality-card"><div class="q-label">Rows Loaded</div><div class="q-value">{validation['raw_rows']:,}</div></div>
                    <div class="ov-quality-card"><div class="q-label">Missing Timestamps</div><div class="q-value">{validation['missing_timestamps']:,}</div></div>
                    <div class="ov-quality-card"><div class="q-label">Missing Demand</div><div class="q-value">{validation['missing_demand_values']:,}</div></div>
                    <div class="ov-quality-card"><div class="q-label">Duplicate Timestamps</div><div class="q-value">{validation['duplicate_timestamps']:,}</div></div>
                    <div class="ov-quality-card"><div class="q-label">15-min Consistency</div><div class="q-value">{validation['sampling_consistency_pct']:.1f}%</div></div>
                    <div class="ov-quality-card"><div class="q-label">Prepared Rows</div><div class="q-value">{validation['standardised_rows']:,}</div></div>
                    <div class="ov-quality-card"><div class="q-label">Date Range</div><div class="q-value">{validation['coverage_days']:.1f} days</div></div>
                    <div class="ov-quality-card"><div class="q-label">Validation Outcome</div><div class="q-value">{quality_status}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            validation_table = pd.DataFrame(
                [
                    ("Timestamp column detected", validation["timestamp_column"]),
                    ("Demand column detected", validation["demand_column"]),
                    ("Prepared sampling interval", validation["sampling_interval"]),
                    ("Sampling consistency", f"{validation['sampling_consistency_pct']:.1f}% at 15-minute intervals"),
                    ("Missing timestamps", validation["missing_timestamps"]),
                    ("Missing demand values", validation["missing_demand_values"]),
                    ("Duplicate timestamps", validation["duplicate_timestamps"]),
                    ("Start", validation["start"].strftime("%d %b %Y %H:%M")),
                    ("End", validation["end"].strftime("%d %b %Y %H:%M")),
                    (
                        "Negative demand values clipped",
                        validation["negative_values_clipped"],
                    ),
                    ("Minimum history check", f"Passed (≥ {int(st.session_state.get('min_upload_history_days', 28))} days)"),
                ],
                columns=["Validation Check", "Result"],
            )
            st.dataframe(
                validation_table,
                use_container_width=True,
                hide_index=True,
            )

            default_capacity = derive_uploaded_planning_threshold(
                standardised_upload
            )

            uploaded_capacity = st.number_input(
                "Planning Capacity Threshold for uploaded depot (kW)",
                min_value=0.1,
                value=float(round(default_capacity, 1)),
                step=1.0,
                help=(
                    "A default is derived from the development portion of the uploaded "
                    "telemetry. Replace it with a verified planning value if available."
                ),
            )

            evaluation_key = (
                f"{upload_id}:{float(uploaded_capacity):.3f}:"
                f"{st.session_state['current_depot']}:"
                f"{model_meta.get('training_date', '')}"
            )
            evaluation_done = (
                st.session_state.get("last_evaluated_upload_key")
                == evaluation_key
            )

            st.markdown("#### Step 1 — Evaluate the existing model")

            previously_trained_depot, previously_trained_meta = (
                find_uploaded_depot_by_upload_id(
                    upload_id,
                    st.session_state.get("uploaded_depots_registry", {}),
                )
            )
            upload_already_used_for_training = (
                st.session_state.get("trained_upload_id") == upload_id
                or previously_trained_depot is not None
            )

            if upload_already_used_for_training:
                st.warning(
                    "This uploaded dataset has already been used to train the active "
                    "model, so it can no longer be treated as unseen data. "
                    "Upload newer observations that were not used for fitting to assess "
                    "generalisation."
                )
                st.caption(
                    f"Previously trained file: "
                    f"**{(previously_trained_meta or {}).get('filename') or st.session_state.get('trained_upload_filename') or uploaded_file.name}**"
                )
            else:
                st.caption(
                    f"This tests the model currently active for "
                    f"**{st.session_state['current_depot']}** on observations that have "
                    "not been used to fit that model."
                )

            st.markdown(
                """
                <style>
                div[data-testid="stButton"] button[kind="primary"] {
                    background-color: #2563EB !important;
                    border-color: #2563EB !important;
                    color: #FFFFFF !important;
                }
                div[data-testid="stButton"] button[kind="primary"]:hover {
                    background-color: #1D4ED8 !important;
                    border-color: #1D4ED8 !important;
                    color: #FFFFFF !important;
                }
                div[data-testid="stButton"] button[kind="primary"]:focus {
                    box-shadow: 0 0 0 0.2rem rgba(37, 99, 235, 0.22) !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            evaluate_clicked = st.button(
                "Evaluate Existing Model on Uploaded Data",
                type="primary",
                use_container_width=True,
                disabled=upload_already_used_for_training,
                help=(
                    "This file has already been used for training. Upload newer unseen "
                    "observations to evaluate generalisation."
                    if upload_already_used_for_training
                    else "Evaluate the currently fitted model without retraining."
                ),
            )

            if evaluate_clicked:
                eval_progress = st.progress(
                    0,
                    text="0% — Starting unseen-data evaluation",
                )
                eval_status = st.status(
                    "🔎 Evaluating existing model",
                    state="running",
                    expanded=True,
                )

                try:
                    eval_progress.progress(
                        20,
                        text="20% — Standardising uploaded telemetry",
                    )
                    eval_status.write(
                        "**20%** — Standardising timestamps and 15-minute demand"
                    )

                    eval_progress.progress(
                        45,
                        text="45% — Building 96-step LSTM demand sequences",
                    )
                    eval_status.write(
                        "**45%** — Building 24-hour LSTM input sequences"
                    )

                    eval_progress.progress(
                        65,
                        text="65% — Running the existing LSTM without retraining",
                    )
                    eval_status.write(
                        "**65%** — Predicting with the currently fitted LSTM model"
                    )

                    _eval_model, _eval_bundle = _ensure_project_lstm_for_unseen_eval(pipeline)
                    evaluation = evaluate_existing_model_on_uploaded_data(
                        _eval_model,
                        _eval_bundle,
                        raw_uploaded_df,
                        float(uploaded_capacity),
                    )

                    eval_progress.progress(
                        90,
                        text="90% — Comparing LSTM with the previous-day baseline",
                    )
                    eval_status.write(
                        "**90%** — Calculating held-out-style error and breach metrics"
                    )

                    xgb_eval = evaluation["xgb_metrics"]
                    base_eval = evaluation["baseline_metrics"]

                    st.session_state["last_evaluated_upload_key"] = evaluation_key
                    st.session_state["uploaded_evaluation_summary"] = {
                        "depot_name": uploaded_depot_label.strip() or "Uploaded Depot",
                        "source_model_depot": st.session_state["current_depot"],
                        "start": evaluation["start"],
                        "end": evaluation["end"],
                        "rows": evaluation["rows"],
                        "xgb_metrics": xgb_eval,
                        "baseline_metrics": base_eval,
                        "timestamps": evaluation["timestamps"],
                        "actual": evaluation["actual"],
                        "predictions": evaluation["predictions"],
                        "baseline": evaluation["baseline"],
                    }

                    eval_progress.progress(
                        100,
                        text="100% — Existing-model evaluation completed",
                    )
                    eval_status.update(
                        label="✅ Existing-model evaluation completed",
                        state="complete",
                        expanded=True,
                    )

                except Exception as eval_exc:
                    eval_progress.empty()
                    eval_status.update(
                        label="❌ Existing-model evaluation failed",
                        state="error",
                        expanded=True,
                    )
                    eval_status.write(str(eval_exc))

            # Persist evaluation results below the button on subsequent reruns.
            evaluation_done = (
                st.session_state.get("last_evaluated_upload_key")
                == evaluation_key
                and not upload_already_used_for_training
            )
            eval_summary = (
                st.session_state.get("uploaded_evaluation_summary")
                if evaluation_done
                else None
            )

            if upload_already_used_for_training:
                st.session_state["last_evaluated_upload_key"] = None
                st.session_state["uploaded_evaluation_summary"] = None

            if eval_summary:
                xgb_eval = eval_summary["xgb_metrics"]
                base_eval = eval_summary["baseline_metrics"]

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Unseen-data MAE", f"{xgb_eval['mae']:.2f} kW")
                e2.metric("Unseen-data RMSE", f"{xgb_eval['rmse']:.2f} kW")
                e3.metric("R²", f"{xgb_eval['r2']:.3f}")
                e4.metric("Evaluated rows", f"{eval_summary['rows']:,}")

                comparison = pd.DataFrame(
                    {
                        "Metric": [
                            "MAE",
                            "RMSE",
                            "Breach Recall",
                            "False Alarm Ratio",
                        ],
                        "Existing LSTM": [
                            f"{xgb_eval['mae']:.2f} kW",
                            f"{xgb_eval['rmse']:.2f} kW",
                            (
                                f"{xgb_eval['recall']:.2f}%"
                                if xgb_eval["recall"] is not None
                                else "N/A"
                            ),
                            (
                                f"{xgb_eval['far']:.2f}%"
                                if xgb_eval["far"] is not None
                                else "N/A"
                            ),
                        ],
                        "Previous-Day Baseline": [
                            f"{base_eval['mae']:.2f} kW",
                            f"{base_eval['rmse']:.2f} kW",
                            (
                                f"{base_eval['recall']:.2f}%"
                                if base_eval["recall"] is not None
                                else "N/A"
                            ),
                            (
                                f"{base_eval['far']:.2f}%"
                                if base_eval["far"] is not None
                                else "N/A"
                            ),
                        ],
                    }
                )
                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"Evaluation period: "
                    f"{eval_summary['start']:%d %b %Y %H:%M} to "
                    f"{eval_summary['end']:%d %b %Y %H:%M}. "
                    "No fitting was performed on these uploaded observations during this evaluation."
                )

                # ---------------------------------------------------------
                # Visual generalisation checks on the genuinely unseen data
                # ---------------------------------------------------------
                eval_timestamps = pd.to_datetime(eval_summary["timestamps"])
                eval_actual = np.asarray(eval_summary["actual"], dtype=float)
                eval_pred = np.asarray(eval_summary["predictions"], dtype=float)
                eval_baseline = np.asarray(eval_summary["baseline"], dtype=float)
                eval_residuals = eval_actual - eval_pred

                st.markdown("##### Unseen-data visual evaluation")

                # Plot 1: actual demand vs existing LSTM vs previous-day baseline.
                fig_eval_ts = go.Figure()
                fig_eval_ts.add_trace(
                    go.Scatter(
                        x=eval_timestamps,
                        y=eval_actual,
                        mode="lines",
                        name="Actual Demand",
                        line=dict(width=1.5),
                    )
                )
                fig_eval_ts.add_trace(
                    go.Scatter(
                        x=eval_timestamps,
                        y=eval_pred,
                        mode="lines",
                        name="Existing LSTM",
                        line=dict(width=1.3),
                    )
                )
                fig_eval_ts.add_trace(
                    go.Scatter(
                        x=eval_timestamps,
                        y=eval_baseline,
                        mode="lines",
                        name="Previous-Day Baseline",
                        line=dict(width=1.0, dash="dot"),
                    )
                )
                fig_eval_ts.update_layout(
                    title="Actual vs Existing LSTM vs Previous-Day Baseline",
                    xaxis_title="Timestamp",
                    yaxis_title="Demand (kW)",
                    hovermode="x unified",
                    height=440,
                    margin=dict(l=40, r=20, t=65, b=40),
                    legend=dict(orientation="h", y=1.12, x=0),
                )
                _optivolt_plotly_chart(fig_eval_ts, use_container_width=True)

                # Plot 2: actual vs predicted scatter, with y=x ideal line.
                # Use identical zero-based axes so the visual comparison is not distorted.
                scatter_data_max = float(
                    max(np.max(eval_actual), np.max(eval_pred), 1.0)
                )
                scatter_max = scatter_data_max * 1.05

                fig_eval_scatter = go.Figure()
                fig_eval_scatter.add_trace(
                    go.Scatter(
                        x=eval_actual,
                        y=eval_pred,
                        mode="markers",
                        name="Unseen Observations",
                        marker=dict(size=5, opacity=0.30),
                    )
                )
                fig_eval_scatter.add_trace(
                    go.Scatter(
                        x=[0, scatter_max],
                        y=[0, scatter_max],
                        mode="lines",
                        name="Ideal: Predicted = Actual",
                        line=dict(dash="dash"),
                    )
                )
                fig_eval_scatter.update_layout(
                    title=dict(
                        text="Actual vs Predicted Demand on Unseen Data",
                        x=0.0,
                        xanchor="left",
                    ),
                    xaxis_title="Actual Demand (kW)",
                    yaxis_title="Predicted Demand (kW)",
                    height=460,
                    margin=dict(l=65, r=30, t=95, b=65),
                    legend=dict(
                        orientation="h",
                        y=1.08,
                        x=0,
                        yanchor="bottom",
                        xanchor="left",
                    ),
                )
                fig_eval_scatter.update_xaxes(
                    range=[0, scatter_max],
                    zeroline=False,
                )
                fig_eval_scatter.update_yaxes(
                    range=[0, scatter_max],
                    zeroline=False,
                )
                _optivolt_plotly_chart(fig_eval_scatter, use_container_width=True)

                # Plot 3: residual distribution to show error bias/spread.
                # Reference lines are placed in the legend to avoid overlapping labels.
                mean_residual = float(np.mean(eval_residuals))
                median_residual = float(np.median(eval_residuals))
                residual_ymax = max(
                    1,
                    int(np.histogram(eval_residuals, bins=60)[0].max() * 1.08),
                )

                fig_eval_resid = go.Figure()
                fig_eval_resid.add_trace(
                    go.Histogram(
                        x=eval_residuals,
                        nbinsx=60,
                        name="Residuals",
                        opacity=0.80,
                        showlegend=False,
                    )
                )

                for x_value, label, dash_style in [
                    (0.0, "Zero error", "dash"),
                    (mean_residual, f"Mean: {mean_residual:.2f} kW", "dot"),
                    (median_residual, f"Median: {median_residual:.2f} kW", "dashdot"),
                ]:
                    fig_eval_resid.add_trace(
                        go.Scatter(
                            x=[x_value, x_value],
                            y=[0, residual_ymax],
                            mode="lines",
                            name=label,
                            line=dict(dash=dash_style, width=2),
                            hoverinfo="skip",
                        )
                    )

                fig_eval_resid.update_layout(
                    title=dict(
                        text="Unseen-Data Residual Distribution (Actual − Predicted)",
                        x=0.0,
                        xanchor="left",
                    ),
                    xaxis_title="Residual (kW)",
                    yaxis_title="Frequency",
                    height=440,
                    margin=dict(l=65, r=30, t=120, b=65),
                    bargap=0.03,
                    legend=dict(
                        orientation="h",
                        y=1.04,
                        x=0,
                        yanchor="bottom",
                        xanchor="left",
                    ),
                )
                _optivolt_plotly_chart(fig_eval_resid, use_container_width=True)

                st.caption(
                    "Residual = Actual − Predicted. Values near zero indicate small "
                    "prediction errors; a distribution centred near zero indicates low systematic bias."
                )

                if xgb_eval["mae"] < base_eval["mae"]:
                    st.success(
                        "The existing LSTM model outperforms the previous-day "
                        "baseline on this uploaded dataset. Retraining is optional."
                    )
                else:
                    st.warning(
                        "The existing LSTM model does not outperform the "
                        "previous-day baseline on MAE. Review the results before retraining."
                    )

            st.markdown("#### Step 2 — Retrain only if adaptation is needed")
            st.caption(
                "Retraining fits a new LSTM model to this uploaded dataset "
                "using a chronological 70% training / 15% validation / 15% held-out "
                "test split. The final 15% remains untouched during fitting."
            )

            retrain_clicked = st.button(
                "Retrain LSTM Using Uploaded Data",
                use_container_width=True,
                disabled=(not evaluation_done) or upload_already_used_for_training,
                help=(
                    "This exact file has already been used for model fitting."
                    if upload_already_used_for_training
                    else (
                        "Evaluate the existing model first. Retraining is enabled after "
                        "the unseen-data evaluation has completed."
                    )
                ),
            )

            if upload_already_used_for_training:
                st.caption(
                    "This file has already been used for retraining. Upload newer data "
                    "to perform another genuine unseen-data evaluation."
                )
            elif not evaluation_done:
                st.caption(
                    "Complete Step 1 before retraining so generalisation is measured "
                    "before the model is adapted to the uploaded data."
                )

            if retrain_clicked:
                clean_name = uploaded_depot_label.strip() or "Uploaded Depot"

                if clean_name in REAL_DEPOTS:
                    st.error(
                        f"{clean_name} is a protected built-in Optimise Prime depot. "
                        "Choose a different name for uploaded telemetry."
                    )
                    st.stop()

                progress_bar = st.progress(
                    0,
                    text="Preparing retraining workflow...",
                )
                retrain_status = st.status(
                    "⏳ Retraining in progress",
                    expanded=True,
                    state="running",
                )

                last_message = {"text": None}

                def update_retraining_progress(percent: int, message: str):
                    progress_bar.progress(
                        min(max(percent, 0), 100),
                        text=f"{percent}% — {message}",
                    )
                    if message != last_message["text"]:
                        retrain_status.write(
                            f"**{percent}%** — {message}"
                        )
                        last_message["text"] = message

                try:
                    retrain_status.write(
                        "Model configuration: **LSTM, 96-step lookback with early stopping**"
                    )
                    retrain_status.write(
                        "Uploaded-data split: **70% training / 15% validation / "
                        "15% untouched held-out test**"
                    )

                    # Preserve the existing model's unseen-data predictions before
                    # retraining. These will later be aligned to the NEW model's exact
                    # held-out test timestamps for an apples-to-apples comparison.
                    _eval_model, _eval_bundle = _ensure_project_lstm_for_unseen_eval(pipeline)
                    pre_retrain_eval = evaluate_existing_model_on_uploaded_data(
                        _eval_model,
                        _eval_bundle,
                        raw_uploaded_df,
                        float(uploaded_capacity),
                    )

                    pre_existing_series = pd.Series(
                        np.asarray(pre_retrain_eval["predictions"], dtype=float),
                        index=pd.to_datetime(pre_retrain_eval["timestamps"]),
                    )

                    # This button means retrain, so clear the cached fitted pipeline
                    # to guarantee a new fit rather than returning a cached result.
                    execute_production_ml_pipeline.clear()

                    retrained_pipeline = execute_production_ml_pipeline_with_progress(
                        clean_name,
                        float(uploaded_capacity),
                        int(st.session_state["forecast_horizon"]),
                        raw_uploaded_df,
                        update_retraining_progress,
                    )

                    st.session_state["uploaded_df"] = raw_uploaded_df
                    st.session_state["uploaded_depot_name"] = clean_name
                    st.session_state["uploaded_capacity_kw"] = float(uploaded_capacity)
                    st.session_state["upload_meta"] = {
                        "filename": uploaded_file.name,
                        "rows": len(raw_uploaded_df),
                        "columns": list(raw_uploaded_df.columns),
                        "validation": validation,
                        "split_strategy": (
                            "70% train / 15% validation / 15% held-out test"
                        ),
                        "pre_retrain_evaluation": eval_summary,
                    }
                    st.session_state["current_depot"] = clean_name
                    st.session_state["grid_limit"] = float(uploaded_capacity)
                    st.session_state["warn_threshold"] = round(
                        WARNING_ALPHA * float(uploaded_capacity),
                        1,
                    )
                    st.session_state["last_activated_upload_id"] = upload_id
                    st.session_state["trained_upload_id"] = upload_id
                    st.session_state["trained_upload_filename"] = uploaded_file.name
                    st.session_state["active_uploaded_pipeline"] = retrained_pipeline
                    st.session_state["active_uploaded_pipeline_depot"] = clean_name

                    updated_registry = save_uploaded_depot_state(
                        clean_name,
                        float(uploaded_capacity),
                        raw_uploaded_df,
                        uploaded_file.name,
                        upload_id,
                    )
                    st.session_state["uploaded_depots_registry"] = updated_registry

                    st.session_state["upload_activation_notice"] = {
                        "depot": clean_name,
                        "filename": uploaded_file.name,
                        "completed": True,
                    }

                    completed_meta = retrained_pipeline["metadata"]

                    progress_bar.progress(
                        94,
                        text="94% — Finalising evaluation outputs",
                    )
                    retrain_status.update(
                        label=f"⏳ Finalising results for {clean_name}",
                        state="running",
                        expanded=True,
                    )

                    retrain_status.write(
                        f"Held-out MAE: "
                        f"**{completed_meta['metrics_xgb']['mae']} kW**"
                    )
                    retrain_status.write(
                        f"Held-out RMSE: "
                        f"**{completed_meta['metrics_xgb']['rmse']} kW**"
                    )
                    retrain_status.write(
                        f"Breach Recall: "
                        f"**{completed_meta['metrics_xgb']['recall']}**"
                    )
                    retrain_status.write(
                        f"False Alarm Ratio: "
                        f"**{completed_meta['metrics_xgb']['far']}**"
                    )

                    st.caption(
                        f"Training: {completed_meta.get('train_period', 'N/A')}  |  "
                        f"Validation: {completed_meta.get('validation_period', 'N/A')}  |  "
                        f"Held-out test: {completed_meta.get('test_period', 'N/A')}"
                    )

                    retrained_y_test = retrained_pipeline["y_test"]
                    retrained_test_index = pd.to_datetime(retrained_y_test.index)
                    retrained_actual = np.asarray(retrained_y_test, dtype=float)
                    retrained_pred = np.asarray(
                        retrained_pipeline["xgb_preds"],
                        dtype=float,
                    )

                    existing_on_same_test = (
                        pre_existing_series.reindex(retrained_test_index).to_numpy(dtype=float)
                    )

                    valid_compare = np.isfinite(existing_on_same_test)
                    compare_index = retrained_test_index[valid_compare]
                    compare_actual = retrained_actual[valid_compare]
                    compare_existing = existing_on_same_test[valid_compare]
                    compare_retrained = retrained_pred[valid_compare]

                    progress_bar.progress(
                        97,
                        text="97% — Comparing pre-retraining and retrained performance",
                    )
                    retrain_status.write(
                        "**97%** — Preparing the before-versus-after held-out comparison"
                    )

                    existing_same_metrics = regression_metrics_for_comparison(
                        compare_actual,
                        compare_existing,
                    )
                    retrained_same_metrics = regression_metrics_for_comparison(
                        compare_actual,
                        compare_retrained,
                    )

                    st.session_state["retraining_completion_summary"] = {
                        "upload_id": upload_id,
                        "depot": clean_name,
                        "mae": completed_meta["metrics_xgb"]["mae"],
                        "rmse": completed_meta["metrics_xgb"]["rmse"],
                        "recall": completed_meta["metrics_xgb"]["recall"],
                        "far": completed_meta["metrics_xgb"]["far"],
                        "train_period": completed_meta.get("train_period", "N/A"),
                        "validation_period": completed_meta.get("validation_period", "N/A"),
                        "test_period": completed_meta.get("test_period", "N/A"),
                        "compare_timestamps": compare_index.to_numpy(),
                        "compare_actual": compare_actual,
                        "compare_existing": compare_existing,
                        "compare_retrained": compare_retrained,
                        "existing_same_metrics": existing_same_metrics,
                        "retrained_same_metrics": retrained_same_metrics,
                    }

                    progress_bar.progress(
                        100,
                        text="100% — Retraining, evaluation and result preparation completed",
                    )
                    retrain_status.update(
                        label=f"✅ Retraining completed for {clean_name}",
                        state="complete",
                        expanded=True,
                    )

                    # Rerun so the sidebar is rebuilt immediately with the newly
                    # activated uploaded depot included in the selector. The completed
                    # in-session pipeline is reused, so this rerun does not fit again.
                    st.rerun()

                except Exception as retrain_exc:
                    progress_bar.empty()
                    retrain_status.update(
                        label="❌ Retraining failed",
                        state="error",
                        expanded=True,
                    )
                    retrain_status.write(str(retrain_exc))

            completion = st.session_state.get("retraining_completion_summary")
            if (
                completion
                and completion.get("upload_id") == upload_id
                and completion.get("depot") == st.session_state.get("uploaded_depot_name")
            ):
                st.success(
                    f"✅ Retraining completed for **{completion['depot']}**. "
                    "The new depot is now active and has been added to the left-hand depot selector."
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Held-out MAE", f"{completion['mae']} kW")
                c2.metric("Held-out RMSE", f"{completion['rmse']} kW")
                c3.metric("Breach Recall", f"{completion['recall']}")
                c4.metric("False Alarm Ratio", f"{completion['far']}")

                st.caption(
                    f"Training: {completion['train_period']}  |  "
                    f"Validation: {completion['validation_period']}  |  "
                    f"Held-out test: {completion['test_period']}"
                )

                st.info(
                    "This uploaded file has now contributed to model fitting and is no "
                    "longer valid as unseen data. Upload later/newer telemetry for the "
                    "next genuine generalisation test."
                )

                if (
                    completion.get("compare_timestamps") is not None
                    and completion.get("existing_same_metrics") is not None
                    and completion.get("retrained_same_metrics") is not None
                ):
                    st.markdown("#### Before vs After Retraining")
                    st.caption(
                        "The plots below compare the existing model and the retrained "
                        "model on the exact same untouched held-out test rows. This makes "
                        "the improvement comparison directly comparable."
                    )

                    comp_ts = pd.to_datetime(completion["compare_timestamps"])
                    comp_actual = np.asarray(completion["compare_actual"], dtype=float)
                    comp_existing = np.asarray(completion["compare_existing"], dtype=float)
                    comp_retrained = np.asarray(completion["compare_retrained"], dtype=float)

                    before_m = completion["existing_same_metrics"]
                    after_m = completion["retrained_same_metrics"]

                    # Plot 1: exact same held-out time series before vs after retraining.
                    fig_retrain_ts = go.Figure()
                    fig_retrain_ts.add_trace(
                        go.Scatter(
                            x=comp_ts,
                            y=comp_actual,
                            mode="lines",
                            name="Actual Demand",
                            line=dict(width=1.6),
                        )
                    )
                    fig_retrain_ts.add_trace(
                        go.Scatter(
                            x=comp_ts,
                            y=comp_existing,
                            mode="lines",
                            name="Existing Model Before Retraining",
                            line=dict(width=1.1, dash="dot"),
                        )
                    )
                    fig_retrain_ts.add_trace(
                        go.Scatter(
                            x=comp_ts,
                            y=comp_retrained,
                            mode="lines",
                            name="Retrained LSTM",
                            line=dict(width=1.4),
                        )
                    )
                    fig_retrain_ts.update_layout(
                        title=dict(
                            text="Held-Out Demand: Before vs After Retraining",
                            x=0.0,
                            xanchor="left",
                            y=0.98,
                            yanchor="top",
                        ),
                        xaxis_title="Timestamp",
                        yaxis_title="Demand (kW)",
                        hovermode="x unified",
                        height=470,
                        margin=dict(l=60, r=30, t=125, b=60),
                        legend=dict(
                            orientation="h",
                            y=1.04,
                            x=0,
                            yanchor="bottom",
                            xanchor="left",
                        ),
                    )
                    _optivolt_plotly_chart(fig_retrain_ts, use_container_width=True)

                    # Plot 2: MAE and RMSE before vs after on identical held-out rows.
                    metric_compare = pd.DataFrame(
                        {
                            "Metric": ["MAE", "RMSE"],
                            "Before Retraining": [
                                before_m["mae"],
                                before_m["rmse"],
                            ],
                            "After Retraining": [
                                after_m["mae"],
                                after_m["rmse"],
                            ],
                        }
                    )
                    fig_metric_compare = go.Figure()
                    fig_metric_compare.add_trace(
                        go.Bar(
                            x=metric_compare["Metric"],
                            y=metric_compare["Before Retraining"],
                            name="Before Retraining",
                            text=[
                                f"{v:.2f} kW"
                                for v in metric_compare["Before Retraining"]
                            ],
                            textposition="outside",
                        )
                    )
                    fig_metric_compare.add_trace(
                        go.Bar(
                            x=metric_compare["Metric"],
                            y=metric_compare["After Retraining"],
                            name="After Retraining",
                            text=[
                                f"{v:.2f} kW"
                                for v in metric_compare["After Retraining"]
                            ],
                            textposition="outside",
                        )
                    )
                    fig_metric_compare.update_layout(
                        title=dict(
                            text="Error Metrics Before vs After Retraining",
                            x=0.0,
                            xanchor="left",
                            y=0.98,
                            yanchor="top",
                        ),
                        xaxis_title="Metric",
                        yaxis_title="Error (kW) — lower is better",
                        barmode="group",
                        height=420,
                        margin=dict(l=60, r=30, t=120, b=60),
                        legend=dict(
                            orientation="h",
                            y=1.04,
                            x=0,
                            yanchor="bottom",
                            xanchor="left",
                        ),
                    )
                    _optivolt_plotly_chart(
                        fig_metric_compare,
                        use_container_width=True,
                    )

                    # Plot 3: residual distributions on identical held-out rows.
                    before_resid = comp_actual - comp_existing
                    after_resid = comp_actual - comp_retrained

                    resid_min = float(min(np.min(before_resid), np.min(after_resid)))
                    resid_max = float(max(np.max(before_resid), np.max(after_resid)))
                    bin_size = max((resid_max - resid_min) / 45.0, 0.25)

                    before_mean = float(np.mean(before_resid))
                    after_mean = float(np.mean(after_resid))
                    before_median = float(np.median(before_resid))
                    after_median = float(np.median(after_resid))

                    fig_resid_compare = go.Figure()
                    fig_resid_compare.add_trace(
                        go.Histogram(
                            x=before_resid,
                            xbins=dict(
                                start=resid_min,
                                end=resid_max,
                                size=bin_size,
                            ),
                            name="Before Retraining",
                            opacity=0.50,
                        )
                    )
                    fig_resid_compare.add_trace(
                        go.Histogram(
                            x=after_resid,
                            xbins=dict(
                                start=resid_min,
                                end=resid_max,
                                size=bin_size,
                            ),
                            name="After Retraining",
                            opacity=0.50,
                        )
                    )

                    # Zero-error reference as a trace so it appears cleanly in the legend.
                    hist_max = max(
                        np.histogram(before_resid, bins=45, range=(resid_min, resid_max))[0].max(),
                        np.histogram(after_resid, bins=45, range=(resid_min, resid_max))[0].max(),
                    )
                    fig_resid_compare.add_trace(
                        go.Scatter(
                            x=[0, 0],
                            y=[0, float(hist_max) * 1.08],
                            mode="lines",
                            name="Zero Error",
                            line=dict(dash="dash", width=2),
                            hoverinfo="skip",
                        )
                    )

                    fig_resid_compare.update_layout(
                        title=dict(
                            text="Held-Out Residual Distribution Before vs After Retraining",
                            x=0.0,
                            xanchor="left",
                            y=0.98,
                            yanchor="top",
                        ),
                        xaxis_title="Residual (Actual − Predicted) kW",
                        yaxis_title="Frequency",
                        barmode="overlay",
                        height=440,
                        margin=dict(l=60, r=30, t=125, b=60),
                        legend=dict(
                            orientation="h",
                            y=1.04,
                            x=0,
                            yanchor="bottom",
                            xanchor="left",
                        ),
                    )
                    _optivolt_plotly_chart(
                        fig_resid_compare,
                        use_container_width=True,
                    )
                    st.caption(
                        f"Residual = Actual − Predicted. "
                        f"Before retraining: mean {before_mean:.2f} kW, median {before_median:.2f} kW. "
                        f"After retraining: mean {after_mean:.2f} kW, median {after_median:.2f} kW. "
                        "A tighter distribution centred closer to zero indicates better calibration."
                    )

                    mae_change = before_m["mae"] - after_m["mae"]
                    rmse_change = before_m["rmse"] - after_m["rmse"]

                    if mae_change > 0 and rmse_change > 0:
                        st.success(
                            f"Retraining improved performance on the identical held-out "
                            f"rows: MAE reduced by **{mae_change:.2f} kW** and RMSE "
                            f"reduced by **{rmse_change:.2f} kW**."
                        )
                    else:
                        st.warning(
                            "Retraining did not improve both MAE and RMSE on the identical "
                            "held-out rows. Review the plots and metrics before adopting "
                            "the retrained model operationally."
                        )

        except Exception as exc:
            st.error(f"Upload validation failed: {exc}")

    elif st.session_state.get("uploaded_depots_registry"):
        st.info(
            f"{len(st.session_state['uploaded_depots_registry'])} uploaded depot(s) are "
            "available. Select any of them from the left-hand depot selector."
        )

elif nav_selection == "📈 Demand Forecast":
    st.markdown("### Forecast Comparison Horizon")
    st.caption(
        "Switch the operational view without retraining the model. "
        "The LSTM recursively extends the forecast to the selected horizon."
    )
    comparison_horizons = [2, 4, 6, 12, 24]
    active_horizon = int(st.session_state.get("forecast_horizon", 24))
    if active_horizon not in comparison_horizons:
        active_horizon = 24

    forecast_compare_horizon = st.radio(
        "Compare horizon",
        comparison_horizons,
        index=comparison_horizons.index(active_horizon),
        format_func=lambda h: f"{h}h",
        horizontal=True,
        label_visibility="collapsed",
        key="demand_forecast_horizon_compare",
    )

    if int(forecast_compare_horizon) != active_horizon:
        st.session_state["forecast_horizon"] = int(forecast_compare_horizon)
        st.session_state["forecast_horizon_selector"] = int(forecast_compare_horizon)
        execute_production_ml_pipeline.clear()
        st.rerun()

    st.markdown(
        f"""
        <h1 style="
            font-size: 2.65rem;
            line-height: 1.08;
            margin-bottom: 0.35rem;
            white-space: nowrap;
            color: #0F172A;
        ">
            📈 Historical Demand & {int(st.session_state["forecast_horizon"])}-Hour Forecast
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Historical observed demand is shown up to the forecast start. "
        f"The dashed green line is the {int(st.session_state['forecast_horizon'])}-hour LSTM forecast based on the "
        "latest available project data. The shaded area is an empirical 80% forecast range derived from held-out residuals."
    )
    
    # Compact operational risk summary for the active forecast horizon.
    available_headroom = float(st.session_state["grid_limit"]) - predicted_peak
    first_warning_text = (
        future_times_index[warnings[0]].strftime("%d %b %H:%M")
        if len(warnings) > 0 else "None"
    )
    first_breach_text = (
        first_breach_start_time.strftime("%d %b %H:%M")
        if first_breach_start_time is not None else "None"
    )
    breach_duration_text = (
        f"{total_breach_minutes // 60}h {total_breach_minutes % 60:02d}m"
        if total_breach_minutes > 0 else "0 min"
    )
    headroom_class = "ov-safe" if available_headroom > 0 else "ov-danger"
    warning_class = "ov-warning" if len(warnings) > 0 else "ov-safe"
    breach_class = "ov-danger" if len(breaches) > 0 else "ov-safe"

    st.markdown(
        f"""
        <div class="ov-status-strip">
            <div class="ov-status-card ov-model">
                <div class="label">Predicted Peak</div>
                <div class="value">{predicted_peak:.1f} kW</div>
            </div>
            <div class="ov-status-card {headroom_class}">
                <div class="label">Available Headroom</div>
                <div class="value">{available_headroom:.1f} kW</div>
            </div>
            <div class="ov-status-card {warning_class}">
                <div class="label">First Warning</div>
                <div class="value">{first_warning_text}</div>
            </div>
            <div class="ov-status-card {breach_class}">
                <div class="label">First Breach</div>
                <div class="value">{first_breach_text}</div>
            </div>
            <div class="ov-status-card {breach_class}">
                <div class="label">Breach Duration</div>
                <div class="value">{breach_duration_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    hist_x = history_series.index[-192:]
    fig.add_trace(go.Scatter(x=hist_x, y=history_series.values[-192:], name="Actual Demand", line=dict(color="#0F172A")))
    future_x = future_times
    fig.add_trace(go.Scatter(
        x=future_x,
        y=forecast_upper,
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name="Upper forecast range",
    ))
    fig.add_trace(go.Scatter(
        x=future_x,
        y=forecast_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(16,185,129,0.16)",
        name="Empirical 80% Forecast Range",
        hovertemplate="%{x|%d %b %H:%M}<br>Lower range: %{y:.2f} kW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=future_x,
        y=horizon_forecast,
        name="LSTM Forecast",
        line=dict(color="#10B981", width=2.4, dash="dash"),
        hovertemplate="%{x|%d %b %H:%M}<br>Forecast: %{y:.2f} kW<extra></extra>",
    ))
    
    fig.add_shape(type="line", x0=hist_x[0], y0=st.session_state["grid_limit"], x1=future_x[-1], y1=st.session_state["grid_limit"], line=dict(color="#EF4444", width=2))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#EF4444', width=2), name='Planning Capacity Threshold'))
    
    fig.add_shape(type="line", x0=hist_x[0], y0=st.session_state["warn_threshold"], x1=future_x[-1], y1=st.session_state["warn_threshold"], line=dict(color="#F59E0B", dash="dot"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#F59E0B', dash='dot'), name='Warning Threshold'))
    
    fig.add_shape(type="line", x0=hist_x[-1], y0=0, x1=hist_x[-1], y1=predicted_peak * 1.2, line=dict(color="#64748B", dash="dash"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#64748B', dash='dash'), name='Forecast Start'))
    
    fig.update_layout(template="plotly_white", height=440, legend=dict(orientation="h", y=1.15))
    _optivolt_plotly_chart(fig, use_container_width=True)
    with st.expander("About the empirical 80% forecast range"):
        st.write(
            f"At the predicted peak, the empirical range is "
            f"{forecast_peak_lower:.1f}–{forecast_peak_upper:.1f} kW. "
            "It is calculated by applying the 10th and 90th percentiles of held-out "
            "forecast residuals to the recursive LSTM forecast. It is an uncertainty "
            "indicator for operational interpretation and is not a formal probabilistic "
            "confidence interval."
        )
        st.caption(
            "Held-out residuals come from the same historical project dataset used for model evaluation."
        )

elif nav_selection == "🚗 Scenario Analysis":
    st.title("🚗 Scenario Analysis")
    st.caption(
        "Scenario values are hypothetical adjustments applied to the genuine "
        "LSTM base forecast."
    )

    growth = st.slider(
        "Projected Fleet / Demand Growth (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )

    scenario_y = horizon_forecast * (1.0 + (growth / 100.0))
    timeline = future_times

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])
    scenario_peak = float(np.max(scenario_y))

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario Peak", f"{scenario_peak:.1f} kW")
    c2.metric("Warning Threshold", f"{warning_threshold:.1f} kW")
    c3.metric("Planning Capacity Threshold", f"{planning_threshold:.1f} kW")

    fig_sc = go.Figure()

    fig_sc.add_trace(
        go.Scatter(
            x=timeline,
            y=horizon_forecast,
            name="Base LSTM Forecast",
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dot"),
            hovertemplate="%{x|%H:%M}<br>Base forecast: %{y:.2f} kW<extra></extra>",
        )
    )

    fig_sc.add_trace(
        go.Scatter(
            x=timeline,
            y=scenario_y,
            name="Scenario-Adjusted Forecast",
            mode="lines",
            line=dict(color="#2563EB", width=2.5),
            hovertemplate="%{x|%H:%M}<br>Scenario forecast: %{y:.2f} kW<extra></extra>",
        )
    )

    fig_sc.add_hline(
        y=warning_threshold,
        line=dict(color="#F59E0B", width=1.8, dash="dash"),
        annotation_text="Warning Threshold",
        annotation_position="top left",
    )

    fig_sc.add_hline(
        y=planning_threshold,
        line=dict(color="#DC2626", width=1.8, dash="dash"),
        annotation_text="Planning Capacity Threshold",
        annotation_position="top left",
    )

    fig_sc.update_layout(
        template="plotly_white",
        height=440,
        xaxis_title="Forecast Time",
        yaxis_title="Demand (kW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
        hovermode="x unified",
    )

    fig_sc.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_sc.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
    )

    _optivolt_plotly_chart(fig_sc, use_container_width=True)

    # Baseline vs hypothetical scenario comparison.
    base_peak = float(np.max(horizon_forecast))
    base_headroom = planning_threshold - base_peak
    scenario_headroom = planning_threshold - scenario_peak

    base_warning_intervals = int(np.sum(np.asarray(horizon_forecast) >= warning_threshold))
    scenario_warning_intervals = int(np.sum(np.asarray(scenario_y) >= warning_threshold))
    base_breach_intervals = int(np.sum(np.asarray(horizon_forecast) >= planning_threshold))
    scenario_breach_intervals = int(np.sum(np.asarray(scenario_y) >= planning_threshold))
    base_breach_minutes = base_breach_intervals * SAMPLING_MINUTES
    scenario_breach_minutes = scenario_breach_intervals * SAMPLING_MINUTES

    scenario_comparison = pd.DataFrame(
        [
            ("Peak demand (kW)", base_peak, scenario_peak, scenario_peak - base_peak),
            ("Available headroom (kW)", base_headroom, scenario_headroom, scenario_headroom - base_headroom),
            ("Warning intervals", base_warning_intervals, scenario_warning_intervals, scenario_warning_intervals - base_warning_intervals),
            ("Breach intervals", base_breach_intervals, scenario_breach_intervals, scenario_breach_intervals - base_breach_intervals),
            ("Breach duration (minutes)", base_breach_minutes, scenario_breach_minutes, scenario_breach_minutes - base_breach_minutes),
        ],
        columns=["Measure", "Baseline Forecast", "Scenario Forecast", "Difference"],
    )

    st.markdown("#### Baseline vs Scenario Comparison")
    st.dataframe(
        scenario_comparison.style.format(
            {
                "Baseline Forecast": "{:.1f}",
                "Scenario Forecast": "{:.1f}",
                "Difference": "{:+.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    additional_headroom_required = max(0.0, scenario_peak - planning_threshold)
    if additional_headroom_required > 0:
        st.warning(
            f"The scenario would require approximately {additional_headroom_required:.1f} kW "
            "of additional planning headroom at the predicted peak."
        )
    else:
        st.success(
            f"The scenario remains within the Planning Capacity Threshold with "
            f"{max(0.0, scenario_headroom):.1f} kW of peak headroom."
        )

    if scenario_peak >= planning_threshold:
        st.warning(
            "The hypothetical scenario exceeds the planning capacity threshold."
        )
    elif scenario_peak >= warning_threshold:
        st.info(
            "The hypothetical scenario reaches the warning threshold but remains "
            "below the planning capacity threshold."
        )
    else:
        st.caption(
            "The hypothetical scenario remains below the warning threshold."
        )

elif nav_selection == "⚙️ Model Performance":
    st.title("⚙️ Model Performance")

    # Final model comparison: baseline + two conventional ML models + LSTM.
    if FINAL_METRICS_FILE.exists():
        try:
            _cmp = pd.read_csv(FINAL_METRICS_FILE)
            st.subheader("Final Model Comparison")
            st.caption(
                "Previous-day same-time is the operational baseline. Random Forest and XGBoost "
                "are conventional machine-learning alternatives; LSTM is the selected deep-learning model."
            )
            _show = _cmp.copy()
            _show.loc[_show["Model"].astype(str).str.upper() == "LSTM", "Role"] = "Selected deep-learning model"
            _show["MAE (kW)"] = _show["MAE_kW"].map(lambda v: f"{v:.3f}")
            _show["RMSE (kW)"] = _show["RMSE_kW"].map(lambda v: f"{v:.3f}")
            _show["R²"] = _show["R2"].map(lambda v: f"{v:.3f}")
            _show["Breach Recall"] = _show["Breach_Recall"].map(lambda v: f"{100*v:.1f}%")
            _show["False Alarm Ratio"] = _show["False_Alarm_Ratio"].map(lambda v: f"{100*v:.1f}%")
            _show["F1"] = _show["F1_Score"].map(lambda v: f"{100*v:.1f}%")
            _show["Balanced Accuracy"] = _show["Balanced_Accuracy"].map(lambda v: f"{100*v:.1f}%")
            _show["Training Time"] = _show["Training_Time_Seconds"].map(lambda v: f"{v:.1f}s")

            _comparison_view = _show[
                ["Model","Role","MAE (kW)","RMSE (kW)","R²","Breach Recall",
                 "False Alarm Ratio","F1","Balanced Accuracy","Training Time"]
            ].copy()

            # Render as a custom HTML table so header and selected-model styling
            # remain consistent across Streamlit versions.
            _headers = list(_comparison_view.columns)
            _table_html = """
            <div class="ov-model-table-wrap">
                <table class="ov-model-table">
                    <thead><tr>
            """
            for _col in _headers:
                _table_html += f"<th>{_col}</th>"
            _table_html += "</tr></thead><tbody>"

            for _, _row in _comparison_view.iterrows():
                _row_class = "selected-model-row" if str(_row["Model"]).upper() == "LSTM" else ""
                _table_html += f'<tr class="{_row_class}">'
                for _col in _headers:
                    _table_html += f"<td>{_row[_col]}</td>"
                _table_html += "</tr>"

            _table_html += "</tbody></table></div>"
            st.markdown(_table_html, unsafe_allow_html=True)

            st.markdown(
                """
                <div class="ov-explanation-panel">
                    <div class="title">Why LSTM was selected</div>
                    <div class="body">
                        LSTM was selected as the preferred model based on the overall evaluation profile,
                        ranking first across 6 of 8 evaluation criteria, including MAE, RMSE, R²,
                        breach recall, F1-score and balanced accuracy. Random Forest recorded the
                        lowest false alarm ratio, while XGBoost recorded the shortest observed
                        training time in the final comparison run.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if FINAL_SELECTION_FILE.exists():
                _sel = pd.read_csv(FINAL_SELECTION_FILE)
                _wins = _sel["Winner"].value_counts()
                w1,w2,w3 = st.columns(3)
                w1.metric("LSTM top-ranked criteria", int(_wins.get("LSTM",0)))
                w2.metric("Random Forest top-ranked criteria", int(_wins.get("Random Forest",0)))
                w3.metric("XGBoost top-ranked criteria", int(_wins.get("XGBoost",0)))
                st.success(
                    "LSTM was selected as the strongest overall model, ranking first across 6 of 8 evaluation criteria. "
                    "Random Forest achieved the lowest false alarm ratio, while XGBoost achieved "
                    "the shortest training time in the final comparison run."
                )
        except Exception as _comparison_error:
            st.caption(f"Final comparison CSV could not be displayed: {_comparison_error}")


    st.subheader("Model Health & Traceability")
    st.markdown(
        f"""
        <div class="ov-info-grid-4">
            <div class="ov-info-card">
                <div class="ov-info-label">Model Health</div>
                <div class="ov-info-value ov-info-value-small">{model_health_status}</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Model Version</div>
                <div class="ov-info-value ov-info-value-small">{model_version}</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Held-Out R²</div>
                <div class="ov-info-value">{f"{model_r2:.3f}" if np.isfinite(model_r2) else "N/A"}</div>
            </div>
            <div class="ov-info-card">
                <div class="ov-info-label">Peak Forecast Range</div>
                <div class="ov-info-value">{forecast_peak_lower:.1f}–{forecast_peak_upper:.1f} kW</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Model health is based on whether held-out MAE and RMSE outperform the previous-day same-time baseline. "
        "The forecast range is an empirical uncertainty indicator derived from held-out residuals."
    )

    metric_test_start = y_test.index.min().strftime("%d %b %Y")
    metric_test_end = y_test.index.max().strftime("%d %b %Y")
    st.caption(
        f"Metric scope: **{st.session_state['current_depot']}** | "
        f"same untouched held-out test period for LSTM and baseline: "
        f"**{metric_test_start} – {metric_test_end}** | "
        f"**{len(y_test):,} observations**"
    )
    
    # Custom metric cards are used instead of st.metric so the full labels
    # remain visible without Streamlit truncating them with ellipses.
    mae_value = f"{model_meta['metrics_xgb']['mae']} kW"
    rmse_value = f"{model_meta['metrics_xgb']['rmse']} kW"
    recall_value = str(model_meta['metrics_xgb']['recall'])
    far_value = str(model_meta['metrics_xgb']['far'])

    st.markdown(
        f"""
        <style>
        .perf-card-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 14px 0 22px 0;
        }}
        .perf-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 14px 16px 18px 20px;
            min-height: 112px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .perf-card-label {{
            color: #64748B;
            font-size: 0.88rem;
            font-weight: 600;
            line-height: 1.25;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            margin-bottom: 8px;
        }}
        .perf-card-value {{
            color: #0F172A;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.05;
        }}
        @media (max-width: 900px) {{
            .perf-card-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 560px) {{
            .perf-card-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>

        <div class="perf-card-grid">
            <div class="perf-card">
                <div class="perf-card-label">Mean Absolute Error</div>
                <div class="perf-card-value">{mae_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">Root Mean Squared Error</div>
                <div class="perf-card-value">{rmse_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">Breach Recall</div>
                <div class="perf-card-value">{recall_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">False Alarm Ratio</div>
                <div class="perf-card-value">{far_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    tbl = {
        "Analytics Architecture Signature": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Breach Recall Capture Rate (TPR)", "False Alarm Ratio (FP / (TP + FP))"],
        "Previous-Day Same-Hour Baseline": [f"{model_meta['metrics_baseline']['mae']} kW", f"{model_meta['metrics_baseline']['rmse']} kW", model_meta['metrics_baseline']['recall'], model_meta['metrics_baseline']['far']],
        "LSTM Depot Demand Forecast Model": [f"{model_meta['metrics_xgb']['mae']} kW", f"{model_meta['metrics_xgb']['rmse']} kW", model_meta['metrics_xgb']['recall'], model_meta['metrics_xgb']['far']]
    }
    _arch_df = pd.DataFrame(tbl).set_index("Analytics Architecture Signature")
    _arch_html = """
    <div class="ov-architecture-table-wrap">
        <table class="ov-architecture-table">
            <thead>
                <tr>
                    <th>Evaluation Metric</th>
                    <th>Previous-Day Same-Time Baseline</th>
                    <th>LSTM Depot Demand Forecast Model</th>
                </tr>
            </thead>
            <tbody>
    """
    for _metric_name, _row in _arch_df.iterrows():
        _arch_html += (
            "<tr>"
            f"<td>{_metric_name}</td>"
            f"<td>{_row.iloc[0]}</td>"
            f"<td>{_row.iloc[1]}</td>"
            "</tr>"
        )
    _arch_html += "</tbody></table></div>"
    st.markdown(_arch_html, unsafe_allow_html=True)

    mae_improvement = (
        100.0
        * (model_meta["metrics_baseline"]["mae"] - model_meta["metrics_xgb"]["mae"])
        / model_meta["metrics_baseline"]["mae"]
        if model_meta["metrics_baseline"]["mae"] > 0
        else 0.0
    )
    rmse_improvement = (
        100.0
        * (model_meta["metrics_baseline"]["rmse"] - model_meta["metrics_xgb"]["rmse"])
        / model_meta["metrics_baseline"]["rmse"]
        if model_meta["metrics_baseline"]["rmse"] > 0
        else 0.0
    )

    st.markdown("#### LSTM Improvement vs Previous-Day Same-Time Baseline")
    st.markdown(
        f"""
        <div class="perf-card-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
            <div class="perf-card">
                <div class="perf-card-label">MAE Reduction vs Baseline</div>
                <div class="perf-card-value">{mae_improvement:.1f}%</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">RMSE Reduction vs Baseline</div>
                <div class="perf-card-value">{rmse_improvement:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "All error metrics, breach metrics, residuals, scatter points and the "
        "baseline comparison on this page are calculated from the same held-out "
        "test rows. The held-out test data are not used to fit the LSTM model."
    )

    st.subheader(
        f"Held-Out Test Analysis — {y_test.index.min():%d %b %Y} to {y_test.index.max():%d %b %Y}"
    )

    # Full held-out test metrics. The 7-day plots below are visual zooms only.
    residuals = y_test.values - xgb_predictions
    test_start_label = y_test.index.min().strftime("%d %b %Y")
    test_end_label = y_test.index.max().strftime("%d %b %Y")
    r2 = r2_score(y_test.values, xgb_predictions)

    zoom_days = 7
    zoom_end = y_test.index.max()
    zoom_start = zoom_end - pd.Timedelta(days=zoom_days)
    zoom_mask = y_test.index >= zoom_start
    y_test_zoom = y_test.loc[zoom_mask]
    xgb_predictions_zoom = np.asarray(xgb_predictions)[zoom_mask]
    residuals_zoom = residuals[zoom_mask]
    zoom_start_label = y_test_zoom.index.min().strftime("%d %b %Y %H:%M")
    zoom_end_label = y_test_zoom.index.max().strftime("%d %b %Y %H:%M")

    # Performance summary row for the selected depot / full held-out period.
    ps1, ps2, ps3 = st.columns(3)
    ps1.metric("Held-out test period", f"{y_test.index.min():%d %b %Y} → {y_test.index.max():%d %b %Y}")
    ps2.metric("Test observations", f"{len(y_test):,}")
    ps3.metric("R²", f"{r2:.3f}")

    st.info(
        "The model-performance metrics use the complete untouched held-out test period. "
        "A 7-day view is also shown below to make individual demand and residual patterns easier to interpret."
    )

    # ================================================================
    # FULL HELD-OUT VIEW
    # ================================================================
    st.markdown("### Full Held-Out Test View")
    st.caption(
        "This is the complete evaluation window used for MAE, RMSE, R², breach recall and false alarm ratio."
    )

    fig_avp_full = go.Figure()
    fig_avp_full.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test.values,
            name="Actual Demand",
            mode="lines",
            line=dict(color="#0F172A", width=1.4),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp_full.add_trace(
        go.Scatter(
            x=y_test.index,
            y=xgb_predictions,
            name="LSTM Prediction",
            mode="lines",
            line=dict(color="#10B981", width=1.2, dash="dash"),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp_full.update_layout(
        title={
            "text": "Actual vs Predicted Demand — Full Held-Out Test Period",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Demand (kW)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.01, xanchor="left"),
        margin=dict(l=20, r=20, t=105, b=35),
    )
    fig_avp_full.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_avp_full.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    _optivolt_plotly_chart(fig_avp_full, use_container_width=True)

    abs_res = np.abs(residuals)
    p95 = float(np.percentile(abs_res, 95))

    fig_res_full = go.Figure()
    fig_res_full.add_hrect(
        y0=-p95,
        y1=p95,
        fillcolor="rgba(16,185,129,0.08)",
        line_width=0,
    )
    fig_res_full.add_trace(
        go.Scatter(
            x=y_test.index,
            y=residuals,
            name="Residual (Actual - Predicted)",
            mode="lines",
            line=dict(color="#EF4444", width=1),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Residual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_res_full.add_hline(
        y=0,
        line=dict(color="#64748B", dash="dash", width=1.2),
    )
    fig_res_full.update_layout(
        title={
            "text": "Residuals Over Time — Full Held-Out Test Period",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Residual (kW)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=95, b=35),
    )
    fig_res_full.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.08),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_res_full.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    st.caption(
        f"Shaded green region = ±95th percentile absolute residual (**{p95:.2f} kW**)."
    )
    _optivolt_plotly_chart(fig_res_full, use_container_width=True)

    # ================================================================
    # 7-DAY ZOOMED VIEW
    # ================================================================
    st.markdown("### 7-Day Zoomed Held-Out View")
    st.caption(
        f"Displayed window: **{zoom_start_label} – {zoom_end_label}**. "
        "This is a visual zoom only; all formal model metrics above still use the full held-out test period."
    )

    fig_avp_zoom = go.Figure()
    fig_avp_zoom.add_trace(
        go.Scatter(
            x=y_test_zoom.index,
            y=y_test_zoom.values,
            name="Actual Demand",
            mode="lines",
            line=dict(color="#0F172A", width=1.6),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp_zoom.add_trace(
        go.Scatter(
            x=y_test_zoom.index,
            y=xgb_predictions_zoom,
            name="LSTM Prediction",
            mode="lines",
            line=dict(color="#10B981", width=1.4, dash="dash"),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp_zoom.update_layout(
        title={
            "text": "Actual vs Predicted Demand — 7-Day Zoomed View",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Demand (kW)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.01, xanchor="left"),
        margin=dict(l=20, r=20, t=105, b=35),
    )
    fig_avp_zoom.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_avp_zoom.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    _optivolt_plotly_chart(fig_avp_zoom, use_container_width=True)

    fig_res_zoom = go.Figure()
    fig_res_zoom.add_hrect(
        y0=-p95,
        y1=p95,
        fillcolor="rgba(16,185,129,0.08)",
        line_width=0,
    )
    fig_res_zoom.add_trace(
        go.Scatter(
            x=y_test_zoom.index,
            y=residuals_zoom,
            name="Residual (Actual - Predicted)",
            mode="lines",
            line=dict(color="#EF4444", width=1.15),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Residual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_res_zoom.add_hline(
        y=0,
        line=dict(color="#64748B", dash="dash", width=1.2),
    )
    fig_res_zoom.update_layout(
        title={
            "text": "Residuals Over Time — 7-Day Zoomed View",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Residual (kW)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=95, b=35),
    )
    fig_res_zoom.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_res_zoom.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    st.caption(
        "The same ±95th percentile residual band from the full test period is retained so the zoomed chart remains directly comparable."
    )
    _optivolt_plotly_chart(fig_res_zoom, use_container_width=True)

    # Plot 3: Actual vs Predicted scatter — full held-out range only
    # One scatter view is retained to avoid redundant/less-informative zooming.
    scatter_max_raw = float(max(y_test.max(), xgb_predictions.max()))
    scatter_max = max(1.0, np.ceil(scatter_max_raw * 1.05))

    st.subheader("Actual vs Predicted Scatter — Full Held-Out Test Period")
    st.caption(
        f"R² = **{r2:.3f}** | "
        f"MAE = **{model_meta['metrics_xgb']['mae']} kW** | "
        f"RMSE = **{model_meta['metrics_xgb']['rmse']} kW**"
    )

    fig_scat_full = go.Figure()
    fig_scat_full.add_trace(
        go.Scatter(
            x=y_test.values,
            y=xgb_predictions,
            mode="markers",
            marker=dict(size=5, opacity=0.24, color="#2563EB"),
            name="Held-out observations",
            hovertemplate=(
                "Actual: %{x:.2f} kW"
                "<br>Predicted: %{y:.2f} kW"
                "<extra></extra>"
            ),
        )
    )
    fig_scat_full.add_trace(
        go.Scatter(
            x=[0, scatter_max],
            y=[0, scatter_max],
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dash"),
            name="Ideal prediction (y = x)",
            hoverinfo="skip",
        )
    )
    fig_scat_full.update_layout(
        xaxis_title="Actual Demand (kW)",
        yaxis_title="LSTM Predicted Demand (kW)",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=55, b=45),
        hovermode="closest",
    )
    fig_scat_full.update_xaxes(
        range=[0, scatter_max],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.55)",
    )
    fig_scat_full.update_yaxes(
        range=[0, scatter_max],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.55)",
    )
    _optivolt_plotly_chart(fig_scat_full, use_container_width=True)

    # Plot 4A: Residual distribution — full held-out range
    mean_residual = float(np.mean(residuals))
    median_residual = float(np.median(residuals))

    st.subheader("Residual Distribution — Full Held-Out Test Period")
    st.caption(
        "Residual = Actual − Predicted | "
        f"Mean residual = **{mean_residual:.2f} kW** | "
        f"Median residual = **{median_residual:.2f} kW**"
    )

    fig_hist_full = go.Figure()
    fig_hist_full.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=60,
            name="Residual count",
            marker=dict(color="#4F66F2"),
            hovertemplate="Residual bin: %{x:.2f} kW<br>Count: %{y}<extra></extra>",
            showlegend=False,
        )
    )

    # Keep visible vertical reference lines.
    fig_hist_full.add_vline(
        x=0,
        line=dict(color="#111827", dash="dash", width=1.6),
    )
    fig_hist_full.add_vline(
        x=mean_residual,
        line=dict(color="#2563EB", dash="dot", width=1.5),
    )
    fig_hist_full.add_vline(
        x=median_residual,
        line=dict(color="#059669", dash="dot", width=1.5),
    )

    # Dummy line traces provide clear legend entries for the reference lines.
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#111827", dash="dash", width=1.6),
            name="Zero error",
            hoverinfo="skip",
        )
    )
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#2563EB", dash="dot", width=1.5),
            name=f"Mean residual = {mean_residual:.2f} kW",
            hoverinfo="skip",
        )
    )
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#059669", dash="dot", width=1.5),
            name=f"Median residual = {median_residual:.2f} kW",
            hoverinfo="skip",
        )
    )

    fig_hist_full.update_layout(
        xaxis_title="Residual (Actual - Predicted) kW",
        yaxis_title="Count",
        template="plotly_white",
        height=410,
        bargap=0.04,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
    )
    fig_hist_full.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_hist_full.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        rangemode="tozero",
    )
    _optivolt_plotly_chart(fig_hist_full, use_container_width=True)

    # Plot 4B: Residual distribution — zoomed central view
    zoom_limit = float(np.percentile(np.abs(residuals), 95))
    zoom_limit = max(1.0, min(5.0, zoom_limit))

    zoomed_residuals = residuals[
        (residuals >= -zoom_limit) & (residuals <= zoom_limit)
    ]
    central_share = 100.0 * len(zoomed_residuals) / len(residuals)

    st.subheader("Residual Distribution — Zoomed Central View")
    st.caption(
        f"Central range: **−{zoom_limit:.2f} to +{zoom_limit:.2f} kW** | "
        f"Contains **{central_share:.1f}%** of held-out residuals | "
        "Full distribution shown above."
    )

    fig_hist_zoom = go.Figure()
    fig_hist_zoom.add_trace(
        go.Histogram(
            x=zoomed_residuals,
            nbinsx=40,
            xbins=dict(
                start=-zoom_limit,
                end=zoom_limit,
                size=(2 * zoom_limit) / 40,
            ),
            name="Residual count",
            marker=dict(color="#4F66F2"),
            hovertemplate="Residual bin: %{x:.2f} kW<br>Count: %{y}<extra></extra>",
            showlegend=False,
        )
    )

    fig_hist_zoom.add_vline(
        x=0,
        line=dict(color="#111827", dash="dash", width=1.6),
    )
    fig_hist_zoom.add_vline(
        x=mean_residual,
        line=dict(color="#2563EB", dash="dot", width=1.5),
    )
    fig_hist_zoom.add_vline(
        x=median_residual,
        line=dict(color="#059669", dash="dot", width=1.5),
    )

    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#111827", dash="dash", width=1.6),
            name="Zero error",
            hoverinfo="skip",
        )
    )
    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#2563EB", dash="dot", width=1.5),
            name=f"Mean residual = {mean_residual:.2f} kW",
            hoverinfo="skip",
        )
    )
    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#059669", dash="dot", width=1.5),
            name=f"Median residual = {median_residual:.2f} kW",
            hoverinfo="skip",
        )
    )

    fig_hist_zoom.update_layout(
        xaxis_title="Residual (Actual - Predicted) kW",
        yaxis_title="Count",
        template="plotly_white",
        height=410,
        bargap=0.04,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
    )
    fig_hist_zoom.update_xaxes(
        range=[-zoom_limit, zoom_limit],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_hist_zoom.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        rangemode="tozero",
    )
    _optivolt_plotly_chart(fig_hist_zoom, use_container_width=True)

elif nav_selection == "🚨 System Alerts":
    st.title("🚨 System Alerts")
    st.caption(
        "Forecast-based alert status for the selected depot using the latest "
        "available Optimise Prime data context."
    )

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])

    if predicted_peak >= planning_threshold:
        alert_status = "Capacity Breach Predicted"
        status_reason = (
            "Forecast peak is at or above the planning capacity threshold."
        )
    elif predicted_peak >= warning_threshold:
        alert_status = "Warning"
        status_reason = (
            "Forecast peak is at or above the warning threshold but below "
            "the planning capacity threshold."
        )
    else:
        alert_status = "No Alert"
        status_reason = "Forecast peak is below the warning threshold."

    # Use the latest available data timestamp rather than the current clock,
    # so the page does not imply live telemetry.
    latest_data_timestamp = history_series.index.max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Peak", f"{predicted_peak:.1f} kW")
    c2.metric("Warning Threshold", f"{warning_threshold:.1f} kW")
    c3.metric("Planning Capacity Threshold", f"{planning_threshold:.1f} kW")
    c4.metric(
        "Forecast Breach Duration",
        _format_duration(total_breach_minutes) if total_breach_minutes else "None",
    )

    if alert_status == "Capacity Breach Predicted":
        st.error(f"**{alert_status}** — {status_reason}")
    elif alert_status == "Warning":
        st.warning(f"**{alert_status}** — {status_reason}")

    ledger = pd.DataFrame([{
        "Forecast Reference Time": latest_data_timestamp.strftime("%d %b %Y, %H:%M"),
        "Depot": st.session_state["current_depot"],
        "Predicted Peak": f"{predicted_peak:.1f} kW",
        "Warning Threshold": f"{warning_threshold:.1f} kW",
        "Planning Capacity Threshold": f"{planning_threshold:.1f} kW",
        "Alert Status": alert_status,
        "Breach Duration": _format_duration(total_breach_minutes) if total_breach_minutes else "None",
        "First Breach Window": (
            f"{first_breach_start_time:%d %b %Y %H:%M} – {first_breach_end_time:%d %b %Y %H:%M}"
            if first_breach_start_time is not None else "None"
        ),
        "Status Reason": status_reason,
    }])

    st.dataframe(
        ledger,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Operational Recommendation")
    if alert_status == "Capacity Breach Predicted":
        st.error(operational_recommendation)
    elif alert_status == "Warning":
        st.warning(operational_recommendation)
    else:
        st.success(operational_recommendation)

    st.caption(
        "Recommendation is decision-support guidance only and does not represent an automatic charging-control action."
    )

    st.info(
        "Forecast status is based on the historical project dataset and does not "
        "represent live grid telemetry."
    )

elif nav_selection == "📋 Reports":
    st.title("📋 Reports")
    st.caption(
        "Export a concise operational model report for the selected depot."
    )

    xgb_metrics = model_meta.get("metrics_xgb", {})

    def _friendly_period(period_text):
        try:
            start_text, end_text = [part.strip() for part in str(period_text).split(" to ", 1)]
            start_dt = pd.to_datetime(start_text)
            end_dt = pd.to_datetime(end_text)
            return f"{start_dt:%d %b %Y} – {end_dt:%d %b %Y}"
        except Exception:
            return str(period_text)

    training_period_raw = model_meta.get("train_period", "Not available")
    validation_period_raw = model_meta.get("validation_period", "Not available")
    test_period_raw = model_meta.get("test_period", "Not available")
    test_rows_value = int(model_meta.get("test_observations", len(y_test)))

    forecast_reference_time = pd.Timestamp(history_series.index.max()).strftime("%d %b %Y %H:%M")
    report_rows_display = [
        ("Depot Name", st.session_state["current_depot"]),
        ("Model", "LSTM"),
        ("Model Version", model_version),
        ("Model Health", model_health_status),
        ("Forecast Reference Time", forecast_reference_time),
        (
            "Planning Capacity Threshold",
            f"{float(st.session_state['grid_limit']):.2f} kW",
        ),
        ("Warning Threshold", f"{float(st.session_state['warn_threshold']):.2f} kW"),
        (
            "Forecast Horizon",
            f"{int(model_meta.get('forecast_horizon_hours', st.session_state['forecast_horizon']))} hours",
        ),
        ("Predicted Peak", f"{float(predicted_peak):.2f} kW"),
        ("Empirical 80% Range at Peak", f"{forecast_peak_lower:.2f}–{forecast_peak_upper:.2f} kW"),
        ("Breach Status", "Predicted" if len(breaches) else "No breach predicted"),
        ("Forecast Breach Duration", _format_duration(total_breach_minutes) if total_breach_minutes else "None"),
        ("Operational Recommendation", operational_recommendation),
        ("MAE", f"{xgb_metrics.get('mae', 'N/A')} kW"),
        ("RMSE", f"{xgb_metrics.get('rmse', 'N/A')} kW"),
        ("Breach Recall", xgb_metrics.get("recall", "N/A")),
        ("False Alarm Ratio", xgb_metrics.get("far", "N/A")),
        ("Training Period", _friendly_period(training_period_raw)),
        ("Validation Period", _friendly_period(validation_period_raw)),
        ("Held-out Test Period", _friendly_period(test_period_raw)),
        ("Test Rows", f"{test_rows_value:,}"),
        (
            "Synthetic Data Used",
            "No" if not bool(model_meta.get("synthetic_data_used", False)) else "Yes",
        ),
    ]

    report_df_display = pd.DataFrame(
        report_rows_display,
        columns=["Report Field", "Value"],
    )

    st.subheader("Operational Asset Report Preview")
    st.dataframe(
        report_df_display,
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    # CSV version keeps Test Rows as a plain integer (7392, not "7,392")
    # while retaining the same reader-friendly report structure.
    report_rows_csv = [
        ("Depot Name", st.session_state["current_depot"]),
        ("Model", "LSTM"),
        ("Model Version", model_version),
        ("Model Health", model_health_status),
        ("Forecast Reference Time", forecast_reference_time),
        ("Planning Capacity Threshold", f"{float(st.session_state['grid_limit']):.2f} kW"),
        ("Warning Threshold", f"{float(st.session_state['warn_threshold']):.2f} kW"),
        (
            "Forecast Horizon",
            f"{int(model_meta.get('forecast_horizon_hours', st.session_state['forecast_horizon']))} hours",
        ),
        ("Predicted Peak", f"{float(predicted_peak):.2f} kW"),
        ("Empirical 80% Range at Peak", f"{forecast_peak_lower:.2f}-{forecast_peak_upper:.2f} kW"),
        ("Breach Status", "Predicted" if len(breaches) else "No breach predicted"),
        ("Forecast Breach Duration", _format_duration(total_breach_minutes) if total_breach_minutes else "None"),
        ("Operational Recommendation", operational_recommendation),
        ("MAE", f"{xgb_metrics.get('mae', 'N/A')} kW"),
        ("RMSE", f"{xgb_metrics.get('rmse', 'N/A')} kW"),
        ("Breach Recall", xgb_metrics.get("recall", "N/A")),
        ("False Alarm Ratio", xgb_metrics.get("far", "N/A")),
        ("Training Period", _friendly_period(training_period_raw)),
        ("Validation Period", _friendly_period(validation_period_raw)),
        ("Held-out Test Period", _friendly_period(test_period_raw)),
        ("Test Rows", test_rows_value),
        (
            "Synthetic Data Used",
            "No" if not bool(model_meta.get("synthetic_data_used", False)) else "Yes",
        ),
    ]

    report_df_csv = pd.DataFrame(
        report_rows_csv,
        columns=["Report Field", "Value"],
    )
    report_csv = report_df_csv.to_csv(index=False).encode("utf-8")

    pdf_bytes = _build_operational_pdf(
        report_rows_display,
        f"{st.session_state['current_depot']} Operational Model Report",
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download Operational Model Report (CSV)",
            data=report_csv,
            file_name=f"{st.session_state['current_depot'].replace(' ', '_')}_Operational_Model_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Download Operational Model Report (PDF)",
            data=pdf_bytes,
            file_name=f"{st.session_state['current_depot'].replace(' ', '_')}_Operational_Model_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.info(
        "Planning capacity is an application planning threshold and not a verified "
        "contracted grid connection limit."
    )

elif nav_selection == "📜 Archive History":
    st.title("📜 Archive History")
    st.caption(
        "Held-out test history for the selected depot. "
        "Residual = Actual Observed − Forecast."
    )

    history_table = pd.DataFrame({
        "Timestamp": y_test.index,
        "Forecast kW": xgb_predictions,
        "Actual Observed kW": y_test.values,
        "Residual (Actual - Predicted) kW": pipeline["errors"],
    })

    # Default to the most recent observations so the table does not open on a
    # long run of overnight zero-demand intervals.
    sort_order = st.selectbox(
        "Display order",
        ["Most recent first", "Oldest first"],
        index=0,
    )

    with st.container(border=True):
        st.markdown("#### 🔎 Archive filter")
        show_non_zero_only = st.checkbox(
            "Show observed-demand intervals only",
            value=True,
            help=(
                "When selected, the table shows only intervals where actual observed "
                "demand is greater than 0.01 kW. Untick it to show all held-out intervals, "
                "including zero-demand periods."
            ),
        )
        st.caption(
            "✓ Checked = observed-demand intervals only  |  "
            "Unchecked = all held-out intervals"
        )

    display_table = history_table.copy()

    if show_non_zero_only:
        # "Non-zero demand" should refer to observed demand, not merely a
        # non-zero model prediction. This keeps the filtered archive focused
        # on intervals where charging demand was actually present.
        display_table = display_table[
            display_table["Actual Observed kW"].abs() > 0.01
        ]

    ascending = sort_order == "Oldest first"
    display_table = display_table.sort_values("Timestamp", ascending=ascending)

    total_rows = len(history_table)
    shown_rows = len(display_table)

    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out observations", f"{total_rows:,}")
    c2.metric("Observed-demand intervals", f"{shown_rows:,}")
    c3.metric(
        "Observed-demand share",
        f"{(shown_rows / total_rows * 100):.1f}%" if show_non_zero_only and total_rows else "100.0%",
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="DD MMM YYYY, HH:mm",
            ),
            "Forecast kW": st.column_config.NumberColumn(
                "Forecast kW",
                format="%.2f",
            ),
            "Actual Observed kW": st.column_config.NumberColumn(
                "Actual Observed kW",
                format="%.2f",
            ),
            "Residual (Actual - Predicted) kW": st.column_config.NumberColumn(
                "Residual (Actual - Predicted) kW",
                format="%.2f",
            ),
        },
        height=520,
    )

    st.caption(
        "Tip: untick **Show observed-demand intervals only** to inspect every held-out "
        "15-minute interval, including zero-demand periods."
    )

elif nav_selection == "⚙️ Settings / Admin":
    st.title("⚙️ Settings / Admin")
    st.caption(
        "Editable application controls are separated from read-only technical and audit information."
    )

    if st.session_state.get("settings_flash_message"):
        st.success(st.session_state["settings_flash_message"])
        st.session_state["settings_flash_message"] = None

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#E8F3FF,#E8F8F4);
            border:1px solid #B7D6EC;
            border-radius:14px;
            padding:14px 16px;
            margin:8px 0 18px 0;">
            <div style="font-size:.72rem;font-weight:800;color:#37627F;
                        text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px;">
                Currently Applied Settings
            </div>
            <div style="font-weight:750;color:#102A43;line-height:1.75;">
                Organisation: {st.session_state.get('org_name', 'Optivolt Solutions Limited')}<br>
                Forecast horizon: {int(st.session_state.get('forecast_horizon', 24))} hours ·
                Chart height: {int(st.session_state.get('preferred_chart_height', 450))} px ·
                Layout: {st.session_state.get('display_density', 'Compact')} ·
                Theme: {st.session_state.get('display_theme', 'Optivolt Blue / Teal')}<br>
                Minimum upload history: {int(st.session_state.get('min_upload_history_days', 28))} days ·
                Uploaded-depot retraining: {int(st.session_state.get('retrain_epochs', LSTM_EPOCHS))} epochs /
                batch {int(st.session_state.get('retrain_batch_size', LSTM_BATCH_SIZE))}
<br>
                Settings audit entries: {len(st.session_state.get('admin_change_log', []))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Editable Application Controls")

    # Organisation
    with st.expander("Organisation Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            admin_org_name = st.text_input(
                "Organisation name",
                value=st.session_state.get("org_name", "Optivolt Solutions Limited"),
                key="admin_org_name",
            )
        with c2:
            admin_portal_title = st.text_input(
                "Portal title",
                value=st.session_state.get("portal_title", "Optivolt EV Depot Intelligence Portal"),
                key="admin_portal_title",
            )

        if st.button("Apply organisation settings", key="apply_org_settings"):
            new_org_name = admin_org_name.strip() or "Optivolt Solutions Limited"
            new_portal_title = admin_portal_title.strip() or "Optivolt EV Depot Intelligence Portal"

            _record_admin_changes(
                "Organisation Settings",
                [
                    ("Organisation name", st.session_state.get("org_name", "Optivolt Solutions Limited"), new_org_name),
                    ("Portal title", st.session_state.get("portal_title", "Optivolt EV Depot Intelligence Portal"), new_portal_title),
                ],
            )

            st.session_state["pending_admin_settings"] = {
                "org_name": new_org_name,
                "portal_title": new_portal_title,
            }
            st.session_state["settings_flash_message"] = "Organisation settings applied."
            st.rerun()

    # Depot / alert settings
    with st.expander("Depot & Alert Settings", expanded=True):
        d1, d2 = st.columns(2)
        with d1:
            admin_capacity = st.number_input(
                "Planning Capacity Threshold (kW)",
                min_value=0.1,
                value=float(st.session_state["grid_limit"]),
                step=0.5,
                key="admin_capacity_kw",
            )

        current_warning_pct = (
            int(round(
                100.0 * float(st.session_state["warn_threshold"])
                / float(st.session_state["grid_limit"])
            ))
            if float(st.session_state["grid_limit"]) > 0 else 80
        )
        current_warning_pct = min(max(current_warning_pct, 50), 95)

        with d2:
            admin_warning_percent = st.slider(
                "Warning threshold percentage",
                min_value=50,
                max_value=95,
                value=current_warning_pct,
                step=1,
                key="admin_warning_percent",
            )

        derived_warning = round(
            float(admin_capacity) * float(admin_warning_percent) / 100.0,
            1,
        )
        st.info(
            f"Derived warning threshold: {derived_warning:.1f} kW "
            f"({admin_warning_percent}% of the Planning Capacity Threshold)."
        )
        st.caption(
            f"Current active thresholds — Planning: {float(st.session_state['grid_limit']):.1f} kW | "
            f"Warning: {float(st.session_state['warn_threshold']):.1f} kW"
        )

        threshold_change_pct = (
            abs(float(admin_capacity) - float(st.session_state["grid_limit"]))
            / float(st.session_state["grid_limit"]) * 100.0
            if float(st.session_state["grid_limit"]) > 0 else 0.0
        )
        large_threshold_change = threshold_change_pct >= 20.0

        if large_threshold_change:
            st.warning(
                f"This changes the Planning Capacity Threshold by {threshold_change_pct:.1f}%. "
                "Confirm the change before applying it."
            )
            confirm_large_threshold_change = st.checkbox(
                "I confirm this material threshold change",
                key="confirm_large_threshold_change",
            )
        else:
            confirm_large_threshold_change = True

        if st.button(
            "Apply depot & alert settings",
            key="apply_depot_alert_settings",
            disabled=not confirm_large_threshold_change,
        ):
            active_depot = st.session_state["current_depot"]
            new_capacity = float(admin_capacity)
            new_warning = float(derived_warning)

            old_capacity = float(st.session_state["grid_limit"])
            old_warning = float(st.session_state["warn_threshold"])

            _record_admin_changes(
                "Depot & Alert Settings",
                [
                    ("Planning Capacity Threshold (kW)", old_capacity, new_capacity),
                    ("Warning Threshold (kW)", old_warning, new_warning),
                    ("Warning Threshold (%)", round(100.0 * old_warning / old_capacity, 1) if old_capacity > 0 else "N/A", int(admin_warning_percent)),
                ],
            )

            st.session_state["grid_limit"] = new_capacity
            st.session_state["warn_threshold"] = new_warning

            # Persist built-in depot overrides for the full Streamlit session.
            # This prevents the sidebar rerun from restoring the original defaults.
            if active_depot in DEPOT_CAPACITY_KW:
                overrides = dict(st.session_state.get("depot_threshold_overrides", {}))
                overrides[active_depot] = {
                    "capacity_kw": new_capacity,
                    "warning_kw": new_warning,
                    "warning_percent": int(admin_warning_percent),
                }
                st.session_state["depot_threshold_overrides"] = overrides

            # Uploaded depots keep their active threshold values in session state.
            elif active_depot in st.session_state.get("uploaded_depots_registry", {}):
                registry = dict(st.session_state.get("uploaded_depots_registry", {}))
                if active_depot in registry:
                    registry[active_depot] = dict(registry[active_depot])
                    registry[active_depot]["capacity_kw"] = new_capacity
                    registry[active_depot]["warning_kw"] = new_warning
                    st.session_state["uploaded_depots_registry"] = registry

            execute_production_ml_pipeline.clear()
            st.session_state["settings_flash_message"] = (
                f"{active_depot} thresholds applied: "
                f"{new_capacity:.1f} kW planning / {new_warning:.1f} kW warning."
            )
            # Clear stale widget values on the next rerun.
            st.session_state["pending_admin_settings"] = {}
            st.rerun()

    # Forecast / display settings
    with st.expander("Forecast & Display Settings", expanded=True):
        fd1, fd2 = st.columns(2)
        horizon_choices = [2, 4, 6, 12, 24]
        current_horizon = int(st.session_state.get("forecast_horizon", 24))
        if current_horizon not in horizon_choices:
            current_horizon = 24

        with fd1:
            admin_horizon = st.selectbox(
                "Default forecast horizon",
                horizon_choices,
                index=horizon_choices.index(current_horizon),
                format_func=lambda h: f"{h} hours",
                key="admin_default_horizon",
            )
            admin_density = st.selectbox(
                "Layout density",
                ["Compact", "Standard"],
                index=["Compact", "Standard"].index(
                    st.session_state.get("display_density", "Compact")
                    if st.session_state.get("display_density", "Compact") in ["Compact", "Standard"]
                    else "Compact"
                ),
                key="admin_display_density",
            )

        with fd2:
            chart_choices = [320, 380, 450, 520, 600]
            current_height = int(st.session_state.get("preferred_chart_height", 450))
            if current_height not in chart_choices:
                current_height = 450
            admin_chart_height = st.selectbox(
                "Chart height",
                chart_choices,
                index=chart_choices.index(current_height),
                format_func=lambda h: f"{h} px",
                key="admin_chart_height",
            )
            theme_choices = ["Optivolt Blue / Teal", "High Contrast Navy"]
            current_theme = st.session_state.get("display_theme", "Optivolt Blue / Teal")
            if current_theme not in theme_choices:
                current_theme = "Optivolt Blue / Teal"
            admin_theme = st.selectbox(
                "Theme",
                theme_choices,
                index=theme_choices.index(current_theme),
                key="admin_display_theme",
            )

        if st.button("Apply forecast & display settings", key="apply_forecast_display"):
            _record_admin_changes(
                "Forecast & Display Settings",
                [
                    ("Forecast horizon (hours)", st.session_state.get("forecast_horizon", 24), int(admin_horizon)),
                    ("Chart height (px)", st.session_state.get("preferred_chart_height", 450), int(admin_chart_height)),
                    ("Layout density", st.session_state.get("display_density", "Compact"), admin_density),
                    ("Theme", st.session_state.get("display_theme", "Optivolt Blue / Teal"), admin_theme),
                ],
            )

            st.session_state["pending_admin_settings"] = {
                "forecast_horizon": int(admin_horizon),
                "preferred_chart_height": int(admin_chart_height),
                "display_density": admin_density,
                "display_theme": admin_theme,
            }
            st.session_state["settings_flash_message"] = (
                f"Forecast/display settings applied: {int(admin_horizon)}h horizon, "
                f"{int(admin_chart_height)}px charts, {admin_density.lower()} layout."
            )
            execute_production_ml_pipeline.clear()
            st.rerun()

    # Upload validation settings
    with st.expander("Data Validation Settings — applies to new uploads"):
        min_days = st.number_input(
            "Minimum upload history (days)",
            min_value=14,
            max_value=90,
            value=int(st.session_state.get("min_upload_history_days", 28)),
            step=1,
            key="admin_min_upload_days",
            help="Applies to new uploaded depot datasets.",
        )
        st.caption(
            "Sampling remains fixed at 15 minutes because the deployed LSTM was designed for a 96-step, 24-hour input sequence."
        )
        if st.button("Apply data validation settings", key="apply_data_validation"):
            _record_admin_changes(
                "Data Validation Settings",
                [
                    ("Minimum upload history (days)", st.session_state.get("min_upload_history_days", 28), int(min_days)),
                ],
            )

            st.session_state["pending_admin_settings"] = {
                "min_upload_history_days": int(min_days),
            }
            st.session_state["settings_flash_message"] = (
                f"Minimum upload history changed to {int(min_days)} days."
            )
            st.rerun()

    # Uploaded-depot retraining defaults
    with st.expander("Uploaded-Depot Retraining Defaults — applies to next retraining"):
        st.warning(
            "These controls apply only to future retraining of uploaded depots. "
            "They do not alter the pre-trained built-in depot models."
        )
        rt1, rt2 = st.columns(2)
        with rt1:
            retrain_epochs = st.number_input(
                "Maximum epochs",
                min_value=5,
                max_value=100,
                value=int(st.session_state.get("retrain_epochs", LSTM_EPOCHS)),
                step=5,
                key="admin_retrain_epochs",
            )
            retrain_batch = st.selectbox(
                "Batch size",
                [32, 64, 128, 256, 512],
                index=[32, 64, 128, 256, 512].index(
                    int(st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE))
                    if int(st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE)) in [32,64,128,256,512]
                    else 256
                ),
                key="admin_retrain_batch",
            )
            retrain_patience = st.number_input(
                "Early stopping patience",
                min_value=2,
                max_value=15,
                value=int(st.session_state.get("retrain_patience", 5)),
                step=1,
                key="admin_retrain_patience",
            )
        with rt2:
            retrain_lr = st.selectbox(
                "Learning rate",
                [0.0001, 0.0005, 0.001, 0.002, 0.005],
                index=[0.0001,0.0005,0.001,0.002,0.005].index(
                    float(st.session_state.get("retrain_learning_rate", 0.001))
                    if float(st.session_state.get("retrain_learning_rate", 0.001)) in [0.0001,0.0005,0.001,0.002,0.005]
                    else 0.001
                ),
                key="admin_retrain_lr",
            )
            retrain_delta = st.selectbox(
                "Early stopping minimum delta",
                [1e-6, 1e-5, 1e-4, 1e-3],
                index=[1e-6,1e-5,1e-4,1e-3].index(
                    float(st.session_state.get("retrain_min_delta", 1e-5))
                    if float(st.session_state.get("retrain_min_delta", 1e-5)) in [1e-6,1e-5,1e-4,1e-3]
                    else 1e-5
                ),
                format_func=lambda v: f"{v:.0e}",
                key="admin_retrain_delta",
            )

        if st.button("Apply retraining defaults", key="apply_retrain_defaults"):
            _record_admin_changes(
                "Uploaded-Depot Retraining Defaults",
                [
                    ("Maximum epochs", st.session_state.get("retrain_epochs", LSTM_EPOCHS), int(retrain_epochs)),
                    ("Batch size", st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE), int(retrain_batch)),
                    ("Early stopping patience", st.session_state.get("retrain_patience", 5), int(retrain_patience)),
                    ("Learning rate", st.session_state.get("retrain_learning_rate", 0.001), float(retrain_lr)),
                    ("Early stopping minimum delta", st.session_state.get("retrain_min_delta", 1e-5), float(retrain_delta)),
                ],
            )

            st.session_state["pending_admin_settings"] = {
                "retrain_epochs": int(retrain_epochs),
                "retrain_batch_size": int(retrain_batch),
                "retrain_patience": int(retrain_patience),
                "retrain_learning_rate": float(retrain_lr),
                "retrain_min_delta": float(retrain_delta),
            }
            st.session_state["settings_flash_message"] = (
                "Uploaded-depot retraining defaults applied. "
                "They will be used for the next uploaded-depot retraining run."
            )
            st.rerun()

    st.markdown("---")
    st.subheader("Model Training Summary (Grounded Pipeline Configuration)")

    training_summary_rows = [
        ("Active depot", st.session_state["current_depot"]),
        ("Active model", model_meta.get("model_type", "LSTM Depot Demand Forecast Model")),
        ("Model version", model_version),
        ("Training / model date", model_meta.get("training_date", "Not available")),
        ("Data source", model_meta.get("dataset_source", "Not available")),
        ("Split strategy", model_meta.get("split_strategy", "Not available")),
        ("Training period", model_meta.get("train_period", "Not available")),
        ("Validation period", model_meta.get("validation_period", "Not available")),
        ("Held-out test period", model_meta.get("test_period", "Not available")),
        ("Held-out observations", model_meta.get("test_observations", "Not available")),
        ("Sampling interval", model_meta.get("sampling_interval", "15 minutes")),
        ("Target variable", model_meta.get("target_variable", "Demand_kW")),
        ("Lookback", f"{LSTM_LOOKBACK} intervals / 24 hours"),
        ("Forecast horizon", f"{int(st.session_state['forecast_horizon'])} hours"),
        ("Uploaded retraining maximum epochs", str(st.session_state.get("retrain_epochs", LSTM_EPOCHS))),
        ("Uploaded retraining batch size", str(st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE))),
        ("Uploaded retraining patience", str(st.session_state.get("retrain_patience", 5))),
        ("Uploaded retraining minimum delta", f"{float(st.session_state.get('retrain_min_delta', 1e-5)):.0e}"),
        ("Uploaded retraining learning rate", str(st.session_state.get("retrain_learning_rate", 0.001))),
        ("Held-out MAE", f"{model_meta.get('metrics_xgb', {}).get('mae', 'N/A')} kW"),
        ("Held-out RMSE", f"{model_meta.get('metrics_xgb', {}).get('rmse', 'N/A')} kW"),
        ("Breach recall", model_meta.get("metrics_xgb", {}).get("recall", "N/A")),
        ("False alarm ratio", model_meta.get("metrics_xgb", {}).get("far", "N/A")),
    ]
    st.dataframe(
        pd.DataFrame(training_summary_rows, columns=["Pipeline Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View full pipeline metadata"):
        st.json(model_meta)

    st.markdown("---")
    st.subheader("Read-Only Model Architecture")
    architecture_rows = [
        ("Input sequence", "96 × 1"),
        ("LSTM layer", "48 units, tanh / sigmoid"),
        ("Dense layer", "24 units, ReLU"),
        ("Output layer", "1 unit"),
        ("Default optimizer", "Adam"),
        ("Default learning rate", "0.001"),
        ("Default loss", "Mean squared error"),
        ("Default maximum epochs", str(LSTM_EPOCHS)),
        ("Default batch size", str(LSTM_BATCH_SIZE)),
        ("Shuffle", "False"),
        ("Random seed", str(MODEL_SEED)),
    ]
    st.dataframe(
        pd.DataFrame(architecture_rows, columns=["Architecture Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The deployed built-in model architecture is intentionally read-only. "
        "Editable retraining defaults above affect only future uploaded-depot retraining."
    )

    st.markdown("---")
    st.subheader("Model Management")
    mm1, mm2 = st.columns(2)
    with mm1:
        if st.button("Clear forecast cache", use_container_width=True, key="clear_forecast_cache"):
            execute_production_ml_pipeline.clear()
            st.toast("Forecast cache cleared.")
    with mm2:
        if st.button("Reload model resources", use_container_width=True, key="reload_model_resources"):
            execute_production_ml_pipeline.clear()
            st.session_state["active_uploaded_pipeline"] = None
            st.session_state["active_uploaded_pipeline_depot"] = None
            st.toast("Model resources will reload on the next request.")

    st.markdown("---")
    st.subheader("Admin & Model Configuration Report")

    admin_report_rows = [
        ("Organisation", st.session_state.get("org_name", "Optivolt Solutions Limited")),
        ("Portal title", st.session_state.get("portal_title", "Optivolt EV Depot Intelligence Portal")),
        ("Active depot", st.session_state["current_depot"]),
        ("Model", "LSTM Depot Demand Forecast Model"),
        ("Model version", model_version),
        ("Forecast horizon", f"{int(st.session_state['forecast_horizon'])} hours"),
        ("Planning Capacity Threshold", f"{float(st.session_state['grid_limit']):.1f} kW"),
        ("Warning Threshold", f"{float(st.session_state['warn_threshold']):.1f} kW"),
        ("Model health", model_health_status),
        ("Training period", model_meta.get("train_period", "Not available")),
        ("Validation period", model_meta.get("validation_period", "Not available")),
        ("Held-out test period", model_meta.get("test_period", "Not available")),
        ("Lookback", f"{LSTM_LOOKBACK} intervals / 24 hours"),
        ("Uploaded retraining maximum epochs", st.session_state.get("retrain_epochs", LSTM_EPOCHS)),
        ("Uploaded retraining batch size", st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE)),
        ("Uploaded retraining patience", st.session_state.get("retrain_patience", 5)),
        ("Uploaded retraining minimum delta", st.session_state.get("retrain_min_delta", 1e-5)),
        ("Uploaded retraining learning rate", st.session_state.get("retrain_learning_rate", 0.001)),
        ("Python", platform.python_version()),
        ("Streamlit", st.__version__),
        ("TensorFlow", tf.__version__),
        ("Data status", "Historical project dataset; not live telemetry"),
    ]
    admin_report_df = pd.DataFrame(admin_report_rows, columns=["Configuration Item", "Value"])

    ar1, ar2 = st.columns(2)
    with ar1:
        st.download_button(
            "Download Admin Configuration (CSV)",
            data=admin_report_df.to_csv(index=False).encode("utf-8"),
            file_name="Optivolt_Admin_Model_Configuration.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_admin_config_csv",
        )

    with ar2:
        admin_pdf_buffer = BytesIO()
        admin_doc = SimpleDocTemplate(
            admin_pdf_buffer,
            pagesize=A4,
            rightMargin=14 * mm,
            leftMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
        )
        admin_styles = getSampleStyleSheet()
        admin_story = [
            Paragraph("Optivolt Admin & Model Configuration Report", admin_styles["Title"]),
            Spacer(1, 8),
            Paragraph(
                "Configuration and traceability snapshot for the historical project dataset.",
                admin_styles["BodyText"],
            ),
            Spacer(1, 10),
        ]
        admin_table_data = [["Configuration Item", "Value"]] + [
            [str(r["Configuration Item"]), str(r["Value"])]
            for _, r in admin_report_df.iterrows()
        ]
        admin_table = Table(admin_table_data, colWidths=[62 * mm, 105 * mm], repeatRows=1)
        admin_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E4F7A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D6E5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F8FD")]),
                ]
            )
        )
        admin_story.append(admin_table)
        admin_doc.build(admin_story)
        admin_pdf_bytes = admin_pdf_buffer.getvalue()

        st.download_button(
            "Download Admin Configuration (PDF)",
            data=admin_pdf_bytes,
            file_name="Optivolt_Admin_Model_Configuration.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_admin_config_pdf",
        )

    st.markdown("---")
    st.subheader("Settings Change Log / Audit Trail")
    st.caption(
        "Records changes made through Settings / Admin during the current application session."
    )

    admin_log = st.session_state.get("admin_change_log", [])
    if admin_log:
        audit_df = pd.DataFrame(admin_log).iloc[::-1].reset_index(drop=True)
        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
        )

        audit_csv = pd.DataFrame(admin_log).to_csv(index=False).encode("utf-8")
        audit_c1, audit_c2 = st.columns([2, 1])

        with audit_c1:
            st.download_button(
                "Download settings audit log",
                data=audit_csv,
                file_name="Optivolt_Settings_Audit_Log.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_admin_audit_log",
            )

        with audit_c2:
            confirm_clear_audit = st.checkbox(
                "Confirm clear",
                key="confirm_clear_admin_audit",
            )

            if st.button(
                "Clear audit log",
                use_container_width=True,
                disabled=not confirm_clear_audit,
                key="clear_admin_audit_log",
            ):
                st.session_state["admin_change_log"] = []
                st.session_state["settings_flash_message"] = "Settings audit log cleared."
                st.rerun()
    else:
        st.info(
            "No Settings / Admin changes have been recorded in this session yet."
        )

    st.markdown("---")
    st.subheader("System Information")
    system_rows = [
        ("Application", st.session_state.get("portal_title", "Optivolt EV Depot Intelligence Portal")),
        ("Application version", "1.1"),
        ("Python", platform.python_version()),
        ("Streamlit", st.__version__),
        ("TensorFlow", tf.__version__),
        ("Pandas", pd.__version__),
        ("NumPy", np.__version__),
        ("Runtime platform", platform.system()),
    ]
    st.dataframe(
        pd.DataFrame(system_rows, columns=["System Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Audit / Traceability")
    uploaded_registry = st.session_state.get("uploaded_depots_registry", {})
    audit_rows = [
        ("Active depot", st.session_state["current_depot"]),
        ("Active model version", model_version),
        ("Model health", model_health_status),
        ("Forecast horizon", f"{int(st.session_state['forecast_horizon'])} hours"),
        ("Latest observed timestamp", history_series.index.max().strftime("%d %b %Y %H:%M")),
        ("Last uploaded depot", st.session_state.get("uploaded_depot_name") or "None"),
        ("Uploaded depot records", str(len(uploaded_registry))),
        ("Data status", "Historical project dataset; not live telemetry"),
    ]
    st.dataframe(
        pd.DataFrame(audit_rows, columns=["Audit Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Reset Controls")
    st.warning(
        "Reset controls affect the current application session. "
        "Built-in source data and model files are not deleted."
    )
    reset_confirm = st.checkbox(
        "I understand that reset actions restore administrative defaults for this session.",
        key="confirm_admin_reset_actions",
    )

    r1, r2 = st.columns(2)
    with r1:
        if st.button(
            "Reset editable settings",
            use_container_width=True,
            key="reset_editable_settings",
            disabled=not reset_confirm,
        ):
            _record_admin_changes(
                "Reset Controls",
                [
                    ("Organisation name", st.session_state.get("org_name", "Optivolt Solutions Limited"), "Optivolt Solutions Limited"),
                    ("Portal title", st.session_state.get("portal_title", "Optivolt EV Depot Intelligence Portal"), "Optivolt EV Depot Intelligence Portal"),
                    ("Forecast horizon (hours)", st.session_state.get("forecast_horizon", 24), 24),
                    ("Chart height (px)", st.session_state.get("preferred_chart_height", 450), 450),
                    ("Layout density", st.session_state.get("display_density", "Compact"), "Compact"),
                    ("Theme", st.session_state.get("display_theme", "Optivolt Blue / Teal"), "Optivolt Blue / Teal"),
                    ("Minimum upload history (days)", st.session_state.get("min_upload_history_days", 28), 28),
                    ("Maximum epochs", st.session_state.get("retrain_epochs", LSTM_EPOCHS), LSTM_EPOCHS),
                    ("Batch size", st.session_state.get("retrain_batch_size", LSTM_BATCH_SIZE), LSTM_BATCH_SIZE),
                    ("Early stopping patience", st.session_state.get("retrain_patience", 5), 5),
                    ("Early stopping minimum delta", st.session_state.get("retrain_min_delta", 1e-5), 1e-5),
                    ("Learning rate", st.session_state.get("retrain_learning_rate", 0.001), 0.001),
                ],
            )

            st.session_state["depot_threshold_overrides"] = {}
            st.session_state["pending_admin_settings"] = {
                "org_name": "Optivolt Solutions Limited",
                "portal_title": "Optivolt EV Depot Intelligence Portal",
                "forecast_horizon": 24,
                "preferred_chart_height": 450,
                "display_density": "Compact",
                "display_theme": "Optivolt Blue / Teal",
                "min_upload_history_days": 28,
                "retrain_epochs": LSTM_EPOCHS,
                "retrain_batch_size": LSTM_BATCH_SIZE,
                "retrain_patience": 5,
                "retrain_min_delta": 1e-5,
                "retrain_learning_rate": 0.001,
            }
            st.session_state["settings_flash_message"] = "Editable settings reset to defaults."
            execute_production_ml_pipeline.clear()
            st.rerun()

    with r2:
        if st.button(
            "Reset session to Bexleyheath",
            use_container_width=True,
            key="reset_session_default",
            disabled=not reset_confirm,
        ):
            st.session_state["current_depot"] = "Bexleyheath"
            st.session_state["grid_limit"] = DEPOT_CAPACITY_KW["Bexleyheath"]
            st.session_state["warn_threshold"] = round(WARNING_ALPHA * DEPOT_CAPACITY_KW["Bexleyheath"], 1)
            st.session_state["forecast_horizon"] = 24
            st.session_state["pending_admin_settings"] = {
                "forecast_horizon": 24,
            }
            st.session_state["uploaded_df"] = None
            st.session_state["uploaded_depot_name"] = None
            st.session_state["uploaded_capacity_kw"] = None
            st.session_state["active_uploaded_pipeline"] = None
            st.session_state["active_uploaded_pipeline_depot"] = None
            st.session_state["depot_threshold_overrides"] = {}
            execute_production_ml_pipeline.clear()
            st.toast("Session reset to Bexleyheath.")
            st.rerun()

st.markdown(f"<div class='commercial-footer'>🛡️ Security Status: Checked | Core Architecture: {model_meta['model_type']} | Horizon: {st.session_state['forecast_horizon']}H</div>", unsafe_allow_html=True)
