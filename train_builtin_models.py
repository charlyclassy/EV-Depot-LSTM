from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

SAMPLING_MINUTES = 15
LOOKBACK = 96
EPOCHS = 40
BATCH_SIZE = 256
SEED = 42

DATA_START = pd.Timestamp("2021-07-08 00:00:00")
TRAIN_END = pd.Timestamp("2022-04-01")
VALID_END = pd.Timestamp("2022-05-10")

DEPOTS = [
    "Bexleyheath", "Dartford", "Islington", "Mount Pleasant",
    "Orpington", "Premier Park", "Whitechapel", "Camden", "Victoria",
]
FILES = {
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
COMBINED = Path("UKPN_OptimisePrime_9_Depots_Processed.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def slug(name):
    return name.lower().replace("&", "and").replace(" ", "_").replace("-", "_")


def load_depot(name):
    if COMBINED.exists():
        raw = pd.read_csv(COMBINED, parse_dates=["timestamp"])
        raw = raw.loc[
            raw["depot_id"].astype(str) == name,
            ["timestamp", "demand_kw"],
        ].copy()
    else:
        raw = pd.read_csv(FILES[name], parse_dates=["timestamp"])[
            ["timestamp", "demand_kw"]
        ].copy()

    raw = raw.rename(columns={"timestamp": "Timestamp", "demand_kw": "Demand_kW"})
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw["Demand_kW"] = pd.to_numeric(raw["Demand_kW"], errors="coerce")
    raw = raw.dropna().sort_values("Timestamp")
    s = (
        raw.set_index("Timestamp")["Demand_kW"]
        .resample("15min")
        .mean()
        .interpolate("time")
    )
    return s.clip(lower=0).to_frame("Demand_kW")


def make_sequences(scaled_values, full_index, target_times):
    positions = pd.Series(np.arange(len(full_index)), index=full_index)
    X, times = [], []
    for ts in pd.DatetimeIndex(target_times):
        if ts not in positions.index:
            continue
        pos = int(positions.loc[ts])
        if pos < LOOKBACK:
            continue
        X.append(scaled_values[pos - LOOKBACK:pos])
        times.append(ts)
    return (
        np.asarray(X, dtype=np.float32).reshape(-1, LOOKBACK, 1),
        pd.DatetimeIndex(times),
    )


def new_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, 1)),
        keras.layers.LSTM(
            48,
            activation="tanh",
            recurrent_activation="sigmoid",
        ),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


for i, depot in enumerate(DEPOTS, start=1):
    print(f"[{i}/{len(DEPOTS)}] Training {depot}")
    df = load_depot(depot)

    train_times = df.index[(df.index >= DATA_START) & (df.index < TRAIN_END)]
    valid_times = df.index[(df.index >= TRAIN_END) & (df.index < VALID_END)]
    if not len(train_times) or not len(valid_times):
        raise RuntimeError(f"Missing expected periods for {depot}")

    development_end = valid_times.max()
    development = df.loc[df.index <= development_end, ["Demand_kW"]]
    x_scaler = StandardScaler().fit(development)
    y_scaler = StandardScaler().fit(development)
    scaled_all = x_scaler.transform(df[["Demand_kW"]]).ravel()

    X_train, t_train = make_sequences(scaled_all, df.index, train_times)
    X_valid, t_valid = make_sequences(scaled_all, df.index, valid_times)
    y_train = y_scaler.transform(
        df.loc[t_train, ["Demand_kW"]]
    ).ravel().astype(np.float32)
    y_valid = y_scaler.transform(
        df.loc[t_valid, ["Demand_kW"]]
    ).ravel().astype(np.float32)

    model = new_model()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=2,
        callbacks=[early_stop],
    )

    s = slug(depot)
    model.save(MODEL_DIR / f"{s}.keras")
    np.savez(
        MODEL_DIR / f"{s}_scalers.npz",
        x_mean=np.asarray(x_scaler.mean_, dtype=float),
        x_scale=np.asarray(x_scaler.scale_, dtype=float),
        y_mean=np.asarray(y_scaler.mean_, dtype=float),
        y_scale=np.asarray(y_scaler.scale_, dtype=float),
    )
    metadata = {
        "depot": depot,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "lookback_intervals": LOOKBACK,
        "sampling_minutes": SAMPLING_MINUTES,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": BATCH_SIZE,
        "train_start": str(t_train.min()),
        "train_end": str(t_train.max()),
        "validation_start": str(t_valid.min()),
        "validation_end": str(t_valid.max()),
    }
    (MODEL_DIR / f"{s}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

print("All built-in depot models saved in ./models")
