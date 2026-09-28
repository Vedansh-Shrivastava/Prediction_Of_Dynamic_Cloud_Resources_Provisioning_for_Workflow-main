from pathlib import Path
import json

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "AzureReadings_at_a_timestamp.csv"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURES = ["min cpu", "max cpu", "avg cpu"]
LOOK_BACK = 5


def make_sequences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    for index in range(len(values) - LOOK_BACK):
        inputs.append(values[index : index + LOOK_BACK, :])
        targets.append(values[index + LOOK_BACK, :])
    return np.asarray(inputs, dtype="float32"), np.asarray(targets, dtype="float32")


def main() -> None:
    keras.utils.set_random_seed(42)
    frame = pd.read_csv(DATA_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp")
    values = frame[FEATURES].astype("float32").to_numpy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values).astype("float32")
    train_size = int(len(scaled) * 0.8)
    train_values = scaled[:train_size]
    train_x, train_y = make_sequences(train_values)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(LOOK_BACK, len(FEATURES))),
            keras.layers.LSTM(128),
            keras.layers.Dense(len(FEATURES)),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    model.fit(
        train_x,
        train_y,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        verbose=2,
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    model.save(ARTIFACT_DIR / "workload_lstm.keras")
    joblib.dump(scaler, ARTIFACT_DIR / "workload_scaler.joblib")
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "look_back": LOOK_BACK,
                "frequency": "5 minutes",
                "training_rows": int(len(train_values)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model and scaler to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
