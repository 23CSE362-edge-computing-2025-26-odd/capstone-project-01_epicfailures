# lstm_mobility.py
"""
LSTM mobility predictor adapted to the provided dataset (dataset/traces.csv).

Key behavior:
- Default CSV: dataset/traces.csv
- Saves model + scaler to LSTM_MODEL/
- Uses cumulative distance by default
- Predicts delta by default (predict next-step change and add to last pos)
- Default epochs = 20
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --------- Utilities ----------
def set_global_seed(seed):
    """Set random seeds for reproducibility (numpy, python, tf)."""
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# --------- Data loading ----------
# ---------- Replace load_traces_from_csv with this improved version ----------
def load_traces_from_csv(csv_path, seq_len, min_len=None, time_col='sim_time',
                         id_col='veh_id', x_col='x', y_col='y', use_cumulative=True,
                         enforce_equal_length=True, trim_to=None, auto_relax=True):
    """
    Load mobility CSV and return traces dict {veh_id: np.array(positions)}.

    Improvements:
    - prints per-vehicle lengths (debugging)
    - if auto_relax=True will try a relaxed min_len (seq_len + 5) and/or
      disable enforce_equal_length if strict filtering leaves 0 traces.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # validate/auto-detect columns (same as before)
    cols = df.columns.tolist()
    # fallback mapping
    if time_col not in cols:
        for c in ['time', 't', 'sim_time']:
            if c in cols:
                time_col = c; break
    if id_col not in cols:
        for c in ['veh_id', 'id', 'vehicle_id']:
            if c in cols:
                id_col = c; break
    if x_col not in cols:
        for c in ['x', 'pos_x', 'lon']:
            if c in cols:
                x_col = c; break
    if y_col not in cols:
        for c in ['y', 'pos_y', 'lat']:
            if c in cols:
                y_col = c; break

    required = [time_col, id_col, x_col, y_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}. Available columns: {cols}")

    if min_len is None:
        min_len = seq_len + 50  # original conservative default

    df = df[[time_col, id_col, x_col, y_col]].dropna(subset=[time_col, id_col, x_col, y_col])
    df = df.sort_values([id_col, time_col])

    per_vid_lengths = {}
    traces = {}
    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        xs = g[x_col].values.astype(float)
        ys = g[y_col].values.astype(float)

        if use_cumulative:
            deltas = np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2)
            seq = np.cumsum(deltas)
        else:
            seq = xs

        per_vid_lengths[str(vid)] = len(seq)
        if len(seq) >= min_len:
            traces[str(vid)] = seq.copy()

    # DEBUG: print per-vehicle counts (brief)
    print("Per-vehicle lengths (sample, first 10):")
    sample_items = list(per_vid_lengths.items())[:10]
    for k, v in sample_items:
        print(f"  {k}: {v}")

    if len(traces) == 0 and auto_relax:
        # Try relaxing: reduce min_len to seq_len + 5 and/or allow variable lengths
        relaxed_min = max(seq_len + 1, seq_len + 5)
        print(f"No traces >= {min_len}. Trying relaxed min_len={relaxed_min} and enforce_equal_length=False.")
        traces_relaxed = {}
        for vid, g in df.groupby(id_col):
            g = g.sort_values(time_col)
            xs = g[x_col].values.astype(float)
            ys = g[y_col].values.astype(float)
            if use_cumulative:
                deltas = np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2)
                seq = np.cumsum(deltas)
            else:
                seq = xs
            if len(seq) >= relaxed_min:
                traces_relaxed[str(vid)] = seq.copy()
        if len(traces_relaxed) > 0:
            print(f"Found {len(traces_relaxed)} traces with relaxed_min={relaxed_min}. Proceeding with those (variable lengths).")
            traces = traces_relaxed
            enforce_equal_length = False
        else:
            # last resort: accept any trace of length >= seq_len+1
            final_min = seq_len + 1
            print(f"No traces for relaxed_min={relaxed_min}. Trying final_min={final_min}.")
            traces_final = {}
            for vid, g in df.groupby(id_col):
                g = g.sort_values(time_col)
                xs = g[x_col].values.astype(float)
                ys = g[y_col].values.astype(float)
                if use_cumulative:
                    deltas = np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2)
                    seq = np.cumsum(deltas)
                else:
                    seq = xs
                if len(seq) >= final_min:
                    traces_final[str(vid)] = seq.copy()
            if len(traces_final) > 0:
                print(f"Found {len(traces_final)} traces with final_min={final_min}. Proceeding with those.")
                traces = traces_final
                enforce_equal_length = False

    if len(traces) == 0:
        # nothing we can do
        raise ValueError("No vehicle trace long enough. Check CSV or reduce seq_len/min_len manually.")

    if enforce_equal_length:
        if trim_to is None:
            trim_to = min(len(v) for v in traces.values())
        for k in list(traces.keys()):
            traces[k] = traces[k][:trim_to].copy()

    return traces

def build_dataset_from_traces(traces, seq_len, predict_delta=False):
    """
    Build (X, y):
      - X: shape (n_samples, seq_len) raw scalar sequences
      - y: shape (n_samples,) target (next absolute or delta)
    If predict_delta=True, y = pos[t+1] - pos[t]
    """
    X, y = [], []
    for pos in traces.values():
        n = len(pos)
        for i in range(n - seq_len):
            seq = pos[i:i+seq_len]
            next_idx = i + seq_len
            if next_idx >= n:
                break
            if predict_delta:
                target = pos[next_idx] - pos[next_idx - 1]
            else:
                target = pos[next_idx]
            X.append(seq)
            y.append(target)
    if len(X) == 0:
        return np.empty((0, seq_len)), np.empty((0,))
    return np.array(X, dtype=float), np.array(y, dtype=float)

# --------- Model ----------
def make_lstm(seq_len, lstm_units=128, dropout=0.2):
    """Create a stacked LSTM model with dropout and a small dense head."""
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(max(8, lstm_units // 2)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# --------- Train / save / load ----------
def save_model_and_scaler(model, scaler, out_dir="LSTM_MODEL",
                          model_out_name="lstm_mobility_predictor.keras",
                          scaler_out_name="scaler.pkl"):
    """Save keras model and scaler tuple (scaler_X, scaler_y) inside out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    model_out_path = os.path.join(out_dir, model_out_name)
    scaler_out_path = os.path.join(out_dir, scaler_out_name)

    model.save(model_out_path)
    joblib.dump(scaler, scaler_out_path)
    print(f"✅ Model saved to {model_out_path}")
    print(f"✅ Scaler saved to {scaler_out_path}")

def load_model_and_scaler(model_path="LSTM_MODEL/lstm_mobility_predictor.keras",
                          scaler_path="LSTM_MODEL/scaler.pkl"):
    """Load model and scaler from LSTM_MODEL folder."""
    if not os.path.exists(model_path):
        alt = model_path.replace(".keras", ".h5")
        if os.path.exists(alt):
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def train_lstm(traces, seq_len=8, epochs=20, batch_size=128, validation_split=0.1,
               out_dir="LSTM_MODEL", model_out_name="lstm_mobility_predictor.keras",
               scaler_out_name="scaler.pkl", lstm_units=128, dropout=0.2,
               predict_delta=False, seed=42, verbose=1, patience=6):
    """
    Trains LSTM with callbacks and optional delta targets.
    Returns: model, scalers_tuple (scaler_X, scaler_y), history, X_test, y_test, mse_scaled
    """
    set_global_seed(seed)

    X_raw, y_raw = build_dataset_from_traces(traces, seq_len, predict_delta=predict_delta)
    if X_raw.size == 0:
        raise ValueError("No training samples generated - check seq_len / traces.")

    # Fit scaler for X on all positions (or x-values)
    all_positions = np.concatenate([traces[k] for k in traces])
    scaler_X = MinMaxScaler()
    scaler_X.fit(all_positions.reshape(-1, 1))

    X_scaled = scaler_X.transform(X_raw.reshape(-1, 1)).reshape(-1, seq_len, 1)

    # Fit scaler for y (handles absolute or delta targets)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_raw.reshape(-1, 1))
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).reshape(-1, 1)

    # train/test split (90/10)
    split_idx = int(0.9 * len(X_scaled)) if len(X_scaled) > 1 else 1
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    model = make_lstm(seq_len, lstm_units=lstm_units, dropout=dropout)

    # Callbacks: early stop + reduce LR + checkpoint
    ckpt_path = os.path.join(out_dir, model_out_name + ".best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience//3), verbose=0, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split if len(X_train) > 1 else 0.0,
        callbacks=callbacks,
        verbose=verbose
    )

    pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])
    mse_scaled = float(mean_squared_error(y_test, pred_test)) if len(X_test) > 0 else float('nan')

    # Save model and both scalers together as a tuple
    save_model_and_scaler(model, (scaler_X, scaler_y), out_dir=out_dir,
                          model_out_name=model_out_name, scaler_out_name=scaler_out_name)

    return model, (scaler_X, scaler_y), history, X_test, y_test, mse_scaled

# --------- Prediction ----------
def predict_next_pos(raw_seq, model, scalers, seq_len, predict_delta=False):
    """
    raw_seq: array-like length >= seq_len (if longer, last seq_len used)
    model: keras model
    scalers: tuple (scaler_X, scaler_y) returned by train_lstm/save
    predict_delta: whether model was trained to predict delta; must match training mode
    Returns predicted next absolute position (float).
    """
    scaler_X, scaler_y = scalers
    arr = np.array(raw_seq, dtype=float).reshape(-1)
    if arr.size < seq_len:
        raise ValueError(f"raw_seq length {arr.size} < seq_len {seq_len}")
    if arr.size > seq_len:
        arr = arr[-seq_len:]
    x = arr.reshape(-1, 1)
    x_scaled = scaler_X.transform(x).reshape(1, seq_len, 1)
    p_scaled = model.predict(x_scaled, verbose=0)[0, 0]
    p_unscaled = scaler_y.inverse_transform(np.array([[p_scaled]]))[0, 0]
    if predict_delta:
        # predicted delta -> next pos = last pos + delta
        return float(arr[-1] + p_unscaled)
    else:
        return float(p_unscaled)

# --------- Small CLI for quick run ----------
# --------- Small CLI for quick run ----------
if __name__ == "__main__":
    # now expects traces.csv in the same directory as this script
    CSV_PATH = "traces.csv"
    SEQ_LEN = 8      # lower if traces are short (e.g., 4)
    EPOCHS = 20
    PREDICT_DELTA = True
    USE_CUMULATIVE = True
    BATCH_SIZE = 128
    OUT_DIR = "LSTM_MODEL"

    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}. Put traces.csv in same folder and re-run.")
    else:
        print("Loading traces...")
        # pass a small min_len or let auto_relax handle it
        traces = load_traces_from_csv(
            CSV_PATH,
            seq_len=SEQ_LEN,
            use_cumulative=USE_CUMULATIVE,
            enforce_equal_length=True,   # we allow loader to relax if necessary
            auto_relax=True
        )
        n_traces = len(traces)
        print(f"Loaded {n_traces} vehicle traces. Example IDs: {list(traces.keys())[:5]}")
        # Quick dataset check
        X_raw, y_raw = build_dataset_from_traces(traces, SEQ_LEN, predict_delta=PREDICT_DELTA)
        print(f"Generated {len(X_raw)} training samples from traces.")
        if len(X_raw) < 10:
            print("Warning: Very few training samples (<10). Consider reducing SEQ_LEN or collecting longer traces.")
        # Train
        print(f"Training improved LSTM (predict_delta={PREDICT_DELTA}) for {EPOCHS} epochs...")
        model, scalers, history, X_test, y_test, mse = train_lstm(
            traces,
            seq_len=SEQ_LEN,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            out_dir=OUT_DIR,
            model_out_name="lstm_mobility_predictor.keras",
            scaler_out_name="scaler.pkl",
            lstm_units=128,
            dropout=0.2,
            predict_delta=PREDICT_DELTA,
            seed=42,
            verbose=1,
            patience=6
        )
        print("Scaled MSE on test:", mse)
        vid = list(traces.keys())[0]
        seq = traces[vid][:SEQ_LEN]
        pred = predict_next_pos(seq, model, scalers, SEQ_LEN, predict_delta=PREDICT_DELTA)
        print("Example sequence (first vehicle first SEQ_LEN values):", seq)
        print("Predicted next (absolute):", pred)