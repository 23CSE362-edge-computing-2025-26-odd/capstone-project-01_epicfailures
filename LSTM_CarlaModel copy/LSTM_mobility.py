# lstm_predict_xyz.py
"""
Multi-output LSTM predictor for trajectories.csv
Predicts next (x, y, z) coordinates.

- Default CSV: trajectories.csv (same folder)
- Saves model + scaler to LSTM_MODEL/
- Saves predictions to predictions.csv
- Prints a clean predictions block (only CSV rows) between clear separators
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------- settings ----------------
CSV_PATH = "trajectories.csv"   # file in same folder
SEQ_LEN = 8
EPOCHS = 20
BATCH_SIZE = 128
PREDICT_DELTA = True            # predict delta (next - last) then add to last
USE_CUMULATIVE = False         # keep x,y,z coordinates directly (not cumulative distance)
OUT_DIR = "LSTM_MODEL"
MODEL_NAME = "lstm_xyz_predictor.keras"
SCALER_NAME = "scalers_xyz.pkl"
PRED_OUT = "predictions.csv"
SEED = 42
# ------------------------------------------

def set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def auto_detect_cols(df):
    cols = [c.lower() for c in df.columns.tolist()]
    mapping = {}
    for candidate in ['time','t','sim_time','timestamp','frame_idx','step']:
        if candidate in cols:
            mapping['time'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['vehicle_id','veh_id','id','actor_id','agent_id','vehicle']:
        if candidate in cols:
            mapping['id'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['x','pos_x','px','longitude','lon']:
        if candidate in cols:
            mapping['x'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['y','pos_y','py','latitude','lat']:
        if candidate in cols:
            mapping['y'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['z','pos_z','pz','alt','height']:
        if candidate in cols:
            mapping['z'] = df.columns[cols.index(candidate)]
            break
    return mapping

def load_traces(csv_path, seq_len, use_cumulative=USE_CUMULATIVE, min_len=None,
                time_col=None, id_col=None, x_col=None, y_col=None, z_col=None,
                enforce_equal_length=True, auto_relax=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    mapping = auto_detect_cols(df)
    time_col = time_col or mapping.get('time')
    id_col = id_col or mapping.get('id')
    x_col = x_col or mapping.get('x')
    y_col = y_col or mapping.get('y')
    z_col = z_col or mapping.get('z')

    if None in (time_col, id_col, x_col, y_col, z_col):
        raise ValueError(f"Could not detect all required columns. Detected mapping: {mapping}. CSV columns: {df.columns.tolist()}")

    if min_len is None:
        min_len = seq_len + 50

    df = df[[time_col, id_col, x_col, y_col, z_col]].dropna()
    df = df.sort_values([id_col, time_col])

    per_len = {}
    traces = {}
    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        xs = g[x_col].values.astype(float)
        ys = g[y_col].values.astype(float)
        zs = g[z_col].values.astype(float)
        # if use_cumulative==True you'd compute cumulative distance; we keep raw coords for xyz
        coords = np.vstack([xs, ys, zs]).T  # shape (T,3)
        per_len[str(vid)] = coords.shape[0]
        if coords.shape[0] >= min_len:
            traces[str(vid)] = coords.copy()

    # print dataset stats (concise)
    lens = np.array(list(per_len.values())) if per_len else np.array([])
    if lens.size > 0:
        print(f"Loaded {len(per_len)} vehicles. trace lengths: count={len(lens)}, mean={lens.mean():.1f}, median={np.median(lens):.1f}, min={lens.min()}, max={lens.max()}")
    # sample per-vehicle (first 8)
    for k, v in list(per_len.items())[:8]:
        print(f"  {k}: {v}")

    if len(traces) == 0 and auto_relax:
        # try relaxed thresholds
        relaxed = max(seq_len + 1, seq_len + 5)
        print(f"No traces >= {min_len}. Trying relaxed_min={relaxed}")
        traces_relaxed = {}
        for vid, g in df.groupby(id_col):
            g = g.sort_values(time_col)
            xs = g[x_col].values.astype(float)
            ys = g[y_col].values.astype(float)
            zs = g[z_col].values.astype(float)
            coords = np.vstack([xs, ys, zs]).T
            if coords.shape[0] >= relaxed:
                traces_relaxed[str(vid)] = coords.copy()
        if len(traces_relaxed) > 0:
            traces = traces_relaxed
            enforce_equal_length = False
            print(f"Proceeding with {len(traces)} traces (relaxed).")
        else:
            final_min = seq_len + 1
            print(f"No traces for relaxed_min. Trying final_min={final_min}")
            traces_final = {}
            for vid, g in df.groupby(id_col):
                g = g.sort_values(time_col)
                xs = g[x_col].values.astype(float)
                ys = g[y_col].values.astype(float)
                zs = g[z_col].values.astype(float)
                coords = np.vstack([xs, ys, zs]).T
                if coords.shape[0] >= final_min:
                    traces_final[str(vid)] = coords.copy()
            if len(traces_final) > 0:
                traces = traces_final
                enforce_equal_length = False
                print(f"Proceeding with {len(traces)} traces (final_min).")

    if len(traces) == 0:
        raise ValueError("No vehicle trace long enough. Reduce seq_len/min_len or provide longer traces.")

    if enforce_equal_length:
        trim_to = min(len(v) for v in traces.values())
        for k in list(traces.keys()):
            traces[k] = traces[k][:trim_to].copy()

    return traces

def build_dataset_multi(traces, seq_len, predict_delta=PREDICT_DELTA):
    X, y = [], []
    for coords in traces.values():  # coords shape (T,3)
        T = coords.shape[0]
        for i in range(T - seq_len):
            seq = coords[i:i+seq_len]         # shape (seq_len,3)
            nxt = coords[i+seq_len]          # shape (3,)
            if predict_delta:
                target = nxt - coords[i+seq_len-1]
            else:
                target = nxt
            X.append(seq)
            y.append(target)
    if len(X) == 0:
        return np.empty((0, seq_len, 3)), np.empty((0, 3))
    return np.array(X, dtype=float), np.array(y, dtype=float)

def make_model_multi(seq_len, lstm_units=128, dropout=0.2):
    model = Sequential([
        Input(shape=(seq_len, 3)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(max(8, lstm_units//2)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(3)   # three outputs: x, y, z (or deltas)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

def save_model_scalers(model, scaler_X, scaler_y, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, MODEL_NAME)
    spath = os.path.join(out_dir, SCALER_NAME)
    model.save(mpath)
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, spath)
    print(f"Model saved to: {mpath}")
    print(f"Scalers saved to: {spath}")

def load_model_scalers(out_dir=OUT_DIR):
    mpath = os.path.join(out_dir, MODEL_NAME)
    spath = os.path.join(out_dir, SCALER_NAME)
    if not os.path.exists(mpath) or not os.path.exists(spath):
        raise FileNotFoundError("Saved model or scalers not found in " + out_dir)
    model = load_model(mpath)
    d = joblib.load(spath)
    return model, d['scaler_X'], d['scaler_y']

def main():
    set_seed(SEED)
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}. Place it in the script folder and re-run.")
        return

    traces = load_traces(CSV_PATH, SEQ_LEN, use_cumulative=USE_CUMULATIVE, auto_relax=True)
    print(f"Using {len(traces)} traces for training.")

    X_raw, y_raw = build_dataset_multi(traces, SEQ_LEN, predict_delta=PREDICT_DELTA)
    print(f"Total samples generated: {len(X_raw)}")
    if len(X_raw) < 10:
        print("Warning: very few samples (<10). Consider lowering SEQ_LEN or using longer traces.")

    # Flatten positions across all traces for scaler fit
    all_pos = np.vstack([v for v in traces.values()])  # shape (total_timesteps, 3)
    scaler_X = MinMaxScaler()
    scaler_X.fit(all_pos)   # fit per-feature on entire dataset positions

    # y_raw is shape (N,3)
    scaler_y = MinMaxScaler()
    if y_raw.size > 0:
        scaler_y.fit(y_raw)

    # prepare scaled arrays
    N = X_raw.shape[0]
    X_flat = X_raw.reshape(-1, 3)            # (N*seq_len, 3)
    X_scaled_flat = scaler_X.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, SEQ_LEN, 3)
    y_scaled = scaler_y.transform(y_raw) if y_raw.size > 0 else np.empty((0,3))

    # train/test split
    split = int(0.9 * N) if N > 1 else 1
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = make_model_multi(SEQ_LEN, lstm_units=128, dropout=0.2)
    ckpt = os.path.join(OUT_DIR, MODEL_NAME + ".best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1 if len(X_train) > 1 else 0.0,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test
    preds_scaled = model.predict(X_test) if len(X_test) > 0 else np.empty((0,3))
    preds = scaler_y.inverse_transform(preds_scaled) if preds_scaled.size > 0 else np.empty((0,3))
    y_true = scaler_y.inverse_transform(y_test) if y_test.size > 0 else np.empty((0,3))

    # If predict_delta, convert predictions to absolute positions by adding last pos in seq
    if PREDICT_DELTA and len(preds) > 0:
        last_positions = X_raw[split:][:, -1, :]  # last position in each test sequence before transform
        preds_abs = preds + last_positions
        y_true_abs = y_true + last_positions
    else:
        preds_abs = preds
        y_true_abs = y_true

    # compute MSE per coordinate
    if len(preds_abs) > 0:
        mse_x = mean_squared_error(y_true_abs[:,0], preds_abs[:,0])
        mse_y = mean_squared_error(y_true_abs[:,1], preds_abs[:,1])
        mse_z = mean_squared_error(y_true_abs[:,2], preds_abs[:,2])
    else:
        mse_x = mse_y = mse_z = float('nan')

    # Save model + scalers
    save_model_scalers(model, scaler_X, scaler_y, out_dir=OUT_DIR)

    # Build predictions output for ALL test samples (CSV with header x_pred,y_pred,z_pred,x_true,y_true,z_true)
    rows = []
    for i in range(len(preds_abs)):
        px, py, pz = preds_abs[i]
        tx, ty, tz = y_true_abs[i]
        rows.append([px, py, pz, tx, ty, tz])
    df_out = pd.DataFrame(rows, columns=["x_pred","y_pred","z_pred","x_true","y_true","z_true"])
    df_out.to_csv(PRED_OUT, index=False)

    # --- Now print clean output block (only CSV rows) separated from logs ---
    sep = "=" * 80
    header = "x_pred,y_pred,z_pred,x_true,y_true,z_true"
    print("\n" + sep)
    print("PREDICTIONS (CSV rows only) — saved to:", PRED_OUT)
    print(header)
    # print rows only
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}")
    print(sep + "\n")

    # Also print a small summary (outside the clean block)
    print(f"Test samples: {len(preds_abs)}")
    print(f"MSE x: {mse_x:.6f}, y: {mse_y:.6f}, z: {mse_z:.6f}")
    print("Model and scalers saved to:", OUT_DIR)
    print("Predictions saved to:", PRED_OUT)

if __name__ == "__main__":
    main()
