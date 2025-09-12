import os
import sys
import math
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error

# ----------------- Config -----------------
CSV_PATH = "trajectories.csv"   # should be in same folder as this script
OUT_DIR = "LSTM_MODEL"
MODEL_NAME = "lstm_xyz_predictor.keras"
SCALER_NAME = "scalers_xyz.pkl"
PRED_OUT = "predictions_at_time.csv"

SEQ_LEN = 8      # lookback window
EPOCHS = 20
BATCH_SIZE = 128
PREDICT_DELTA = True   # use delta-mode (predict change then add to last)
SEED = 42
MAX_VEHICLES = 6
# ------------------------------------------

def set_seed(s):
    np.random.seed(s)
    random.seed(s)
    tf.random.set_seed(s)

def auto_detect_cols(df):
    cols = [c.lower() for c in df.columns.tolist()]
    mapping = {}
    for candidate in ['time','t','sim_time','timestamp','frame','step']:
        if candidate in cols:
            mapping['time'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['vehicle_id','veh_id','id','actor_id','agent_id','vehicle']:
        if candidate in cols:
            mapping['id'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['x','pos_x','px','lon','longitude']:
        if candidate in cols:
            mapping['x'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['y','pos_y','py','lat','latitude']:
        if candidate in cols:
            mapping['y'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['z','pos_z','pz','alt','height']:
        if candidate in cols:
            mapping['z'] = df.columns[cols.index(candidate)]
            break
    return mapping

def load_and_group(csv_path):
    df = pd.read_csv(csv_path)
    mapping = auto_detect_cols(df)
    time_col = mapping.get('time')
    id_col = mapping.get('id')
    x_col = mapping.get('x')
    y_col = mapping.get('y')
    z_col = mapping.get('z')
    if None in (time_col, id_col, x_col, y_col, z_col):
        raise ValueError(f"Could not auto-detect required columns. Mapping: {mapping}. CSV columns: {df.columns.tolist()}")
    df = df[[time_col, id_col, x_col, y_col, z_col]].dropna()
    df = df.sort_values([id_col, time_col])
    return df, time_col, id_col, x_col, y_col, z_col

def build_traces(df, time_col, id_col, x_col, y_col, z_col):
    traces = {}
    per_len = {}
    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        coords = np.vstack([g[x_col].values.astype(float),
                            g[y_col].values.astype(float),
                            g[z_col].values.astype(float)]).T  # (T,3)
        times = g[time_col].values.astype(float)
        traces[str(vid)] = {"coords": coords, "times": times}
        per_len[str(vid)] = coords.shape[0]
    return traces, per_len

def make_model(seq_len, lstm_units=128, dropout=0.2):
    model = Sequential([
        Input(shape=(seq_len, 3)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(max(8, lstm_units//2)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

def train_model_on_traces(traces, seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE, out_dir=OUT_DIR):
    # Build multi-output dataset
    X, y = [], []
    for v in traces.values():
        coords = v["coords"]
        T = coords.shape[0]
        for i in range(T - seq_len):
            seq = coords[i:i+seq_len]    # (seq_len,3)
            nxt = coords[i+seq_len]      # (3,)
            if PREDICT_DELTA:
                tgt = nxt - coords[i+seq_len-1]
            else:
                tgt = nxt
            X.append(seq)
            y.append(tgt)
    if len(X) == 0:
        raise ValueError("No samples generated for training. Try reducing SEQ_LEN or use longer traces.")
    X = np.array(X, dtype=float)   # (N,seq_len,3)
    y = np.array(y, dtype=float)   # (N,3)

    # Fit scalers
    all_pos = np.vstack([v["coords"] for v in traces.values()])  # (total_timesteps,3)
    scaler_X = MinMaxScaler()
    scaler_X.fit(all_pos)   # per-feature scaling for X
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)

    # Scale training data
    N = X.shape[0]
    X_flat = X.reshape(-1, 3)
    X_scaled_flat = scaler_X.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, seq_len, 3)
    y_scaled = scaler_y.transform(y)

    # split
    split = int(0.9 * N) if N > 1 else 1
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = make_model(seq_len)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, MODEL_NAME + ".best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0, min_lr=1e-6),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=0)
    ]
    model.fit(X_train, y_train, validation_split=0.1 if len(X_train) > 1 else 0.0,
              epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Save model + scalers
    mpath = os.path.join(out_dir, MODEL_NAME)
    spath = os.path.join(out_dir, SCALER_NAME)
    model.save(mpath)
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, spath)
    return model, scaler_X, scaler_y, (X_test, y_test)

def load_model_scalers(out_dir=OUT_DIR):
    mpath = os.path.join(out_dir, MODEL_NAME)
    spath = os.path.join(out_dir, SCALER_NAME)
    if os.path.exists(mpath) and os.path.exists(spath):
        model = load_model(mpath)
        d = joblib.load(spath)
        return model, d['scaler_X'], d['scaler_y']
    return None, None, None

def steps_between_times(last_time, target_time, dt_est):
    # number of steps we need to predict forward (rounded)
    return int(round((target_time - last_time) / dt_est))

def estimate_dt_per_vehicle(traces):
    # median dt across vehicles (use their time arrays)
    dts = []
    for v in traces.values():
        t = v["times"]
        if len(t) > 1:
            d = np.median(np.diff(t))
            dts.append(d)
    return float(np.median(dts)) if len(dts) > 0 else 1.0

def ensure_seq_of_length(seq_coords, seq_len):
    # seq_coords shape (T,3) or (L,3) where L < seq_len; pad by repeating first value
    L = seq_coords.shape[0]
    if L >= seq_len:
        return seq_coords[-seq_len:].copy()
    # pad at front by repeating first row
    pad = np.repeat(seq_coords[0:1], repeats=(seq_len - L), axis=0)
    return np.vstack([pad, seq_coords]).astype(float)

def iterative_predict_for_vehicle(model, scaler_X, scaler_y, seq_coords, steps, seq_len=SEQ_LEN, predict_delta=PREDICT_DELTA):
    """
    seq_coords: numpy arr (T,3) with last observed positions (T >= seq_len ideally)
    steps: integer >= 0 number of steps to advance (if 0 -> returns last observed)
    Returns predicted absolute position after `steps` steps.
    """
    if steps <= 0:
        return seq_coords[-1].astype(float)
    # create starting input sequence: last seq_len positions (absolute)
    cur_seq = ensure_seq_of_length(seq_coords, seq_len)  # (seq_len,3)
    for s in range(steps):
        # scale cur_seq
        cur_flat = cur_seq.reshape(-1, 3)
        cur_scaled_flat = scaler_X.transform(cur_flat)
        cur_scaled = cur_scaled_flat.reshape(1, seq_len, 3)
        p_scaled = model.predict(cur_scaled, verbose=0)[0]  # (3,)
        p = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]  # (3,)
        if predict_delta:
            next_pos = cur_seq[-1] + p
        else:
            next_pos = p
        # append to cur_seq and drop first
        cur_seq = np.vstack([cur_seq[1:], next_pos])
    return next_pos.astype(float)

def main():
    set_seed(SEED)
    # get target_time from CLI or input
    if len(sys.argv) > 1:
        try:
            target_time = float(sys.argv[1])
        except:
            print("Invalid time argument. Provide a numeric time (e.g., 1234.0)")
            return
    else:
        val = input("Enter target time (same units as CSV time column): ").strip()
        target_time = float(val)

    if not Path(CSV_PATH).exists():
        print(f"CSV not found at {CSV_PATH}. Put the file in this folder.")
        return

    df, time_col, id_col, x_col, y_col, z_col = load_and_group(CSV_PATH)
    traces, per_len = build_traces(df, time_col, id_col, x_col, y_col, z_col)
    vehicle_ids = sorted(list(traces.keys()))[:MAX_VEHICLES]
    if len(vehicle_ids) == 0:
        print("No vehicles found in CSV.")
        return

    # estimate sampling dt
    dt = estimate_dt_per_vehicle(traces)
    # load or train model
    model, scaler_X, scaler_y = load_model_scalers(OUT_DIR)
    if model is None or scaler_X is None or scaler_y is None:
        print("No saved model found — training new LSTM model on dataset (this may take time)...")
        model, scaler_X, scaler_y, _ = train_model_on_traces(traces, seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE, out_dir=OUT_DIR)
    # prepare predictions
    rows = []
    for vid in vehicle_ids:
        data = traces[vid]
        times = data["times"]
        coords = data["coords"]
        first_t = float(times[0])
        last_t = float(times[-1])
        # case: target time exactly observed -> return observed
        # find if there is exact match (or very close within half dt)
        idx_exact = np.where(np.isclose(times, target_time, atol=dt*0.5))[0]
        if idx_exact.size > 0:
            idx = idx_exact[0]
            pos = coords[idx]
            rows.append([vid, target_time, float(pos[0]), float(pos[1]), float(pos[2]), "observed"])
            continue
        # if target_time < first -> return earliest observed
        if target_time <= first_t:
            pos = coords[0]
            rows.append([vid, target_time, float(pos[0]), float(pos[1]), float(pos[2]), "earliest_available"])
            continue
        # if target_time <= last_t and not exact -> return nearest observed
        if target_time <= last_t:
            # find nearest index
            idx = int(np.argmin(np.abs(times - target_time)))
            pos = coords[idx]
            rows.append([vid, target_time, float(pos[0]), float(pos[1]), float(pos[2]), "nearest_observed"])
            continue
        # target_time > last_t -> need to predict forward
        steps = steps_between_times(last_t, target_time, dt)
        if steps < 1:
            steps = 1
        # if vehicle has fewer than seq_len observations, ensure_seq_of_length will pad
        pred = iterative_predict_for_vehicle(model, scaler_X, scaler_y, coords, steps, seq_len=SEQ_LEN, predict_delta=PREDICT_DELTA)
        rows.append([vid, target_time, float(pred[0]), float(pred[1]), float(pred[2]), f"predicted_{steps}_steps"])
    # Save CSV
    df_out = pd.DataFrame(rows, columns=["vehicle_id","target_time","x_pred","y_pred","z_pred","status"])
    df_out.to_csv(PRED_OUT, index=False)

    # Print clean output block (CSV rows only) separated from logs
    sep = "=" * 80
    print("\n" + sep)
    print("PREDICTIONS AT TIME (CSV rows only) — saved to:", PRED_OUT)
    print("vehicle_id,target_time,x_pred,y_pred,z_pred,status")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}")
    print(sep + "\n")
    # small summary
    print(f"Requested time: {target_time} (estimated dt = {dt}). Vehicles processed: {len(vehicle_ids)}")
    print("Model/scaler directory:", OUT_DIR)
    print("Predictions saved to:", PRED_OUT)

if __name__ == "__main__":
    main()
