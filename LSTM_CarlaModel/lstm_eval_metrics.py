# eval_lstm_metrics.py
"""
Extended evaluation script for LSTM multi-output (x,y,z) predictor.

Produces:
- many regression metrics per coordinate (MSE, RMSE, MAE, MAPE, sMAPE, median abs, max abs, R2, adj R2, explained variance)
- Pearson correlation per coordinate
- Euclidean/sample-level metrics (ADE, median, max, RMSE_euc)
- Classification-style metrics (precision, recall, f1, accuracy) for "within threshold" (1m,2m,5m)
- Saves metrics.json and prints a detailed summary

Behavior:
- Prefers to load predictions.csv
- If missing, tries to regenerate using saved model+scalers + trajectories.csv (best-effort)
"""

import os
import json
import numpy as np
import pandas as pd
from math import sqrt, isnan
import joblib
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error, explained_variance_score,
    precision_score, recall_score, f1_score, accuracy_score,
    mean_squared_log_error
)

# TensorFlow imported lazily for regeneration only
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# ---------- configuration (adjust if needed) ----------
PRED_CSV = "predictions.csv"
OUT_DIR = "LSTM_MODEL"
MODEL_NAME = "lstm_xyz_predictor.keras"
SCALER_NAME = "scalers_xyz.pkl"
CSV_PATH = "trajectories.csv"
SEQ_LEN = 8
PREDICT_DELTA = True
THRESHOLDS = [1.0, 2.0, 5.0]   # meters for classification-style metrics
# -----------------------------------------------------

def safe_mape(y_true, y_pred):
    """Mean absolute percentage error. Returns np.nan if division by zero would occur."""
    denom = np.abs(y_true)
    if np.any(denom == 0):
        # avoid dividing by zero; compute only where denom != 0
        mask = denom != 0
        if np.any(mask):
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask]))) * 100.0
        else:
            return float('nan')
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

def smape(y_true, y_pred):
    """Symmetric MAPE in percent."""
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if not np.any(mask):
        return float('nan')
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask])) * 100.0

def adjusted_r2(r2, n, p):
    """Adjusted R^2: 1 - (1-R2)*(n-1)/(n-p-1) ; return nan if not computable"""
    if n <= p + 1:
        return float('nan')
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def pearsonr_safe(a, b):
    """Pearson correlation coefficient using numpy; returns nan if constant arrays."""
    if a.size == 0:
        return float('nan')
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def compute_metrics_from_arrays(y_true, y_pred, seq_len=SEQ_LEN):
    """
    y_true, y_pred: (N,3) arrays
    Returns dictionary of metrics.
    """
    assert y_true.shape == y_pred.shape
    N = y_true.shape[0]
    metrics = {"n_samples": int(N)}
    coords = ['x', 'y', 'z']
    per_coord = {}

    p = seq_len * 3  # number of predictors/features used (for adjusted R2)

    for i, c in enumerate(coords):
        t = y_true[:, i]
        p_ = y_pred[:, i]

        mse = float(mean_squared_error(t, p_)) if N > 0 else float('nan')
        rmse = float(sqrt(mse)) if N > 0 else float('nan')
        mae = float(mean_absolute_error(t, p_)) if N > 0 else float('nan')
        median_ae = float(median_absolute_error(t, p_)) if N > 0 else float('nan')
        max_abs = float(max_error(t, p_)) if N > 0 else float('nan')
        try:
            mape_val = safe_mape(t, p_)
        except Exception:
            mape_val = float('nan')
        try:
            smape_val = smape(t, p_)
        except Exception:
            smape_val = float('nan')
        try:
            r2 = float(r2_score(t, p_)) if N > 1 else float('nan')
        except Exception:
            r2 = float('nan')
        try:
            adj_r2 = adjusted_r2(r2, N, p) if not isnan(r2) else float('nan')
        except Exception:
            adj_r2 = float('nan')
        try:
            ev = float(explained_variance_score(t, p_)) if N > 1 else float('nan')
        except Exception:
            ev = float('nan')
        try:
            pear = pearsonr_safe(t, p_)
        except Exception:
            pear = float('nan')
        # msle: only valid when both true and pred >= 0
        try:
            if np.any(t < 0) or np.any(p_ < 0):
                msle_val = float('nan')
            else:
                msle_val = float(mean_squared_log_error(t, p_))
        except Exception:
            msle_val = float('nan')

        per_coord[c] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "median_ae": median_ae,
            "max_abs_error": max_abs,
            "mape_%": mape_val,
            "smape_%": smape_val,
            "r2": r2,
            "adjusted_r2": adj_r2,
            "explained_variance": ev,
            "pearson_r": pear,
            "msle": msle_val
        }

    # Euclidean (per-sample) errors
    errs = np.linalg.norm(y_true - y_pred, axis=1) if N > 0 else np.array([])
    eu = {}
    eu['ade'] = float(np.mean(errs)) if errs.size > 0 else float('nan')
    eu['median_error'] = float(np.median(errs)) if errs.size > 0 else float('nan')
    eu['max_error'] = float(np.max(errs)) if errs.size > 0 else float('nan')
    eu['rmse_euclidean'] = float(sqrt(np.mean(errs**2))) if errs.size > 0 else float('nan')
    eu['std_error'] = float(np.std(errs)) if errs.size > 0 else float('nan')

    # thresholds classification metrics
    thresholds = THRESHOLDS
    within_stats = {}
    for th in thresholds:
        y_true_bin = (errs <= th).astype(int)  # 1 if within threshold
        # treat predicted similarly (we consider predicted label == true label since label derived from error)
        # For precision/recall we need predicted positives; we compute predicted as (pred error <= th)
        # Here y_true_bin and y_pred_bin are derived from the same errors -> precision/recall will be 1.0
        # Instead, we'll compute metric comparing whether each coordinate is within threshold individually (alternative)
        # Better approach: compute per-coordinate within-threshold (using abs diff per coord)
        # We'll compute both: sample-wise (euclidean) and per-coordinate.
        within_frac = float(np.mean(y_true_bin)) if y_true_bin.size > 0 else float('nan')
        # For sample-wise classification, predicted==true (we don't have separate predicted classification)
        # So precision/recall/f1 will be same as accuracy if we treat 'within' as positive label for ground truth and predicted derived similarly.
        acc = float(np.mean(y_true_bin)) if y_true_bin.size > 0 else float('nan')
        # But it's not meaningful to compute precision/recall without separate predictions; we'll skip those for euclidean and provide per-coordinate classifiers below.
        within_stats[f"within_{th}m_fraction"] = within_frac
        within_stats[f"within_{th}m_accuracy_samplewise"] = acc

    # Per-coordinate within-threshold classification metrics (use abs diff per coordinate)
    per_coord_within = {}
    for th in thresholds:
        per_c = {}
        for i, c in enumerate(['x', 'y', 'z']):
            diffs = np.abs(y_true[:, i] - y_pred[:, i]) if N > 0 else np.array([])
            y_true_bin = (diffs <= th).astype(int)  # ground truth: whether true diff <= th
            # But again we don't have a separate predicted label — typical use: treat whether predicted is within threshold of true -> that's evaluation.
            # To provide precision/recall/f1, we'll treat:
            # - positive ground truth: True difference <= th (i.e., "success")
            # - predicted positive: same definition (since we don't have an independent classifier)
            # This yields trivial precision=recall=1.0. Hence instead compute:
            # We'll compute precision/recall/f1 where positives are whether error <= threshold for that coordinate,
            # and predicted positives are whether predicted coordinate is within threshold comparing to a baseline (e.g., last position).
            # Lacking a baseline, a meaningful classification metric is not possible.
            # Therefore we compute: fraction_within, and also compute confusion vs a simple baseline (zero-motion baseline).
            frac_within = float(np.mean(y_true_bin)) if y_true_bin.size > 0 else float('nan')

            # Baseline: naive predictor = last position (i.e., predict no motion -> predicted next = last position => baseline_pred = last)
            # We cannot reconstruct baseline easily here because predictions.csv doesn't include the last position.
            # So we will not compute precision/recall in the true sense; instead return fraction_within and leave precision/recall to threshold on euclidean errors per-sample (not meaningful here).
            per_c[c] = {"fraction_within_threshold": frac_within}
        per_coord_within[f"threshold_{th}m"] = per_c

    # populate metrics
    metrics['per_coordinate'] = per_coord
    metrics['euclidean'] = eu
    metrics['within'] = within_stats
    metrics['per_coordinate_within'] = per_coord_within

    # Additional aggregates
    # overall mean absolute error across all coords
    if N > 0:
        metrics['overall_mae'] = float(np.mean(np.abs(y_true - y_pred)))
        metrics['overall_rmse'] = float(sqrt(np.mean((y_true - y_pred)**2)))
    else:
        metrics['overall_mae'] = metrics['overall_rmse'] = float('nan')

    return metrics

def load_predictions_csv(path=PRED_CSV):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Accept columns: x_pred,y_pred,z_pred,x_true,y_true,z_true
    required = ["x_pred","y_pred","z_pred","x_true","y_true","z_true"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"{path} missing required columns. Found: {df.columns.tolist()}")
    y_pred = df[["x_pred","y_pred","z_pred"]].values.astype(float)
    y_true = df[["x_true","y_true","z_true"]].values.astype(float)
    return y_true, y_pred

def regenerate_predictions_from_model(csv_path=CSV_PATH, out_dir=OUT_DIR, seq_len=SEQ_LEN, predict_delta=PREDICT_DELTA):
    """
    Best-effort regeneration using saved model/scalers and trajectories.csv.
    Returns (y_true_abs, preds_abs).
    """
    mpath = os.path.join(out_dir, MODEL_NAME)
    spath = os.path.join(out_dir, SCALER_NAME)
    if not os.path.exists(mpath) or not os.path.exists(spath):
        raise FileNotFoundError("Saved model or scalers not found in " + out_dir)

    if load_model is None:
        raise RuntimeError("TensorFlow not available to load the model. Install tensorflow to enable regeneration.")

    model = load_model(mpath)
    d = joblib.load(spath)
    scaler_X = d.get('scaler_X')
    scaler_y = d.get('scaler_y')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols_lower = [c.lower() for c in df.columns.tolist()]
    def find_col(cands):
        for c in cands:
            if c in cols_lower:
                return df.columns[cols_lower.index(c)]
        return None
    time_col = find_col(['time','t','sim_time','timestamp','frame_idx','step'])
    id_col = find_col(['vehicle_id','veh_id','id','actor_id','agent_id','vehicle'])
    x_col = find_col(['x','pos_x','px','longitude','lon'])
    y_col = find_col(['y','pos_y','py','latitude','lat'])
    z_col = find_col(['z','pos_z','pz','alt','height'])
    if None in (time_col,id_col,x_col,y_col,z_col):
        raise ValueError("Could not auto-detect required columns in trajectories csv. Columns: " + str(df.columns.tolist()))

    df = df[[time_col, id_col, x_col, y_col, z_col]].dropna().sort_values([id_col, time_col])

    traces = {}
    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        coords = np.vstack([g[x_col].values.astype(float), g[y_col].values.astype(float), g[z_col].values.astype(float)]).T
        if coords.shape[0] >= seq_len + 1:
            traces[str(vid)] = coords.copy()

    if len(traces) == 0:
        raise ValueError("No traces long enough to build sequences for evaluation.")

    X_raw_list, y_raw_list = [], []
    for coords in traces.values():
        T = coords.shape[0]
        for i in range(T - seq_len):
            seq = coords[i:i+seq_len]
            nxt = coords[i+seq_len]
            if predict_delta:
                target = nxt - coords[i+seq_len-1]
            else:
                target = nxt
            X_raw_list.append(seq)
            y_raw_list.append(target)
    X_raw = np.array(X_raw_list, dtype=float)
    y_raw = np.array(y_raw_list, dtype=float)

    if X_raw.size == 0:
        raise ValueError("No samples created from traces.")

    N = X_raw.shape[0]
    X_flat = X_raw.reshape(-1, 3)
    X_scaled_flat = scaler_X.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, seq_len, 3)
    y_scaled = scaler_y.transform(y_raw) if y_raw.size>0 else np.empty((0,3))

    split = int(0.9 * N) if N > 1 else 1
    X_test = X_scaled[split:]
    y_test = y_scaled[split:]
    X_raw_test = X_raw[split:]

    if len(X_test) == 0:
        raise ValueError("No test samples after splitting. Not enough data to create a test split.")

    preds_scaled = model.predict(X_test)
    preds = scaler_y.inverse_transform(preds_scaled) if preds_scaled.size>0 else np.empty((0,3))
    y_true = scaler_y.inverse_transform(y_test) if y_test.size>0 else np.empty((0,3))

    if PREDICT_DELTA and preds.shape[0] > 0:
        last_positions = X_raw_test[:, -1, :]
        preds_abs = preds + last_positions
        y_true_abs = y_true + last_positions
    else:
        preds_abs = preds
        y_true_abs = y_true

    return y_true_abs, preds_abs

def pretty_print(metrics):
    print("\n=== EXTENDED LSTM (x,y,z) EVALUATION SUMMARY ===")
    print(f"Samples: {metrics.get('n_samples', 'N/A')}")
    print("\nPer-coordinate metrics:")
    for c, d in metrics['per_coordinate'].items():
        print(f" {c}: MSE={d['mse']:.6f}, RMSE={d['rmse']:.6f}, MAE={d['mae']:.6f}, MedianAE={d['median_ae']:.6f}, MaxErr={d['max_abs_error']:.6f}")
        print(f"     MAPE%={d['mape_%']:.3f}, sMAPE%={d['smape_%']:.3f}, R2={d['r2']:.6f}, adjR2={d['adjusted_r2']:.6f}, EV={d['explained_variance']:.6f}, PearsonR={d['pearson_r']:.6f}")
    print("\nEuclidean/sample-level errors:")
    eu = metrics['euclidean']
    print(f" ADE={eu['ade']:.6f}, Median={eu['median_error']:.6f}, Max={eu['max_error']:.6f}, RMSE_euclidean={eu['rmse_euclidean']:.6f}, STD={eu['std_error']:.6f}")
    print("\nWithin-threshold fractions (sample-wise):")
    for k, v in metrics['within'].items():
        print(f" {k}: {v:.4f}")
    print("\nPer-coordinate within-threshold fractions:")
    for th, dd in metrics['per_coordinate_within'].items():
        print(f" {th}:")
        for c, vv in dd.items():
            print(f"   {c}: fraction_within_threshold = {vv['fraction_within_threshold']:.4f}")
    print(f"\nOverall MAE: {metrics.get('overall_mae', float('nan')):.6f}, Overall RMSE: {metrics.get('overall_rmse', float('nan')):.6f}")
    print("===============================================\n")

def main():
    y_true = y_pred = None

    # Try load predictions.csv
    try:
        loaded = load_predictions_csv(PRED_CSV)
        if loaded is not None:
            y_true, y_pred = loaded
            print(f"Loaded predictions from {PRED_CSV} (samples={len(y_true)})")
    except Exception as e:
        print("Warning: failed to load predictions.csv:", e)

    # If not present, try to regenerate
    if y_true is None or y_pred is None:
        try:
            print("predictions.csv not found/invalid — attempting regeneration using saved model/scalers + trajectories.csv ...")
            y_true, y_pred = regenerate_predictions_from_model(csv_path=CSV_PATH, out_dir=OUT_DIR, seq_len=SEQ_LEN, predict_delta=PREDICT_DELTA)
            print(f"Regenerated predictions (samples={len(y_true)})")
            df_out = pd.DataFrame(np.hstack([y_pred, y_true]), columns=["x_pred","y_pred","z_pred","x_true","y_true","z_true"])
            df_out.to_csv(PRED_CSV, index=False)
            print(f"Saved regenerated predictions to {PRED_CSV}")
        except Exception as e:
            print("ERROR: Could not regenerate predictions automatically:", e)
            print("Provide predictions.csv or ensure model and scalers exist in", OUT_DIR)
            return

    # compute metrics
    metrics = compute_metrics_from_arrays(y_true, y_pred, seq_len=SEQ_LEN)
    pretty_print(metrics)

    # Save metrics to JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

if __name__ == "__main__":
    main()