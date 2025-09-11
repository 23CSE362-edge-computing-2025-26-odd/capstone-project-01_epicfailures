# evaluate_lstm.py
"""
Evaluate the trained LSTM on test data created from the same CSV.

Outputs:
 - prints scaled/unscaled MSE and RMSE
 - saves predictions_vs_true.csv (per-sample)
 - saves one example vehicle plot: veh_<id>_prediction.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from lstm_mobility import (
    load_traces_from_csv,
    build_dataset_from_traces,
    load_model_and_scaler,
    predict_next_pos
)

# ---------- Config ----------
CSV_PATH = "mobility_traces.csv"
SEQ_LEN = 8
PREDICT_DELTA = True   # must match how model was trained
MODEL_PATH = "lstm_mobility_predictor.keras"
SCALER_PATH = "scaler.pkl"
N_EXAMPLE_VEHS = 3     # number of vehicles to plot
PLOT_LEN = 120         # how many timesteps of the trace to plot (per vehicle)
OUT_CSV = "predictions_vs_true.csv"

# ---------- Load model & scalers ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found. Run training first.")

model, scalers = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
scaler_X, scaler_y = scalers

# ---------- Load traces and build dataset ----------
traces = load_traces_from_csv(CSV_PATH, seq_len=SEQ_LEN, use_cumulative=True)
X_raw, y_raw = build_dataset_from_traces(traces, seq_len=SEQ_LEN, predict_delta=PREDICT_DELTA)

if X_raw.size == 0:
    raise ValueError("No samples found - check seq_len and CSV")

# Scale X and y the same way as training code did
X_scaled = scaler_X.transform(X_raw.reshape(-1,1)).reshape(-1, SEQ_LEN, 1)
y_scaled = scaler_y.transform(y_raw.reshape(-1,1)).reshape(-1, 1)

# Split (same 90/10 split used in training)
split_idx = int(0.9 * len(X_scaled)) if len(X_scaled) > 1 else 1
X_test_scaled = X_scaled[split_idx:]
y_test_scaled = y_scaled[split_idx:]
X_test_raw = X_raw[split_idx:]
y_test_raw = y_raw[split_idx:]

# ---------- Predict on test set ----------
pred_scaled = model.predict(X_test_scaled, verbose=0)
# unscale preds and true values
pred_unscaled = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).reshape(-1)
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).reshape(-1)

# If model predicts deltas, convert preds to absolute next positions using last input value
if PREDICT_DELTA:
    last_inputs = X_test_raw[:, -1]   # last element of each input seq (raw scale)
    pred_abs = last_inputs + pred_unscaled
    true_abs = []
    # when y_raw was delta, the "true next absolute" = last_input + y_raw
    for i, last in enumerate(last_inputs):
        true_abs.append(last + y_test_unscaled[i])
    true_abs = np.array(true_abs)
else:
    pred_abs = pred_unscaled
    true_abs = y_test_unscaled

# Metrics
mse_scaled = float(np.mean((pred_scaled.reshape(-1) - y_test_scaled.reshape(-1))**2))
rmse_scaled = float(np.sqrt(mse_scaled))
mse_abs = float(np.mean((pred_abs - true_abs)**2))
rmse_abs = float(np.sqrt(mse_abs))

print("=== Evaluation results ===")
print(f"Num test samples: {len(pred_abs)}")
print(f"Scaled MSE: {mse_scaled:.6e}   RMSE: {rmse_scaled:.6e}")
print(f"Absolute MSE: {mse_abs:.6f}   RMSE: {rmse_abs:.6f}")

# ---------- Save per-sample CSV ----------
rows = []
# We need to recover vehicle id and sample idx for each sample.
# We'll reconstruct by iterating traces in the same order build_dataset_from_traces used.
veh_ids = []
sample_info = []  # (veh_id, sample_idx, last_input_val, true_next_abs, pred_next_abs)
count = 0
for vid, pos in traces.items():
    n = len(pos)
    for i in range(n - SEQ_LEN):
        if count >= split_idx:  # this sample is in test set
            idx_in_test = count - split_idx
            last_input = pos[i + SEQ_LEN - 1]
            if PREDICT_DELTA:
                true_next_abs = pos[i + SEQ_LEN]  # absolute next
                pred_next_abs = last_input + pred_unscaled[idx_in_test]
            else:
                true_next_abs = pos[i + SEQ_LEN]
                pred_next_abs = pred_unscaled[idx_in_test]
            sample_info.append((vid, i, float(last_input), float(true_next_abs), float(pred_next_abs)))
        count += 1

df_out = pd.DataFrame(sample_info, columns=["veh_id", "start_idx", "last_input", "true_next", "pred_next"])
df_out.to_csv(OUT_CSV, index=False)
print(f"Wrote per-sample predictions to {OUT_CSV}")

# ---------- Plot example vehicle traces ----------
import random
veh_list = list(traces.keys())
random.seed(42)
chosen = veh_list[:N_EXAMPLE_VEHS]  # deterministic; you can random.sample(veh_list, N_EXAMPLE_VEHS)

for vid in chosen:
    seq = traces[vid]
    L = min(len(seq)-1, PLOT_LEN)  # we need next value up to L
    true_trace = seq[:L]
    # Build model-predicted trace by sliding-window prediction
    preds = []
    for t in range(SEQ_LEN, L):
        raw_window = seq[t-SEQ_LEN:t]
        p = predict_next_pos(raw_window, model, scalers, SEQ_LEN, predict_delta=PREDICT_DELTA)
        preds.append(p)
    # Align arrays: true_nexts correspond to indices SEQ_LEN..L-1
    true_nexts = seq[SEQ_LEN:L]
    times = list(range(L))
    plt.figure(figsize=(8,4))
    plt.plot(times[:len(true_trace)], true_trace, label="True (input portion)")
    plt.plot(times[SEQ_LEN:SEQ_LEN+len(true_nexts)], true_nexts, label="True next")
    plt.plot(times[SEQ_LEN:SEQ_LEN+len(preds)], preds, linestyle="--", label="Predicted next")
    plt.xlabel("Timestep")
    plt.ylabel("Position (original scale)")
    plt.title(f"Vehicle {vid} prediction (first {L} timesteps)")
    plt.legend()
    fname = f"veh_{vid}_prediction.png"
    plt.grid(True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved plot for vehicle {vid} to {fname}")

print("Done evaluation.")
