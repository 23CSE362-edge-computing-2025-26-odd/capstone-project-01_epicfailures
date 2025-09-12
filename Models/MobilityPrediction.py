# ===============================
# Mobility Prediction - Baseline + LSTM
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----- Generate synthetic mobility data (Replace with SUMO traces later) -----
# Columns: [time, x_position]
time = np.arange(0, 200, 1)  # 200 timesteps
x_position = 0.5 * time + 5 * np.sin(0.2 * time)  # linear + oscillation

data = pd.DataFrame({"time": time, "x": x_position})

# ----- Baseline Linear Regression -----
X = data["time"].values.reshape(-1, 1)
y = data["x"].values

model_lr = LinearRegression()
model_lr.fit(X, y)
y_pred_lr = model_lr.predict(X)

print("Baseline Linear Regression MSE:", mean_squared_error(y, y_pred_lr))

# ----- LSTM Prediction -----
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(y.reshape(-1, 1))

seq_len = 10  # lookback window
X_seq, y_seq = [], []
for i in range(len(scaled_x) - seq_len):
    X_seq.append(scaled_x[i:i+seq_len])
    y_seq.append(scaled_x[i+seq_len])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train/Test Split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM Model
model_lstm = Sequential([
    LSTM(64, input_shape=(seq_len, 1), return_sequences=False),
    Dense(1)
])
model_lstm.compile(optimizer="adam", loss="mse")

history = model_lstm.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# Predictions
y_pred_lstm = model_lstm.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_rescaled = scaler.inverse_transform(y_test)

print("LSTM Test MSE:", mean_squared_error(y_test_rescaled, y_pred_lstm))

# Plot
plt.figure(figsize=(10,5))
plt.plot(time[-len(y_test):], y_test_rescaled, label="True Trajectory")
plt.plot(time[-len(y_test):], y_pred_lstm, label="LSTM Prediction")
plt.legend()
plt.title("Mobility Prediction (LSTM)")
plt.show()
