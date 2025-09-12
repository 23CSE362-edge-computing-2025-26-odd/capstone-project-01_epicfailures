"""
plot_lstm_predictions.py

Generates:
 - 2D scatter plot of x_pred vs y_pred with vehicle_id annotations
 - optional 3D scatter plot of x_pred, y_pred, z_pred

If a 'predictions.csv' file exists in the same folder it will be used
(expects columns: vehicle_id,target_time,x_pred,y_pred,z_pred,status).
Otherwise the script uses example data taken from your screenshot.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "predictions.csv"  # change if needed

# --- load data (use file if exists, fallback to inline sample) ---
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    sample = {
        "vehicle_id": [45, 46, 47, 48, 49, 50],
        "target_time": [2.0]*6,
        "x_pred": [
            105.98493957519533, -110.76255798339844, -106.68788146972656,
            -87.27456665039062, 3.0477840900421143, 32.992462158203125
        ],
        "y_pred": [
            62.44650268554688, 46.66009140014648, -4.847376823425293,
            24.441240310668945, 130.2106774902344, -67.90009307861328
        ],
        "z_pred": [
            0.0015032958472147, -0.0548222139477729, -0.0146558759734034,
            -0.0130630685016512, 0.1398258954286575, 0.0237394329160451
        ],
        "status": ["earliest_available"]*6
    }
    df = pd.DataFrame(sample)

# --- basic validation ---
required_cols = {"vehicle_id", "x_pred", "y_pred", "z_pred"}
if not required_cols.issubset(df.columns):
    raise SystemExit(f"CSV missing required columns. Need: {required_cols}")

# --- 2D scatter (x vs y) ---
def plot_2d(df, save_path="lstm_pred_2d.png", show=True):
    plt.figure(figsize=(10, 8))
    plt.scatter(df["x_pred"], df["y_pred"], marker="o", s=120)
    # annotate vehicle ids
    for i, vid in enumerate(df["vehicle_id"]):
        plt.annotate(
            str(vid),
            (df["x_pred"].iat[i], df["y_pred"].iat[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            weight="semibold"
        )

    plt.title(f"Predicted Vehicle Positions (LSTM) at target_time = {df['target_time'].iat[0]}s")
    plt.xlabel("x_pred")
    plt.ylabel("y_pred")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 2D plot to {save_path}")

# --- 3D scatter (optional) ---
def plot_3d(df, save_path="lstm_pred_3d.png", show=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["x_pred"], df["y_pred"], df["z_pred"], marker="o", s=80)

    # annotate points (project 3D points to 2D display coords)
    for i, vid in enumerate(df["vehicle_id"]):
        x, y, z = df["x_pred"].iat[i], df["y_pred"].iat[i], df["z_pred"].iat[i]
        ax.text(x, y, z, f" {vid}", size=9, zorder=1)

    ax.set_title(f"3D Predicted Positions (target_time = {df['target_time'].iat[0]}s)")
    ax.set_xlabel("x_pred")
    ax.set_ylabel("y_pred")
    ax.set_zlabel("z_pred")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 3D plot to {save_path}")

if __name__ == "__main__":
    plot_2d(df)
    # Uncomment the line below to also produce a 3D plot
    # plot_3d(df)
