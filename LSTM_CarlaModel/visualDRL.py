"""
Generate comprehensive matplotlib visualizations for eval_predictions.csv

Saves PNG files to ./plots/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# --- CONFIG ---
CSV_PATH = "eval_predictions.csv"   # <-- update path if needed
OUT_DIR = Path("./plots")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- STYLE SETUP (safe fallback) ---
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    print("Using seaborn whitegrid style")
except ImportError:
    plt.style.use("ggplot")
    print("Seaborn not found — using matplotlib 'ggplot' style")

# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH)

# convert to numeric if needed
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}")

# --- 1) Line trends ---
plt.figure(figsize=(12, 6))
for c in numeric_cols:
    plt.plot(df.index, df[c], label=c, alpha=0.8)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Trends of Variables over Index")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "trends_over_index.png", dpi=150)
plt.close()

# --- 2) Histograms ---
ncols = 3
nrows = int(np.ceil(len(numeric_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
axes = axes.flatten()
for ax, c in zip(axes, numeric_cols):
    ax.hist(df[c], bins=25, alpha=0.8)
    ax.set_title(c)
for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "histograms.png", dpi=150)
plt.close()

# --- 3) Boxplots ---
plt.figure(figsize=(max(6, len(numeric_cols)), 5))
plt.boxplot([df[c].values for c in numeric_cols], labels=numeric_cols, vert=False)
plt.title("Boxplots per Variable")
plt.tight_layout()
plt.savefig(OUT_DIR / "boxplots.png", dpi=150)
plt.close()

# --- 4) Reward vs each feature ---
if "reward" in numeric_cols:
    others = [c for c in numeric_cols if c != "reward"]
    cols_sc = 2
    rows_sc = int(np.ceil(len(others) / cols_sc))
    fig, axes = plt.subplots(rows_sc, cols_sc, figsize=(6*cols_sc, 4*rows_sc))
    axes = axes.flatten()
    for ax, c in zip(axes, others):
        ax.scatter(df[c], df["reward"], s=30, alpha=0.7)
        ax.set_xlabel(c)
        ax.set_ylabel("reward")
        ax.set_title(f"reward vs {c}")
    for ax in axes[len(others):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "reward_vs_features.png", dpi=150)
    plt.close()

# --- 5) Correlation heatmap (matplotlib) ---
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="left")
ax.set_yticklabels(numeric_cols)
for (i, j), val in np.ndenumerate(corr.values):
    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)
ax.set_title("Correlation Matrix", pad=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# --- 6) Scatter matrix (pairwise) ---
max_cols_for_matrix = 6
cols_for_matrix = numeric_cols[:max_cols_for_matrix]
scatter_matrix(df[cols_for_matrix], alpha=0.6,
               figsize=(3*len(cols_for_matrix), 3*len(cols_for_matrix)),
               diagonal="hist")
plt.suptitle(f"Scatter Matrix (first {len(cols_for_matrix)} numeric cols)")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_matrix.png", dpi=150)
plt.close()

# --- Save correlations summary ---
corr.to_csv(OUT_DIR / "correlations_summary.csv")

print(f"All plots saved in: {OUT_DIR.resolve()}")
