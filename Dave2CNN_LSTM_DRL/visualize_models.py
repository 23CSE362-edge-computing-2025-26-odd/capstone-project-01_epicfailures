# visualize_models.py
"""
Visualize model outputs locally.

Usage examples:
  python visualize_models.py --frames-root dataset/frames --feature-model cnn_feature_extractor.keras --lstm-model lstm_feature_predictor.keras

Optional:
  --csv path/to/frames_with_cumdist.csv   (must contain columns: frame_path, cumdist)
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import math
import pandas as pd

tf.get_logger().setLevel('ERROR')  # reduce TF logging

# ---------------- image utils ----------------
def load_image_as_array(path:Path, img_h:int, img_w:int):
    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB").resize((img_w, img_h), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
            return arr
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None

def find_frame_files(frames_root:Path):
    # collect files named frame_XXXXXX.png/jpg or any image files sorted by name
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")
    candidates = sorted([p for p in frames_root.rglob("frame_*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not candidates:
        # fallback to any images in dir
        candidates = sorted([p for p in frames_root.rglob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    return candidates

# ---------------- main visualizer ----------------
def main(args):
    frames_root = Path(args.frames_root)
    feat_model_path = Path(args.feature_model) if args.feature_model else None
    lstm_model_path = Path(args.lstm_model) if args.lstm_model else None
    out_path = Path(args.out)

    # load models if available
    feature_extractor = None
    lstm_model = None

    def try_load_model(p:Path):
        if p is None:
            return None
        if not p.exists():
            print(f"Model file not found: {p}")
            return None
        try:
            m = tf.keras.models.load_model(str(p))
            print(f"Loaded model: {p}  |  input shape: {m.input_shape}  output shape: {m.output_shape}")
            return m
        except Exception as e:
            print(f"Failed to load model {p}: {e}")
            return None

    # try two common extensions if user gave a base name without ext
    if feat_model_path and not feat_model_path.exists():
        for ext in (".keras", ".h5"):
            cand = feat_model_path.with_suffix(ext)
            if cand.exists():
                feat_model_path = cand
                break

    if lstm_model_path and not lstm_model_path.exists():
        for ext in (".keras", ".h5"):
            cand = lstm_model_path.with_suffix(ext)
            if cand.exists():
                lstm_model_path = cand
                break

    feature_extractor = try_load_model(feat_model_path) if feat_model_path else None
    lstm_model = try_load_model(lstm_model_path) if lstm_model_path else None

    # collect frames
    frames = find_frame_files(frames_root)
    if len(frames) == 0:
        print("No frames found under", frames_root)
        sys.exit(1)
    print("Found", len(frames), "frame images (showing first 200).")

    # optionally load CSV mapping frame -> cumdist
    truth_map = {}
    if args.csv:
        df = pd.read_csv(args.csv)
        if 'frame_path' in df.columns and 'cumdist' in df.columns:
            for _, r in df.iterrows():
                p = Path(r['frame_path'])
                # normalize to filename only to map when frames are in subfolders
                truth_map[str(p.name)] = float(r['cumdist'])
            print("Loaded truth mapping from CSV (frame_path -> cumdist).")
        else:
            print("CSV provided but missing required columns 'frame_path' and 'cumdist' - ignoring.")

    # limit frames to a reasonable window
    max_frames = min(len(frames), args.max_frames)
    frames = frames[:max_frames]

    # Precompute features for each frame (either using model or fallback scalar index)
    img_h, img_w = args.img_h, args.img_w
    features = []
    for p in frames:
        arr = load_image_as_array(p, img_h, img_w)
        if arr is None:
            arr = np.zeros((img_h, img_w, 3), dtype=np.float32)
        if feature_extractor is not None:
            # model includes normalization lambda (if built like earlier), so pass raw 0-255 float image
            try:
                f = feature_extractor.predict(np.expand_dims(arr, axis=0), verbose=0)
                f = f.reshape(-1)
            except Exception as e:
                print("Feature extractor predict failed for", p, ":", e)
                # fallback to simple scalar
                f = np.array([float(len(features))], dtype=np.float32)
        else:
            # fallback: scalar = frame index
            f = np.array([float(len(features))], dtype=np.float32)
        features.append(f.astype(np.float32))

    # unify dims
    maxdim = max([f.shape[0] for f in features]) if features else 1
    feat_array = np.zeros((len(features), maxdim), dtype=np.float32)
    for i,f in enumerate(features):
        feat_array[i,:f.shape[0]] = f

    # create sliding-window sequences and run LSTM if available
    seq_len = args.seq_len
    preds = []
    seq_indices = []
    for t in range(seq_len, len(frames)):
        Xseq = feat_array[t-seq_len:t]  # shape (seq_len, feat_dim)
        Xseq = np.expand_dims(Xseq, axis=0).astype(np.float32)
        if lstm_model is not None:
            try:
                pred = lstm_model.predict(Xseq, verbose=0)[0,0]
            except Exception as e:
                print("LSTM predict failed:", e)
                pred = float('nan')
        else:
            # fallback: use last scalar feature as "prediction"
            pred = float(np.mean(Xseq))
        preds.append(pred)
        seq_indices.append(t)

    # build truth array if available (map frames by filename)
    truths = []
    has_truth = False
    for idx in seq_indices:
        fname = frames[idx].name
        if fname in truth_map:
            truths.append(truth_map[fname])
            has_truth = True
        else:
            truths.append(np.nan)

    # Plot predictions (and truth if available)
    plt.figure(figsize=(10,5))
    x = np.arange(len(preds))
    plt.plot(x, preds, label='predicted', marker='o' if len(preds)<=80 else None)
    if has_truth and any(~np.isnan(truths)):
        plt.plot(x, truths, label='truth', marker='x' if len(truths)<=80 else None)
    plt.title("LSTM predictions (sliding window) on frames")
    plt.xlabel(f"frame index starting at t={seq_len}")
    plt.ylabel("predicted value (scaled or raw depending on model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_path))
    print("Saved plot to", out_path)
    # also show sample mapping (first 10)
    print("\nSample frames and preds (first 10):")
    for i in range(min(10, len(preds))):
        print(frames[seq_indices[i]].name, "-> pred:", preds[i], " truth:", (truths[i] if not math.isnan(truths[i]) else "n/a"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-model", default="cnn_feature_extractor.keras", help="Path to CNN feature extractor (.keras or .h5)")
    parser.add_argument("--lstm-model", default="lstm_feature_predictor.keras", help="Path to LSTM model (.keras or .h5)")
    parser.add_argument("--frames-root", default="dataset/frames", help="Folder with frames (frame_XXXXXX.png)")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--img-h", type=int, default=66)
    parser.add_argument("--img-w", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to scan (keeps runtime reasonable)")
    parser.add_argument("--csv", default=None, help="Optional CSV mapping frames to ground-truth cumdist (columns: frame_path,cumdist)")
    parser.add_argument("--out", default="lstm_preds_local.png", help="Output plot filename")
    args = parser.parse_args()
    main(args)
