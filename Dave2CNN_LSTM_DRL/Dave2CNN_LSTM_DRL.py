# pipeline_combined.py
"""
Combined pipeline (updated):
 - fixes Gym/Gymnasium reset signature incompatibility
 - saves Keras native format (.keras) instead of legacy .h5
 - otherwise same pipeline: optional CNN pretrain -> feature extraction -> LSTM -> PPO
"""

import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Lambda, Conv2D, Flatten, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import gym
from gym import spaces
from stable_baselines3 import PPO

# ---------------- Default config ----------------
DEFAULT_IMG_H, DEFAULT_IMG_W = 66, 200
DEFAULT_SEQ_LEN = 8
DEFAULT_NUM_COV = 3
DEFAULT_BATCH = 64
DEFAULT_CNN_EPOCHS = 8
DEFAULT_LSTM_EPOCHS = 10
DEFAULT_PPO_STEPS = 2000
RANDOM_SEED = 42

# ---------------- Helpers ----------------
def set_seeds(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def find_frame_path(frames_root:Path, vid:str, idx:int) -> Optional[str]:
    patterns = [
        frames_root / str(vid) / f"frame_{idx:06d}.png",
        frames_root / str(vid) / f"frame_{idx:06d}.jpg",
        frames_root / f"frame_{idx:06d}.png",
        frames_root / f"frame_{idx:06d}.jpg",
        frames_root / str(vid) / f"frame_{idx:05d}.png",
        frames_root / str(vid) / f"frame_{idx:05d}.jpg",
    ]
    for p in patterns:
        if p.exists():
            return str(p)
    return None

def safe_load_image_rgb(path:str, target_size:Tuple[int,int]) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((target_size[1], target_size[0]), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)
    except Exception:
        return None

# ---------------- CSV / traces loader ----------------
def load_traces(csv_path:Path,
                time_col='sim_time',
                id_col='veh_id',
                x_col='x',
                y_col='y',
                frame_col='frame_path',
                min_len:Optional[int]=None,
                step_length:float=0.5,
                frames_root:Optional[Path]=None):
    if frames_root is None:
        frames_root = Path("dataset/frames")
    else:
        frames_root = Path(frames_root)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols_needed = [time_col, id_col, x_col, y_col]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    has_frame_col = frame_col in df.columns
    keep_cols = cols_needed + ([frame_col] if has_frame_col else [])
    df = df.loc[:, keep_cols].dropna(subset=[time_col, id_col, x_col, y_col])
    df = df.sort_values([id_col, time_col])

    traces: Dict[str, np.ndarray] = {}
    frame_map: Dict[str, List[Optional[str]]] = {}

    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        xs = g[x_col].values.astype(float)
        ys = g[y_col].values.astype(float)
        deltas = np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2)
        cumdist = np.cumsum(deltas)
        if min_len is not None and len(cumdist) < min_len:
            continue
        traces[str(vid)] = cumdist

        fmap = []
        if has_frame_col:
            for val in g[frame_col].tolist():
                if isinstance(val, str) and val.strip():
                    pth = Path(val)
                    if not pth.is_absolute():
                        pth = frames_root / pth
                    fmap.append(str(pth) if pth.exists() else None)
                else:
                    fmap.append(None)
        else:
            times = g[time_col].values.astype(float)
            for sim_t in times:
                idx = int(round(sim_t / step_length))
                cand = None
                for c in (
                    frames_root / f"frame_{idx:06d}.png",
                    frames_root / f"frame_{idx:06d}.jpg",
                    frames_root / str(vid) / f"frame_{idx:06d}.png",
                    frames_root / str(vid) / f"frame_{idx:06d}.jpg",
                    frames_root / str(vid) / f"frame_{idx:05d}.png",
                    frames_root / str(vid) / f"frame_{idx:05d}.jpg",
                ):
                    if c.exists():
                        cand = c
                        break
                fmap.append(str(cand) if cand is not None else None)
        frame_map[str(vid)] = fmap

    if not traces:
        raise RuntimeError("No traces loaded (check CSV/min_len).")
    return traces, frame_map

# ---------------- CNN split (extractor + steering model) ----------------
def build_dave2_split(img_h:int, img_w:int, feature_dim:int=256):
    inp = Input(shape=(img_h, img_w, 3), name='image_input')
    x = Lambda(lambda z: z / 127.5 - 1.0)(inp)
    x = Conv2D(24, (5,5), strides=(2,2), activation='elu', name='c1')(x)
    x = Conv2D(36, (5,5), strides=(2,2), activation='elu', name='c2')(x)
    x = Conv2D(48, (5,5), strides=(2,2), activation='elu', name='c3')(x)
    x = Conv2D(64, (3,3), activation='elu', name='c4')(x)
    x = Conv2D(64, (3,3), activation='elu', name='c5')(x)
    x = Flatten(name='flat')(x)
    x = Dense(1164, activation='elu', name='d1')(x)
    feat = Dense(feature_dim, activation='relu', name='features')(x)

    feature_extractor = Model(inputs=inp, outputs=feat, name='dave2_features')
    steering_out = Dense(1, name='steering')(feat)
    steering_model = Model(inputs=inp, outputs=steering_out, name='dave2_steering')
    steering_model.compile(optimizer='adam', loss='mse')
    return feature_extractor, steering_model

def extract_features_batch(feature_extractor:Model, image_paths:List[str], img_size:Tuple[int,int], batch_size:int=32):
    feats = []
    n = len(image_paths)
    for start in range(0, n, batch_size):
        batch = image_paths[start:start+batch_size]
        imgs = []
        for p in batch:
            arr = safe_load_image_rgb(p, img_size)
            if arr is None:
                arr = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
            imgs.append(arr)
        X = np.array(imgs).astype(np.float32)
        f = feature_extractor.predict(X, verbose=0)
        feats.append(f)
    if feats:
        return np.vstack(feats)
    fd = feature_extractor.output_shape[-1] if feature_extractor is not None else 256
    return np.zeros((0, fd), dtype=np.float32)

def build_vehicle_feature_sequences(traces:Dict[str,np.ndarray],
                                    frame_map:Dict[str,List[Optional[str]]],
                                    frames_root:Path,
                                    feature_extractor:Optional[Model],
                                    img_size:Tuple[int,int],
                                    feature_dim:int):
    vehicle_features = {}
    for vid, seq in traces.items():
        n = len(seq)
        feats_list = []
        fmap = frame_map.get(vid)
        for t in range(n):
            p = None
            if fmap and t < len(fmap) and fmap[t]:
                pth = fmap[t]
                if not os.path.isabs(pth):
                    pth = str(frames_root / pth)
                if os.path.exists(pth):
                    p = pth
            if p is None:
                p = find_frame_path(frames_root, vid, t)
            if (p is None) or (feature_extractor is None):
                feats_list.append(np.array([float(seq[t])], dtype=np.float32))
            else:
                feats_list.append(p)
        if feature_extractor is not None:
            path_indices = [i for i,val in enumerate(feats_list) if isinstance(val,str)]
            if path_indices:
                img_paths = [feats_list[i] for i in path_indices]
                extracted = extract_features_batch(feature_extractor, img_paths, img_size, batch_size=32)
                for idx_i, feat in zip(path_indices, extracted):
                    feats_list[idx_i] = feat.astype(np.float32)
        final_feats = []
        for v in feats_list:
            if isinstance(v, np.ndarray):
                final_feats.append(v)
            else:
                try:
                    final_feats.append(np.array([float(v)], dtype=np.float32))
                except Exception:
                    final_feats.append(np.array([0.0], dtype=np.float32))
        maxdim = max([f.shape[0] for f in final_feats]) if final_feats else feature_dim
        S = np.zeros((n, maxdim), dtype=np.float32)
        for i,f in enumerate(final_feats):
            S[i,:f.shape[0]] = f
        vehicle_features[vid] = S
    return vehicle_features

def build_lstm_dataset(vehicle_features:Dict[str,np.ndarray], traces:Dict[str,np.ndarray], seq_len:int):
    X_list, y_list = [], []
    vids = list(vehicle_features.keys())
    for vid in vids:
        feats = vehicle_features[vid]
        seq = traces[vid]
        n = len(seq)
        if n <= seq_len+1:
            continue
        for i in range(0, n - seq_len - 1):
            X_list.append(feats[i:i+seq_len])
            y_list.append(float(seq[i+seq_len]))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1,1)
    return X, y

def build_feature_lstm(seq_len:int, feat_dim:int):
    inp = Input(shape=(seq_len, feat_dim))
    x = layers.LSTM(128)(inp)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- PPO Env (fixed reset signature) ----------------
# Replace your existing V2VWithVisualPredictorEnv with this class

class V2VWithVisualPredictorEnv(gym.Env):
    """
    Env that uses LSTM predictor to compute predicted next positions for CoVs.

    Fixes included:
      - compute min trace length across (vu + covs) and set max_t accordingly
      - ensure reset() chooses t such that t+1 is valid for all vehicles
      - step() checks bounds before indexing and terminates safely instead of crashing
      - returns Gymnasium-style tuples: reset() -> (obs, info), step() -> (obs, reward, terminated, truncated, info)
    """
    metadata = {'render.modes': []}

    def __init__(self,
                 traces:Dict[str,np.ndarray],
                 vehicle_features:Dict[str,np.ndarray],
                 frame_map:Dict[str,List[Optional[str]]],
                 vu_id:str,
                 cov_ids:List[str],
                 lstm_model:Model,
                 seq_len:int,
                 scaler_y:Optional[MinMaxScaler]=None):
        super().__init__()
        self.traces = traces
        self.vehicle_features = vehicle_features
        self.frame_map = frame_map
        self.vu_id = str(vu_id)
        self.cov_ids = [str(c) for c in cov_ids]
        self.seq_len = int(seq_len)
        self.lstm = lstm_model
        self.scaler_y = scaler_y

        # Determine minimal common length across the vehicles actually used (VU + CoVs)
        used_ids = [self.vu_id] + self.cov_ids
        lengths = []
        for vid in used_ids:
            if vid not in self.traces:
                raise RuntimeError(f"Vehicle id {vid} not found in traces.")
            lengths.append(len(self.traces[vid]))
        self.min_trace_len = min(lengths)
        # max_t such that we can safely index t+1 and also extract seq_len history ending at t
        # we need t >= seq_len and t+1 < min_trace_len  => t <= min_trace_len - 2
        if self.min_trace_len <= self.seq_len + 1:
            raise RuntimeError(f"Not enough timesteps for seq_len={self.seq_len}. Shortest vehicle length={self.min_trace_len}")
        self.max_t = self.min_trace_len - 2

        self.t = self.seq_len

        self.num_cov = len(self.cov_ids)
        # Observation contains: [vu_pos, pred_cov_1..pred_cov_N, delays_1..N]
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(1 + 2*self.num_cov,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cov)
        self.base_delays = np.random.uniform(1.0, 3.0, size=self.num_cov)
        self.alpha = 0.08

    def reset(self, *, seed:Optional[int]=None, options:Optional[dict]=None):
        """
        Return (obs, info) to be compatible with Gymnasium-style reset
        """
        if seed is not None:
            np.random.seed(seed)
        low = self.seq_len
        high = self.max_t
        # safeguard: if low == high, just pick low
        if high < low:
            self.t = low
        else:
            self.t = int(np.random.randint(low, high + 1))
        # randomize delays each episode
        self.base_delays = np.random.uniform(1.0, 3.0, size=self.num_cov)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _predict_next_for(self, vid:str, t_idx:int) -> float:
        """
        Predict next (scaled) cumulative position for vehicle `vid` using LSTM.
        Expects `t_idx` to be >= seq_len and < min_trace_len.
        """
        feats = self.vehicle_features[vid][t_idx-self.seq_len:t_idx]  # (seq_len, feat_dim)
        x = np.expand_dims(feats, axis=0).astype(np.float32)
        scaled = self.lstm.predict(x, verbose=0)[0,0]
        if self.scaler_y is not None:
            val = self.scaler_y.inverse_transform([[scaled]])[0,0]
        else:
            val = float(scaled)
        return float(val)

    def _get_obs(self):
        """
        Build observation vector from current time self.t.
        """
        vu_pos = float(self.traces[self.vu_id][self.t])
        preds = []
        for cid in self.cov_ids:
            try:
                # if not enough history for this cid (defensive), fallback to true pos
                if self.t - self.seq_len < 0 or self.t >= len(self.traces[cid]):
                    p = float(self.traces[cid][self.t])
                else:
                    p = self._predict_next_for(cid, self.t)
            except Exception:
                p = float(self.traces[cid][min(self.t, len(self.traces[cid]) - 1)])
            preds.append(p)
        delays = self.base_delays.copy()
        obs = np.concatenate(([vu_pos], np.array(preds), delays)).astype(np.float32)
        return obs

    def step(self, action:int):
        """
        Perform action and return (obs, reward, terminated, truncated, info).
        Terminate safely if t+1 would be out of bounds for any used vehicle.
        """
        assert 0 <= action < self.num_cov, f"Invalid action {action}"
        # before incrementing, check whether t+1 is valid for all vehicles
        out_of_bounds = False
        for vid in [self.vu_id] + self.cov_ids:
            if (self.t + 1) >= len(self.traces[vid]):
                out_of_bounds = True
                break

        if out_of_bounds:
            # end episode gracefully
            terminated = True
            truncated = False
            reward = 0.0
            info = {'terminated_reason': 'trace_end'}
            obs = self._get_obs()
            return obs, float(reward), terminated, truncated, info

        # safe to access t+1 for all vehicles
        true_next_positions = [float(self.traces[cid][self.t+1]) for cid in self.cov_ids]
        vu_true_next = float(self.traces[self.vu_id][self.t+1])
        chosen_true_pos = true_next_positions[action]
        perception_gain = 1.0 / (1.0 + abs(vu_true_next - chosen_true_pos))
        delay = float(self.base_delays[action])
        reward = perception_gain - self.alpha * delay

        # advance time
        self.t += 1

        # determine termination after advance (if we hit max_t)
        terminated = (self.t >= self.max_t)
        truncated = False
        obs = self._get_obs()
        info = {'perception_gain': perception_gain, 'delay': delay}
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed:Optional[int]=None, options:Optional[dict]=None):
        """
        Return (obs, info) to be compatible with Gymnasium-style reset which
        Stable-Baselines3 may expect via its compatibility wrapper.
        """
        if seed is not None:
            np.random.seed(seed)
        low = self.seq_len
        high = max(low+1, self.max_t - 10)
        self.t = np.random.randint(low, high)
        self.base_delays = np.random.uniform(1.0, 3.0, size=self.num_cov)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _predict_next_for(self, vid:str, t_idx:int) -> float:
        feats = self.vehicle_features[vid][t_idx-self.seq_len:t_idx]
        x = np.expand_dims(feats, axis=0).astype(np.float32)
        scaled = self.lstm.predict(x, verbose=0)[0,0]
        if self.scaler_y is not None:
            val = self.scaler_y.inverse_transform([[scaled]])[0,0]
        else:
            val = float(scaled)
        return float(val)

    def _get_obs(self):
        vu_pos = float(self.traces[self.vu_id][self.t])
        preds = []
        for cid in self.cov_ids:
            try:
                p = self._predict_next_for(cid, self.t)
            except Exception:
                p = float(self.traces[cid][self.t])
            preds.append(p)
        delays = self.base_delays.copy()
        obs = np.concatenate(([vu_pos], np.array(preds), delays)).astype(np.float32)
        return obs

    def step(self, action:int):
        assert 0 <= action < self.num_cov
        true_next_positions = [float(self.traces[cid][self.t+1]) for cid in self.cov_ids]
        vu_true_next = float(self.traces[self.vu_id][self.t+1])
        chosen_true_pos = true_next_positions[action]
        perception_gain = 1.0 / (1.0 + abs(vu_true_next - chosen_true_pos))
        delay = float(self.base_delays[action])
        reward = perception_gain - self.alpha * delay
        self.t += 1
        terminated = self.t >= self.max_t
        truncated = False
        obs = self._get_obs()
        info = {'perception_gain': perception_gain, 'delay': delay}
        # Return (obs, reward, terminated, truncated, info) — Gymnasium-style step signature.
        return obs, float(reward), terminated, truncated, info

# ---------------- Main pipeline ----------------
def main(args):
    set_seeds(args.seed)
    csv_path = Path(args.csv)
    frames_root = Path(args.frames_root)
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    assert frames_root.exists(), f"Frames root not found: {frames_root}"

    print("Loading traces and frame map...")
    traces, frame_map = load_traces(csv_path,
                                    time_col=args.time_col,
                                    id_col=args.id_col,
                                    x_col=args.x_col,
                                    y_col=args.y_col,
                                    frame_col=args.frame_col,
                                    min_len=args.min_len,
                                    step_length=args.step_length,
                                    frames_root=frames_root)

    veh_ids = list(traces.keys())
    if len(veh_ids) < 1 + args.num_cov:
        raise RuntimeError("Not enough vehicles for specified NUM_COV.")
    vu_id = veh_ids[0]
    cov_ids = veh_ids[1:1+args.num_cov]
    print(f"VU={vu_id}; CoVs={cov_ids}; trace length={len(next(iter(traces.values())))}")

    feature_extractor = None
    steering_model = None
    if args.cnn_epochs > 0:
        print("Collecting image->label pairs for CNN pretraining...")
        image_paths = []
        labels = []
        for vid in traces.keys():
            seq = traces[vid]
            for t in range(args.seq_len, len(seq)-1):
                p = None
                fmap = frame_map.get(str(vid))
                if fmap and t < len(fmap) and fmap[t]:
                    pth = fmap[t]
                    if not os.path.isabs(pth):
                        pth = str(frames_root / pth)
                    if os.path.exists(pth):
                        p = pth
                if p is None:
                    p = find_frame_path(frames_root, vid, t)
                if p:
                    image_paths.append(p)
                    labels.append(float(seq[t+1]))
        if len(image_paths) == 0:
            print("No frame images found. Skipping CNN pretraining.")
        else:
            print(f"Found {len(image_paths)} images for CNN pretrain. Building CNN.")
            feature_extractor, steering_model = build_dave2_split(args.img_h, args.img_w, feature_dim=args.feature_dim)

            label_arr = np.array(labels).reshape(-1,1)
            scaler_pos = MinMaxScaler().fit(label_arr)
            labels_scaled = scaler_pos.transform(label_arr).reshape(-1)

            def image_gen(paths, labs, batch_size=args.batch):
                idxs = np.arange(len(paths))
                while True:
                    np.random.shuffle(idxs)
                    for start in range(0, len(idxs), batch_size):
                        batch = idxs[start:start+batch_size]
                        Xb, yb = [], []
                        for i in batch:
                            arr = safe_load_image_rgb(paths[i], (args.img_h, args.img_w))
                            if arr is None:
                                continue
                            Xb.append(arr)
                            yb.append(labs[i])
                        if len(Xb)==0:
                            continue
                        yield np.array(Xb).astype(np.float32), np.array(yb).reshape(-1,1)

            val_frac = 0.1
            val_count = max(1, int(len(image_paths)*val_frac))
            train_paths = image_paths[val_count:]
            train_labels = labels_scaled[val_count:]
            val_paths = image_paths[:val_count]
            val_labels = labels_scaled[:val_count]
            steps_per_epoch = max(1, len(train_paths)//args.batch)
            val_steps = max(1, len(val_paths)//args.batch)

            print("Training steering_model (CNN + head) ...")
            steering_model.fit(image_gen(train_paths, train_labels),
                               steps_per_epoch=steps_per_epoch,
                               validation_data=image_gen(val_paths, val_labels),
                               validation_steps=val_steps,
                               epochs=args.cnn_epochs)

            # Save using native Keras format .keras
            try:
                feature_extractor.save("cnn_feature_extractor.keras")
                steering_model.save("cnn_steering_model.keras")
                print("Saved feature_extractor -> cnn_feature_extractor.keras and steering_model -> cnn_steering_model.keras")
            except Exception as e:
                print("Warning: failed to save CNN models:", e)
            args._scaler_pos = scaler_pos

    print("Building vehicle feature sequences...")
    vehicle_features = build_vehicle_feature_sequences(traces, frame_map, frames_root, feature_extractor, (args.img_h,args.img_w), args.feature_dim)
    example = next(iter(vehicle_features.values()))
    feat_dim = example.shape[1]
    print("Feature dimension:", feat_dim)

    print("Building LSTM dataset...")
    X, y = build_lstm_dataset(vehicle_features, traces, seq_len=args.seq_len)
    if X.shape[0] == 0:
        raise RuntimeError("Empty LSTM dataset — insufficient data.")
    print("X,y shapes:", X.shape, y.shape)
    scaler_y = MinMaxScaler().fit(y)
    y_scaled = scaler_y.transform(y)
    if feat_dim == 1:
        X_flat = X.reshape(-1,1)
        scaler_X = MinMaxScaler().fit(X_flat)
        X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = X

    n = X_scaled.shape[0]
    split = int(0.9 * n)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    print("Building and training LSTM...")
    lstm_model = build_feature_lstm(args.seq_len, feat_dim)
    lstm_model.fit(X_train, y_train, epochs=args.lstm_epochs, batch_size=args.batch, validation_split=0.1, verbose=1)
    pred_test = lstm_model.predict(X_test)
    mse = mean_squared_error(y_test, pred_test)
    print("LSTM scaled MSE:", mse)
    try:
        lstm_model.save("lstm_feature_predictor.keras")
        print("Saved LSTM to lstm_feature_predictor.keras")
    except Exception as e:
        print("Warning: failed to save LSTM model:", e)

    print("Creating PPO environment...")
    env = V2VWithVisualPredictorEnv(traces, vehicle_features, frame_map, vu_id, cov_ids, lstm_model, seq_len=args.seq_len, scaler_y=scaler_y)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    print("Training PPO...")
    chunks = max(1, args.ppo_steps // 1000)
    for i in range(chunks):
        model.learn(total_timesteps=max(1, args.ppo_steps // chunks), reset_num_timesteps=False)
    try:
        model.save("ppo_v2v_agent")
        print("Saved PPO to ppo_v2v_agent (SB3 format)")
    except Exception as e:
        print("Warning: failed to save PPO model:", e)

    # quick eval
    logs = []
    for _ in range(5):
        obs, _info = env.reset()
        ep_reward = 0.0
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            ep_reward += reward
            if done:
                break
        logs.append(ep_reward)
    print("Eval rewards:", logs)

    # sample plot
    try:
        plt.figure(figsize=(6,4))
        plt.plot(pred_test[:100], label='pred (scaled)')
        plt.plot(y_test[:100], label='true (scaled)')
        plt.legend(); plt.grid(True); plt.title('LSTM preds vs true (sample)')
        plt.savefig("lstm_preds_sample.png")
    except Exception:
        pass

    print("Done. Outputs saved where possible (use .keras format for Keras models).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/traces.csv")
    parser.add_argument("--frames-root", default="dataset/frames")
    parser.add_argument("--img-h", type=int, default=DEFAULT_IMG_H)
    parser.add_argument("--img-w", type=int, default=DEFAULT_IMG_W)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, dest="seq_len")
    parser.add_argument("--num-cov", type=int, default=DEFAULT_NUM_COV, dest="num_cov")
    parser.add_argument("--feature-dim", type=int, default=256, dest="feature_dim")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--cnn-epochs", type=int, default=DEFAULT_CNN_EPOCHS, dest="cnn_epochs")
    parser.add_argument("--lstm-epochs", type=int, default=DEFAULT_LSTM_EPOCHS, dest="lstm_epochs")
    parser.add_argument("--ppo-steps", type=int, default=DEFAULT_PPO_STEPS, dest="ppo_steps")
    parser.add_argument("--min-len", type=int, default=None, dest="min_len")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--time-col", default="sim_time", dest="time_col")
    parser.add_argument("--id-col", default="veh_id", dest="id_col")
    parser.add_argument("--x-col", default="x", dest="x_col")
    parser.add_argument("--y-col", default="y", dest="y_col")
    parser.add_argument("--frame-col", default="frame_path", dest="frame_col")
    parser.add_argument("--step-length", type=float, default=0.5, dest="step_length")
    args = parser.parse_args()
    main(args)
