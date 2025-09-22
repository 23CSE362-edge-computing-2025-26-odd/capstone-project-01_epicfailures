import os
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

import gymnasium as gym
from gymnasium import spaces

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

CSV_PATH = "trajectories.csv"
LSTM_MODEL_DIR = "LSTM_MODEL"
LSTM_MODEL_NAME = "lstm_xyz_predictor.keras"
LSTM_SCALER_NAME = "scalers_xyz.pkl"

VU_ID = "45"
NUM_COV = 4
SEQ_LEN = 8
PPO_TIMESTEPS = 40000
RANDOM_SEED = 42

EDGE_PROCESSING_DELAY_RANGE = (0.05, 0.3)
NETWORK_LATENCY_RANGE = (0.02, 0.2)
BANDWIDTH_RANGE_Mbps = (1.0, 50.0)
DT_STALENESS_MAX = 3

W_PERCEPTION = 1.0
W_COMM_DELAY = 0.12
W_PROC_DELAY = 0.08
W_TRUST = 0.25
W_DT_FID = 0.5
W_BW_COST = 0.001

OUT_PPO = "ppo_select_cov_edge_dt"
EVAL_CSV = "eval_predictions_edge_dt.csv"
VERBOSE = 1

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

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

def load_traces(csv_path):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found. Put your trajectories CSV in the script folder.")
    df = pd.read_csv(csv_path)
    mapping = auto_detect_cols(df)
    required = ['time','id','x','y','z']
    if any(k not in mapping for k in required):
        raise ValueError(f"Could not detect required columns automatically. Detected: {mapping}. Please ensure CSV has time,id,x,y,z columns.")
    tcol = mapping['time']; idcol = mapping['id']; xcol = mapping['x']; ycol = mapping['y']; zcol = mapping['z']
    df = df[[tcol, idcol, xcol, ycol, zcol]].dropna()
    df = df.sort_values([idcol, tcol])
    traces = {}
    for vid, g in df.groupby(idcol):
        g = g.sort_values(tcol)
        coords = np.vstack([g[xcol].values.astype(float),
                            g[ycol].values.astype(float),
                            g[zcol].values.astype(float)]).T
        times = g[tcol].values.astype(float)
        traces[str(vid)] = {"coords": coords, "times": times}
    return traces

def load_lstm_predictor(model_dir=LSTM_MODEL_DIR):
    mpath = os.path.join(model_dir, LSTM_MODEL_NAME)
    spath = os.path.join(model_dir, LSTM_SCALER_NAME)
    if not os.path.exists(mpath) or not os.path.exists(spath):
        raise FileNotFoundError(f"LSTM model or scalers not found in {model_dir}. Train LSTM first.")
    model = load_model(mpath)
    d = joblib.load(spath)
    scaler_X = d['scaler_X']
    scaler_y = d['scaler_y']
    return model, scaler_X, scaler_y

def ensure_seq_of_length(seq_coords, seq_len):
    L = seq_coords.shape[0]
    if L >= seq_len:
        return seq_coords[-seq_len:].copy()
    pad = np.repeat(seq_coords[0:1], repeats=(seq_len - L), axis=0)
    return np.vstack([pad, seq_coords]).astype(float)

def predict_next_pos_multi(raw_seq, model, scaler_X, scaler_y, predict_delta=True):
    arr = np.array(raw_seq, dtype=float)
    flat = arr.reshape(-1, 3)
    flat_scaled = scaler_X.transform(flat)
    inp = flat_scaled.reshape(1, arr.shape[0], 3)
    p_scaled = model.predict(inp, verbose=0)[0]
    p = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
    if predict_delta:
        return arr[-1] + p
    else:
        return p

class VuSelectEdgeDTEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y,
                 seq_len=SEQ_LEN, predict_delta=True):
        super().__init__()
        self.traces = traces
        self.vu_id = str(vu_id)
        self.cov_ids = [str(c) for c in cov_ids]
        self.num_cov = len(self.cov_ids)
        self.seq_len = seq_len
        self.predict_delta = predict_delta
        self.lstm_model = lstm_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        obs_dim = 3 + 3 * self.num_cov + self.num_cov + self.num_cov + self.num_cov + 3 + 1
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cov + 1)

        try:
            self.max_t = min(len(self.traces[self.vu_id]['coords']) - 2,
                             min(len(self.traces[c]['coords']) - 2 for c in self.cov_ids))
        except KeyError:
            raise KeyError("VU or CoV ids missing in traces")

        self.t = self.seq_len
        self._init_episode_globals()

    def _init_episode_globals(self):
        self.base_latencies = np.random.uniform(NETWORK_LATENCY_RANGE[0], NETWORK_LATENCY_RANGE[1], size=self.num_cov)
        self.bandwidths = np.random.uniform(BANDWIDTH_RANGE_Mbps[0], BANDWIDTH_RANGE_Mbps[1], size=self.num_cov)
        self.trust_scores = np.random.uniform(0.4, 1.0, size=self.num_cov)
        self.edge_processing_delay = np.random.uniform(EDGE_PROCESSING_DELAY_RANGE[0], EDGE_PROCESSING_DELAY_RANGE[1])
        self.edge_dt_staleness = np.random.randint(0, DT_STALENESS_MAX + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        low = self.seq_len
        high = max(low + 1, self.max_t)
        self.t = np.random.randint(low, high)
        self._init_episode_globals()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _simulate_edge_dt_prediction(self):
        vu_coords = self.traces[self.vu_id]['coords']
        t_for_dt = max(self.seq_len, self.t - self.edge_dt_staleness)
        seq = ensure_seq_of_length(vu_coords[:t_for_dt+1], self.seq_len)
        dt_pred = predict_next_pos_multi(seq, self.lstm_model, self.scaler_X, self.scaler_y, predict_delta=self.predict_delta)
        true_next = vu_coords[self.t + 1]
        err = float(np.linalg.norm(dt_pred - true_next))
        fidelity = 1.0 / (1.0 + err)
        fidelity *= max(0.0, 1.0 - (self.edge_dt_staleness / max(1, DT_STALENESS_MAX)))
        return np.array(dt_pred), float(fidelity)

    def _get_obs(self):
        vu_coords = self.traces[self.vu_id]['coords'][:self.t+1]
        seq = ensure_seq_of_length(vu_coords, self.seq_len)
        vu_local_pred = predict_next_pos_multi(seq, self.lstm_model, self.scaler_X, self.scaler_y, predict_delta=self.predict_delta)

        cov_curr = []
        latencies = []
        bws = []
        trusts = []
        for idx, cid in enumerate(self.cov_ids):
            coords = self.traces[cid]['coords']
            cov_curr.append(coords[self.t].tolist())
            lat = max(0.0, self.base_latencies[idx] + np.random.normal(0.0, 0.02))
            latencies.append(lat)
            bws.append(max(0.1, self.bandwidths[idx] + np.random.normal(0.0, 1.0)))
            trusts.append(self.trust_scores[idx])

        cov_curr_flat = np.array(cov_curr).reshape(-1)

        edge_dt_pred, edge_dt_fidelity = self._simulate_edge_dt_prediction()

        obs = np.concatenate([
            vu_local_pred,
            cov_curr_flat,
            np.array(latencies),
            np.array(bws),
            np.array(trusts),
            edge_dt_pred,
            np.array([edge_dt_fidelity])
        ]).astype(np.float32)

        self._cached = {
            'vu_local_pred': np.array(vu_local_pred),
            'cov_true_next': np.array([self.traces[c]['coords'][self.t + 1] for c in self.cov_ids]),
            'vu_true_next': np.array(self.traces[self.vu_id]['coords'][self.t + 1]),
            'latencies': np.array(latencies),
            'bws': np.array(bws),
            'trusts': np.array(trusts),
            'edge_dt_pred': np.array(edge_dt_pred),
            'edge_dt_fidelity': float(edge_dt_fidelity)
        }

        return obs

    def step(self, action):
        assert 0 <= action < (self.num_cov + 1)
        chosen_idx = int(action)
        cached = self._cached
        vu_true_next = cached['vu_true_next']
        cov_true_next = cached['cov_true_next']

        if chosen_idx < self.num_cov:
            perceived = cov_true_next[chosen_idx]
            comm_delay = float(cached['latencies'][chosen_idx])
            edge_proc = 0.0
            bw_used = float(cached['bws'][chosen_idx]) * 0.05
            trust = float(cached['trusts'][chosen_idx])
            source = f"cov_{self.cov_ids[chosen_idx]}"
        else:
            perceived = cached['edge_dt_pred']
            network_rr = np.mean(self.base_latencies)
            comm_delay = float(max(0.0, network_rr + np.random.normal(0.0, 0.02)))
            edge_proc = float(self.edge_processing_delay)
            bw_used = 1.0 * 0.5
            trust = float(np.mean(self.trust_scores))
            source = "edge_dt"

        dist = float(np.linalg.norm(perceived - vu_true_next))
        perception_gain = 1.0 / (1.0 + dist)

        dt_fidelity = float(cached['edge_dt_fidelity']) if chosen_idx == self.num_cov else 0.0

        reward = (
            W_PERCEPTION * perception_gain
            - W_COMM_DELAY * comm_delay
            - W_PROC_DELAY * edge_proc
            + W_TRUST * trust
            + W_DT_FID * dt_fidelity
            - W_BW_COST * bw_used
        )

        self.t += 1
        terminated = bool(self.t >= self.max_t)
        truncated = False

        obs = self._get_obs() if not terminated else self._get_obs()

        info = {
            'source': source,
            'distance': dist,
            'perception_gain': perception_gain,
            'comm_delay': comm_delay,
            'edge_proc_delay': edge_proc,
            'bw_used_mbps': bw_used,
            'trust': trust,
            'dt_fidelity': dt_fidelity
        }
        return obs, float(reward), terminated, truncated, info

def build_envs_and_train(traces, vu_id, lstm_model, scaler_X, scaler_y, seq_len=SEQ_LEN, num_cov=NUM_COV):
    all_vids = [v for v in traces.keys() if v != str(vu_id)]
    if len(all_vids) < num_cov:
        raise ValueError("Not enough candidate CoVs in dataset.")
    random.shuffle(all_vids)
    cov_ids = all_vids[:num_cov]

    def make_env():
        return VuSelectEdgeDTEnv(traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y, seq_len=seq_len, predict_delta=True)

    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=VERBOSE, seed=RANDOM_SEED)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    model.save(OUT_PPO)
    return model, cov_ids

def evaluate_policy(model, traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y, episodes=8):
    env = VuSelectEdgeDTEnv(traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=RANDOM_SEED + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            results.append({
                'chosen_source': info['source'],
                'distance': info['distance'],
                'perception_gain': info['perception_gain'],
                'comm_delay': info['comm_delay'],
                'edge_proc_delay': info['edge_proc_delay'],
                'bw_used_mbps': info['bw_used_mbps'],
                'trust': info['trust'],
                'dt_fidelity': info['dt_fidelity'],
                'reward': reward
            })
            done = terminated or truncated
    df = pd.DataFrame(results)
    df.to_csv(EVAL_CSV, index=False)

    sep = "=" * 80
    print("\n" + sep)
    print("EVAL RESULTS (CSV rows only) — saved to:", EVAL_CSV)
    print("chosen_source,distance,perception_gain,comm_delay,edge_proc_delay,bw_used_mbps,trust,dt_fidelity,reward")
    for _, r in df.iterrows():
        print(f"{r['chosen_source']},{r['distance']:.6f},{r['perception_gain']:.6f},{r['comm_delay']:.6f},{r['edge_proc_delay']:.6f},{r['bw_used_mbps']:.6f},{r['trust']:.6f},{r['dt_fidelity']:.6f},{r['reward']:.6f}")
    print(sep + "\n")
    print("Mean reward:", df['reward'].mean())
    print("Mean distance:", df['distance'].mean())
    print("Counts by source:\n", df['chosen_source'].value_counts())

def main():
    set_seed(RANDOM_SEED)
    print("Using Gymnasium + Stable-Baselines3 (PPO) — Edge-enabled DT environment.")
    traces = load_traces(CSV_PATH)
    if str(VU_ID) not in traces:
        raise SystemExit(f"VU id {VU_ID} not found in dataset. Available ids: {list(traces.keys())[:10]}")
    lstm_model, scaler_X, scaler_y = load_lstm_predictor(LSTM_MODEL_DIR)
    print("Loaded LSTM predictor and scalers.")

    print("Starting PPO training to learn CoV/Edge offload selection policy...")
    model, cov_ids = build_envs_and_train(traces, VU_ID, lstm_model, scaler_X, scaler_y)
    print("PPO training complete. Saved policy to:", OUT_PPO + ".zip")

    print("Evaluating policy for a few episodes...")
    evaluate_policy(model, traces, VU_ID, cov_ids, lstm_model, scaler_X, scaler_y, episodes=8)

if __name__ == "__main__":
    main()
