# drl_select_cov_gymnasium.py
"""
Edge-Enabled Digital Twin Framework — DRL CoV selection (Gymnasium-compatible)

Changes vs original:
- Agent action space = pick one of NUM_COV candidate CoVs OR choose to offload to EDGE_DT (action index = NUM_COV).
- Observations augmented with per-CoV bandwidth (Mbps), per-CoV network latency (ms), and
  an edge-server "digital twin" predicted VU position plus a DT_fidelity estimate.
- Environment simulates communication delays, edge processing delays, DT staleness (fidelity),
  and bandwidth costs. Reward favors accurate perception (low distance from true VU next position),
  low end-to-end delay, higher trust, higher DT fidelity, and lower bandwidth usage.
- Keeps LSTM predictor & scalers for base VU prediction (used as baseline / local prediction).
- Saves PPO model and evaluation CSV.
"""

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

# ---------------- Config (edge/DT aware) ----------------
CSV_PATH = "trajectories.csv"           # must be present in same folder
LSTM_MODEL_DIR = "LSTM_MODEL"
LSTM_MODEL_NAME = "lstm_xyz_predictor.keras"
LSTM_SCALER_NAME = "scalers_xyz.pkl"

VU_ID = "45"         # primary vehicle id (string)
NUM_COV = 4          # number of candidate CoVs to choose from (fewer to keep action+edge small)
SEQ_LEN = 8
PPO_TIMESTEPS = 40000
RANDOM_SEED = 42

# Edge & DT simulation params
EDGE_PROCESSING_DELAY_RANGE = (0.05, 0.3)   # seconds (edge processing)
NETWORK_LATENCY_RANGE = (0.02, 0.2)        # seconds round-trip baseline per-CoV
BANDWIDTH_RANGE_Mbps = (1.0, 50.0)         # simulated available bandwidth per CoV
DT_STALENESS_MAX = 3                       # how many timesteps old DT input can be (integer)

# Reward coefficients (tunable)
W_PERCEPTION = 1.0        # weight for perception gain (accuracy)
W_COMM_DELAY = 0.12       # penalty per second of communication delay
W_PROC_DELAY = 0.08       # penalty per second of edge processing delay
W_TRUST = 0.25            # reward weight for trust score
W_DT_FID = 0.5            # reward weight for DT fidelity (higher is better)
W_BW_COST = 0.001         # penalty per Mbps used (encourages bandwidth-frugal choices)

# Misc output
OUT_PPO = "ppo_select_cov_edge_dt"
EVAL_CSV = "eval_predictions_edge_dt.csv"
VERBOSE = 1

# ---------------- Utilities ----------------
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
                            g[zcol].values.astype(float)]).T   # (T,3)
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
    arr = np.array(raw_seq, dtype=float)  # shape (seq_len,3)
    flat = arr.reshape(-1, 3)
    flat_scaled = scaler_X.transform(flat)
    inp = flat_scaled.reshape(1, arr.shape[0], 3)
    p_scaled = model.predict(inp, verbose=0)[0]   # shape (3,)
    p = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
    if predict_delta:
        return arr[-1] + p
    else:
        return p

# ---------------- Gymnasium Environment (Edge + Digital Twin aware) ----------------
class VuSelectEdgeDTEnv(gym.Env):
    """
    Observation vector:
      - vu_local_pred (3)            : local/LSTM predicted next VU position
      - cov_curr_positions (3*num_cov)
      - cov_network_latency (num_cov) : seconds (simulated)
      - cov_bandwidth (num_cov)       : Mbps
      - cov_trust (num_cov)
      - edge_dt_pred (3)              : edge digital twin's predicted VU next pos (may be stale)
      - edge_dt_fidelity (1)          : [0..1] higher=better
    Action:
      Discrete(num_cov + 1)  -> choose CoV_i (0..num_cov-1) OR choose EDGE_DT (num_cov)
    """

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

        # observation dimension calculation
        obs_dim = 3 + 3 * self.num_cov + self.num_cov + self.num_cov + self.num_cov + 3 + 1
        # breakdown: local_vu_pred(3) + cov_pos(3*num_cov) + cov_latency(num_cov) + cov_bw(num_cov) + cov_trust(num_cov) + edge_dt_pred(3) + edge_dt_fidelity(1)
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cov + 1)  # last action index = offload to edge DT

        # derive max_t across involved vehicles
        try:
            self.max_t = min(len(self.traces[self.vu_id]['coords']) - 2,
                             min(len(self.traces[c]['coords']) - 2 for c in self.cov_ids))
        except KeyError:
            raise KeyError("VU or CoV ids missing in traces")

        # initialize per-episode variables
        self.t = self.seq_len
        self._init_episode_globals()

    def _init_episode_globals(self):
        # base network latency (s), bandwidth (Mbps), trust per CoV
        self.base_latencies = np.random.uniform(NETWORK_LATENCY_RANGE[0], NETWORK_LATENCY_RANGE[1], size=self.num_cov)
        self.bandwidths = np.random.uniform(BANDWIDTH_RANGE_Mbps[0], BANDWIDTH_RANGE_Mbps[1], size=self.num_cov)
        self.trust_scores = np.random.uniform(0.4, 1.0, size=self.num_cov)
        # edge-server processing delay (s)
        self.edge_processing_delay = np.random.uniform(EDGE_PROCESSING_DELAY_RANGE[0], EDGE_PROCESSING_DELAY_RANGE[1])
        # DT staleness (how old the DT input is, in timesteps) for simulation; fidelity derived from it
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
        """
        Simulate edge digital twin prediction:
        - Use LSTM predictor but feed it a seq that may be STALE (older by edge_dt_staleness timesteps)
        - Increase staleness -> reduce fidelity (higher error)
        """
        vu_coords = self.traces[self.vu_id]['coords']
        t_for_dt = max(self.seq_len, self.t - self.edge_dt_staleness)  # the DT used data up to t - staleness
        seq = ensure_seq_of_length(vu_coords[:t_for_dt+1], self.seq_len)
        dt_pred = predict_next_pos_multi(seq, self.lstm_model, self.scaler_X, self.scaler_y, predict_delta=self.predict_delta)
        # compute a fidelity score based on how far dt_pred is from true next (less staleness -> higher fidelity)
        true_next = vu_coords[self.t + 1]
        err = float(np.linalg.norm(dt_pred - true_next))
        # transform err -> fidelity in (0,1], clamp
        fidelity = 1.0 / (1.0 + err)    # roughly maps smaller err -> fidelity close to 1
        # but also degrade with staleness linearly
        fidelity *= max(0.0, 1.0 - (self.edge_dt_staleness / max(1, DT_STALENESS_MAX)))
        return np.array(dt_pred), float(fidelity)

    def _get_obs(self):
        # local VU predicted next position (using current seq)
        vu_coords = self.traces[self.vu_id]['coords'][:self.t+1]
        seq = ensure_seq_of_length(vu_coords, self.seq_len)
        vu_local_pred = predict_next_pos_multi(seq, self.lstm_model, self.scaler_X, self.scaler_y, predict_delta=self.predict_delta)

        # CoV current positions, network latencies, bandwidths, trusts
        cov_curr = []
        latencies = []
        bws = []
        trusts = []
        for idx, cid in enumerate(self.cov_ids):
            coords = self.traces[cid]['coords']
            cov_curr.append(coords[self.t].tolist())
            # simulate jitter around base latency
            lat = max(0.0, self.base_latencies[idx] + np.random.normal(0.0, 0.02))
            latencies.append(lat)
            # bandwidth with small jitter
            bws.append(max(0.1, self.bandwidths[idx] + np.random.normal(0.0, 1.0)))
            trusts.append(self.trust_scores[idx])

        cov_curr_flat = np.array(cov_curr).reshape(-1)  # 3*num_cov

        # edge DT prediction + fidelity
        edge_dt_pred, edge_dt_fidelity = self._simulate_edge_dt_prediction()

        obs = np.concatenate([
            vu_local_pred,                         # 3
            cov_curr_flat,                         # 3*num_cov
            np.array(latencies),                   # num_cov
            np.array(bws),                         # num_cov
            np.array(trusts),                      # num_cov
            edge_dt_pred,                          # 3
            np.array([edge_dt_fidelity])           # 1
        ]).astype(np.float32)

        # cache relevant items for reward computation
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
        vu_true_next = cached['vu_true_next']        # (3,)
        cov_true_next = cached['cov_true_next']      # (num_cov,3)

        # simulate choice outcome
        if chosen_idx < self.num_cov:
            # picking a CoV: perceived next position = CoV's true next (they send their local estimate)
            perceived = cov_true_next[chosen_idx]
            comm_delay = float(cached['latencies'][chosen_idx])  # seconds
            edge_proc = 0.0
            bw_used = float(cached['bws'][chosen_idx]) * 0.05   # assume small fraction of available BW used (Mbps) — tunable
            trust = float(cached['trusts'][chosen_idx])
            source = f"cov_{self.cov_ids[chosen_idx]}"
        else:
            # offload to edge DT: perceived = edge_dt_pred but account for edge processing & network delay
            perceived = cached['edge_dt_pred']
            # end-to-end delay: we assume a round-trip to edge (average across covs or an edge latency base)
            network_rr = np.mean(self.base_latencies)  # baseline network RTT to edge
            comm_delay = float(max(0.0, network_rr + np.random.normal(0.0, 0.02)))
            edge_proc = float(self.edge_processing_delay)
            bw_used = 1.0 * 0.5  # some moderate bandwidth use (Mbps) for offload; tunable
            # trust is an aggregate of nearby trusts (we approximate)
            trust = float(np.mean(self.trust_scores))
            source = "edge_dt"

        # compute perception accuracy (distance between perceived and true VU next)
        dist = float(np.linalg.norm(perceived - vu_true_next))
        perception_gain = 1.0 / (1.0 + dist)

        # DT fidelity (if choosing edge) else compute a small fidelity-like bonus from trust
        dt_fidelity = float(cached['edge_dt_fidelity']) if chosen_idx == self.num_cov else 0.0

        # reward combining terms
        reward = (
            W_PERCEPTION * perception_gain
            - W_COMM_DELAY * comm_delay
            - W_PROC_DELAY * edge_proc
            + W_TRUST * trust
            + W_DT_FID * dt_fidelity
            - W_BW_COST * bw_used
        )

        # step time forward
        self.t += 1
        terminated = bool(self.t >= self.max_t)
        truncated = False

        # prepare next obs (Gymnasium prefers final obs present even if terminated)
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

# ---------------- Training & Eval ----------------
def build_envs_and_train(traces, vu_id, lstm_model, scaler_X, scaler_y, seq_len=SEQ_LEN, num_cov=NUM_COV):
    # choose candidate CoVs (other vehicles)
    all_vids = [v for v in traces.keys() if v != str(vu_id)]
    if len(all_vids) < num_cov:
        raise ValueError("Not enough candidate CoVs in dataset.")
    random.shuffle(all_vids)
    cov_ids = all_vids[:num_cov]

    def make_env():
        return VuSelectEdgeDTEnv(traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y, seq_len=seq_len, predict_delta=True)

    # vectorize (single env) for SB3
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
            # record
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

# ---------------- Main -----------------
def main():
    set_seed(RANDOM_SEED)
    print("Using Gymnasium + Stable-Baselines3 (PPO) — Edge-enabled DT environment.")
    # load traces
    traces = load_traces(CSV_PATH)
    if str(VU_ID) not in traces:
        raise SystemExit(f"VU id {VU_ID} not found in dataset. Available ids: {list(traces.keys())[:10]}")
    # load LSTM predictor + scalers
    lstm_model, scaler_X, scaler_y = load_lstm_predictor(LSTM_MODEL_DIR)
    print("Loaded LSTM predictor and scalers.")

    # train
    print("Starting PPO training to learn CoV/Edge offload selection policy...")
    model, cov_ids = build_envs_and_train(traces, VU_ID, lstm_model, scaler_X, scaler_y)
    print("PPO training complete. Saved policy to:", OUT_PPO + ".zip")

    # evaluate
    print("Evaluating policy for a few episodes...")
    evaluate_policy(model, traces, VU_ID, cov_ids, lstm_model, scaler_X, scaler_y, episodes=8)

if __name__ == "__main__":
    main()
