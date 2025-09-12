"""
Combined: DAVE-2 style CNN -> LSTM mobility predictor -> PPO environment
Saves:
 - cnn_feature_extractor.h5
 - lstm_feature_predictor.h5
 - ppo_v2v_agent.zip
 - several PNG plots

Requirements:
pip install scikit-learn tensorflow stable-baselines3 gym pandas matplotlib pillow
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.layers import LSTM as KerasLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import gym
from gym import spaces
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- Config ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

SEQ_LEN = 8            # LSTM lookback
NUM_COV = 3
LSTM_EPOCHS = 10
CNN_EPOCHS = 8
PPO_TIMESTEPS = 2000   # increase for serious training
CSV_PATH = "mobility_traces_with_frames.csv"   # CSV must contain: time,veh_id,x,y,frame_path (frame_path optional)
FRAMES_ROOT = "frames"  # base folder for frame images if frame_path is relative
IMG_H, IMG_W = 66, 200  # DAVE-2 input dims
BATCH_SIZE = 64

# ---------------- Utility: load traces (positions) ----------------
def load_traces_from_csv(csv_path, min_len=None, time_col='time',
                         id_col='veh_id', x_col='x', y_col='y', frame_col='frame_path'):
    df = pd.read_csv(csv_path)
    df = df[[time_col, id_col, x_col, y_col] + ([frame_col] if frame_col in df.columns else [])].dropna(subset=[time_col, id_col, x_col, y_col])
    df = df.sort_values([id_col, time_col])
    traces = {}
    frame_map = {}  # {veh_id: [frame_path1, frame_path2, ...]}
    for vid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        xs = g[x_col].values.astype(float)
        ys = g[y_col].values.astype(float)
        deltas = np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2)
        seq = np.cumsum(deltas)  # use cumulative distance as scalar feature
        if len(seq) >= (min_len if min_len is not None else SEQ_LEN + 50):
            traces[str(vid)] = seq
            if frame_col in g.columns:
                frame_map[str(vid)] = g[frame_col].tolist()
            else:
                frame_map[str(vid)] = None
    if len(traces) == 0:
        raise ValueError("No usable traces found. Check CSV/min_len.")
    return traces, frame_map

traces, frame_map = load_traces_from_csv(CSV_PATH)
veh_ids = list(traces.keys())
if len(veh_ids) < 1 + NUM_COV:
    raise ValueError("Not enough vehicles in CSV for chosen NUM_COV.")
vu_id = veh_ids[0]
cov_ids = veh_ids[1:1+NUM_COV]
TOTAL_TIMESTEPS = len(next(iter(traces.values())))
print(f"Loaded {len(veh_ids)} traces, trace length {TOTAL_TIMESTEPS}. VU={vu_id}, CoVs={cov_ids}")

# ---------------- Utility: map trace index -> image path ----------------
def map_trace_to_frame(vid, idx):
    """
    Map a trace index t to a frame file path for vehicle vid.
    The CSV ideally has a frame_path per time step stored in frame_map.
    Otherwise try to find /frames/{vid}/frame_{idx:05d}.jpg
    """
    fmap = frame_map.get(str(vid))
    if fmap and idx < len(fmap) and fmap[idx] and isinstance(fmap[idx], str):
        p = fmap[idx]
        if not os.path.isabs(p):
            p = os.path.join(FRAMES_ROOT, p)
        if os.path.exists(p):
            return p
    # fallback patterns:
    candidate = os.path.join(FRAMES_ROOT, str(vid), f"frame_{idx:05d}.jpg")
    if os.path.exists(candidate):
        return candidate
    candidate_png = os.path.join(FRAMES_ROOT, str(vid), f"frame_{idx:05d}.png")
    if os.path.exists(candidate_png):
        return candidate_png
    return None

# ---------------- CNN model (DAVE-2 style simplified) ----------------
def make_dave2_cnn(input_shape=(IMG_H, IMG_W, 3), feature_dim=256):
    inp = Input(shape=input_shape)
    # Normalization (hard-coded)
    x = Lambda(lambda z: z/127.5 - 1.0)(inp)
    # Convolutions (simplified version of paper)
    x = Conv2D(24, (5,5), strides=(2,2), activation='elu')(x)   # out: ~31x98
    x = Conv2D(36, (5,5), strides=(2,2), activation='elu')(x)   # out: ~14x47
    x = Conv2D(48, (5,5), strides=(2,2), activation='elu')(x)   # out: ~5x22
    x = Conv2D(64, (3,3), activation='elu')(x)
    x = Conv2D(64, (3,3), activation='elu')(x)
    x = Flatten()(x)
    x = Dense(1164, activation='elu')(x)
    feat = Dense(feature_dim, activation='relu', name='features')(x)
    # final steering head (optional) - if you have steering labels
    steering = Dense(1, name='steering')(feat)
    model = Model(inputs=inp, outputs=[steering, feat])
    model.compile(optimizer='adam', loss={'steering':'mse'}, metrics=[])
    return model

cnn = make_dave2_cnn()
cnn.summary()

# ---------------- Prepare dataset for CNN pretraining ----------------
# We'll build (image, label) pairs. Label should be steering or next position.
# We'll use "next step cumulative position" as label if steering not available.
def safe_load_image(path):
    try:
        img = Image.open(path).convert('RGB').resize((IMG_W, IMG_H))
        return np.array(img)
    except Exception:
        return None

image_paths = []
labels = []   # next-position scalar (cumulative dist) scaled later
for vid in traces.keys():
    seq = traces[vid]
    # if frame mapping not available, we still can skip CNN training
    for t in range(SEQ_LEN, len(seq)-1):
        p = map_trace_to_frame(vid, t)
        if p:
            img = p
            image_paths.append(img)
            labels.append(float(seq[t+1]))  # predict next step cumulative distance (proxy)
if len(image_paths) == 0:
    print("No frame images found. Skipping CNN pretraining and using raw positions for LSTM.")
    cnn_trained = False
else:
    # load images (careful on memory) - we will load lazily in generator to avoid OOM
    cnn_trained = True

# Generator for training images
def image_generator(image_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    idxs = np.arange(len(image_paths))
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start+batch_size]
            Xb = []
            yb = []
            for i in batch_idx:
                p = image_paths[i]
                img = safe_load_image(p)
                if img is None:
                    continue
                Xb.append(img)
                yb.append(labels[i])
            if len(Xb) == 0:
                continue
            Xb = np.array(Xb).astype(np.float32)
            yb = np.array(yb).astype(np.float32).reshape(-1,1)
            yield Xb, {'steering': yb}

# scale labels
if cnn_trained:
    label_arr = np.array(labels).reshape(-1,1)
    scaler_pos = MinMaxScaler()
    scaler_pos.fit(label_arr)
    labels_scaled = scaler_pos.transform(label_arr).reshape(-1)
    # replace labels array for generator
    labels = labels_scaled.tolist()

# Train CNN (if we have images)
if cnn_trained:
    steps_per_epoch = max(1, len(image_paths)//BATCH_SIZE)
    val_split = 0.1
    val_count = int(len(image_paths)*val_split)
    train_paths = image_paths[val_count:]
    train_labels = labels[val_count:]
    val_paths = image_paths[:val_count]
    val_labels = labels[:val_count]

    train_gen = image_generator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = image_generator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    history = cnn.fit(train_gen,
                      steps_per_epoch=max(1,len(train_paths)//BATCH_SIZE),
                      validation_data=val_gen,
                      validation_steps=max(1,len(val_paths)//BATCH_SIZE),
                      epochs=CNN_EPOCHS)
    # Save
    cnn.save("cnn_feature_extractor.h5")
    print("Saved CNN to cnn_feature_extractor.h5")
    # plot loss (steering head)
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss', marker='o')
    plt.plot(history.history['val_loss'], label='val_loss', marker='s')
    plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.title('CNN training'); plt.legend(); plt.grid(True)
    plt.savefig("cnn_training.png")
    plt.show()

# ---------------- Create features dataset for LSTM ----------------
# For each veh trace, create sequence of feature vectors. If CNN unavailable, fall back to scalar positions as 1D features.
FEATURE_DIM = 256
def extract_feature_for_frame(frame_path):
    if not cnn_trained:
        return None
    img = safe_load_image(frame_path)
    if img is None:
        return None
    img = img.astype(np.float32)
    # run CNN to get features (use steering,features outputs)
    _, feat = cnn.predict(np.expand_dims(img, axis=0), verbose=0)
    return feat.flatten()

# Build feature sequences for each vehicle.
vehicle_features = {}  # {vid: [feat_t,...]} each feat is FEATURE_DIM vector OR scalar fallback
for vid in traces.keys():
    seq_len = len(traces[vid])
    feats = []
    mapped_frames = frame_map.get(str(vid))
    if mapped_frames is None or not cnn_trained:
        # fallback: use scalar cumulative position normalized
        arr = traces[vid].reshape(-1,1)
        scaler_v = MinMaxScaler()
        scaler_v.fit(arr)
        scaled = scaler_v.transform(arr).reshape(-1)
        # use 1D feature
        for v in scaled:
            feats.append(np.array([v], dtype=np.float32))
    else:
        # try to extract features for each available frame index
        for t in range(seq_len):
            p = map_trace_to_frame(vid, t)
            if p:
                f = extract_feature_for_frame(p)
                if f is None:
                    # fallback scalar
                    f = np.array([float(traces[vid][t])])
                feats.append(f)
            else:
                feats.append(np.array([float(traces[vid][t])]))
    vehicle_features[str(vid)] = np.stack(feats)

# If features are scalar to many vehicles, make sure LSTM input dims reflect shape
example_feat = next(iter(vehicle_features.values()))
if example_feat.ndim == 2:
    feat_dim = example_feat.shape[1]
else:
    feat_dim = 1

print("Feature dim:", feat_dim)

# Build dataset X (seq of feats), y (next-step scalar cumulative pos) for LSTM training
def build_feature_dataset(vehicle_features, traces, seq_len=SEQ_LEN):
    X, y = [], []
    for vid, feats in vehicle_features.items():
        seq = traces[vid]
        for i in range(len(seq) - seq_len - 1):
            X.append(feats[i:i+seq_len])
            y.append(float(seq[i+seq_len]))  # predicting next scalar cumulative pos
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32).reshape(-1,1)
    return X, y

X_raw, y_raw = build_feature_dataset(vehicle_features, traces, seq_len=SEQ_LEN)
print("Built LSTM dataset, shapes:", X_raw.shape, y_raw.shape)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

# scale X per feature if features are scalar; if features are CNN features, leave as-is or scale per-dim
if feat_dim == 1:
    X_flat = X_raw.reshape(-1,1)
    scaler_X = MinMaxScaler().fit(X_flat)
    X_scaled = scaler_X.transform(X_flat).reshape(X_raw.shape)
else:
    X_scaled = X_raw  # keeping CNN features as-is (already learned)

split = int(0.9 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# ---------------- LSTM predictor model (on features) ----------------
def make_feature_lstm(seq_len=SEQ_LEN, feat_dim=feat_dim):
    inp = Input(shape=(seq_len, feat_dim))
    x = KerasLSTM(128)(inp)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

lstm = make_feature_lstm()
lstm.summary()

history_l = lstm.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=64,
                     validation_split=0.1, verbose=1)
pred_test = lstm.predict(X_test)
mse = mean_squared_error(y_test, pred_test)
print(f"LSTM scaled MSE on test: {mse:.6f}")

plt.figure(figsize=(6,4))
plt.plot(history_l.history['loss'], marker='o', label='train')
plt.plot(history_l.history['val_loss'], marker='s', label='val')
plt.title("LSTM training"); plt.xlabel("Epochs"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
plt.savefig("lstm_training.png")
plt.show()

lstm.save("lstm_feature_predictor.h5")
print("Saved LSTM to lstm_feature_predictor.h5")

# Function to predict next scalar from raw feature sequence
def predict_next_from_features(raw_seq_feats):
    """
    raw_seq_feats: np array shape (SEQ_LEN, feat_dim)
    returns scalar predicted next cumulative pos (in original scale)
    """
    x = np.array(raw_seq_feats).astype(np.float32).reshape(1, SEQ_LEN, feat_dim)
    p_scaled = lstm.predict(x, verbose=0)[0,0]
    p = scaler_y.inverse_transform([[p_scaled]])[0,0]
    return float(p)

# ---------------- Gym environment using CNN->LSTM predictor ----------------
class V2VWithVisualPredictorEnv(gym.Env):
    """
    Observation: [VU current scalar pos, predicted CoV next-pos (per CoV), delays...]
    Prediction is produced by: extract features for last SEQ_LEN frames -> LSTM -> predicted scalar next pos
    Action: Discrete choice of CoV index to request
    """
    metadata = {'render.modes': []}
    def __init__(self, traces, frame_map, vu_id, cov_ids, seq_len=SEQ_LEN):
        super().__init__()
        self.traces = traces
        self.frame_map = frame_map
        self.vu_id = vu_id
        self.cov_ids = cov_ids
        self.num_cov = len(cov_ids)
        self.seq_len = seq_len
        self.t = seq_len
        self.max_t = len(next(iter(traces.values()))) - 2
        # obs: 1 (vu_pos) + num_cov (pred next pos) + num_cov (delays)
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(1 + 2*self.num_cov,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cov)
        self.base_delays = np.random.uniform(1.0, 3.0, size=self.num_cov)
        self.alpha = 0.08

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low = self.seq_len
        high = max(low+1, self.max_t - 100)
        self.t = np.random.randint(low, high)
        self.base_delays = np.random.uniform(1.0, 3.0, size=self.num_cov)
        return self._get_obs(), {}

    def _get_obs(self):
        vu_pos = float(self.traces[self.vu_id][self.t])
        predicted_list = []
        for cid in self.cov_ids:
            # construct feature seq for this cov for t-seq_len..t-1
            feats = []
            ff = frame_map.get(str(cid))
            if ff and cnn_trained:
                for k in range(self.t - self.seq_len, self.t):
                    p = map_trace_to_frame(cid, k)
                    if p:
                        feat = extract_feature_for_frame(p)
                        if feat is None:
                            feat = np.array([float(self.traces[cid][k])])
                    else:
                        feat = np.array([float(self.traces[cid][k])])
                    feats.append(feat)
            else:
                # use raw scalar positions normalized to match training
                for k in range(self.t - self.seq_len, self.t):
                    feats.append(np.array([float(self.traces[cid][k])]))
            feats = np.stack(feats)
            pred = predict_next_from_features(feats)
            predicted_list.append(pred)
        delays = self.base_delays.copy()
        obs = np.concatenate(([vu_pos], np.array(predicted_list), delays)).astype(np.float32)
        return obs

    def step(self, action):
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
        return obs, float(reward), terminated, truncated, info

# ---------------- Train PPO ----------------
env = V2VWithVisualPredictorEnv(traces, frame_map, vu_id, cov_ids, seq_len=SEQ_LEN)
model = PPO("MlpPolicy", env, verbose=1, seed=RANDOM_SEED)

print("Training PPO agent...")
reward_log = []
# Do several small learn loops with periodic eval like your previous code
chunks = 10
for i in range(chunks):
    model.learn(total_timesteps=PPO_TIMESTEPS//chunks, reset_num_timesteps=False)
    # quick eval
    obs, _ = env.reset()
    ep_reward = 0.0
    for step in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        if terminated or truncated:
            break
    reward_log.append(ep_reward)

model.save("ppo_v2v_agent")
print("Saved PPO agent to ppo_v2v_agent.zip")

plt.figure(figsize=(6,4))
plt.plot(range(1,len(reward_log)+1), reward_log, marker='o')
plt.xlabel("Evaluation Round"); plt.ylabel("Episode Reward"); plt.title("PPO reward trend"); plt.grid(True)
plt.savefig("ppo_reward_trend.png")
plt.show()

# ---------------- Plot sample prediction vs actual ----------------
sample_vid = vu_id
true_trace = traces[sample_vid][:80]
pred_trace = []
for i in range(SEQ_LEN, 80):
    seq_feats = vehicle_features[sample_vid][i-SEQ_LEN:i]
    p = predict_next_from_features(seq_feats)
    pred_trace.append(p)

plt.figure(figsize=(7,4))
plt.plot(range(80), true_trace, label='true')
plt.plot(range(SEQ_LEN,80), pred_trace, label='pred', linestyle='--')
plt.xlabel('Time step'); plt.ylabel('cumulative-pos'); plt.title('Pred vs True')
plt.legend(); plt.grid(True)
plt.savefig("prediction_vs_truth.png")
plt.show()

# ---------------- test learned policy ----------------
print("\n--- Testing learned policy ---")
for ep in range(2):
    obs, _ = env.reset()
    ep_reward = 0.0
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        if step % 10 == 0:
            print(f"ep {ep} step {step}: action {action}, reward {reward:.3f}, info {info}")
        if terminated or truncated:
            break
    print(f"Episode {ep} reward: {ep_reward:.3f}")

print("✅ All done. Models and plots saved.")
