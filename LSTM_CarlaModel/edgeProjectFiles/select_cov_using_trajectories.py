#!/usr/bin/env python3
"""
select_cov_using_trajectories.py

Variant that always returns a CoV (never "edge_dt").
Uses LSTM to predict VU position, builds features for CoVs,
runs PPO policy if available, but falls back to a deterministic scorer
so a CoV is ALWAYS returned (when any CoV exists).
"""
import os, argparse, numpy as np, pandas as pd, joblib, math
import torch, torch.nn as nn
from tensorflow.keras.models import load_model

FEATURES_PER_COV = 4  # [bandwidth, latency, trust, distance]

# --- SB3-compatible module (same as before) ---
class SB3PolicyCompat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_net = nn.Linear(hidden_dim, output_dim)
        self.value_net = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        latent_pi = self.mlp_extractor.policy_net(x)
        logits = self.action_net(latent_pi)
        return logits

def try_load_sb3_pth(pth_path):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f".pth file not found: {pth_path}")
    state = torch.load(pth_path, map_location='cpu')
    if not isinstance(state, dict):
        raise RuntimeError("Loaded .pth is not a dict-like state_dict.")
    keys = sorted(state.keys())
    def find_key_ending(suffixes):
        for k in keys:
            for suf in suffixes:
                if k.endswith(suf):
                    return k
        return None
    k_policy_w0 = find_key_ending(["mlp_extractor.policy_net.0.weight", ".policy_net.0.weight"])
    k_action_w = find_key_ending(["action_net.weight", ".action_net.weight", "action_net.0.weight"])
    if k_policy_w0 is None or k_action_w is None:
        raise RuntimeError("SB3 keys not found in state_dict. Keys sample: " + ", ".join(keys[:40]))
    w0 = state[k_policy_w0]; act_w = state[k_action_w]
    hidden_dim = int(w0.shape[0]); input_dim = int(w0.shape[1]); output_dim = int(act_w.shape[0])
    policy = SB3PolicyCompat(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    policy.load_state_dict(state, strict=False)
    policy.eval()
    def forward_fn(obs_tensor):
        if obs_tensor.shape[1] != input_dim:
            raise ValueError(f"forward_fn expected input dim {input_dim}, got {obs_tensor.shape}")
        with torch.no_grad():
            logits = policy(obs_tensor)
            actions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        return actions
    return forward_fn, input_dim, output_dim

# ---------------- deterministic scorer ----------------
def deterministic_score_matrix(feats):
    """
    feats: (N,4) array: [bw,lat,trust,dist]
    Returns score array length N (higher is better).
    Normalize each feature column to [0,1] then compute weighted score.
    We want higher bw -> better, lower latency -> better, higher trust -> better, lower dist -> better.
    """
    if feats.shape[0] == 0:
        return np.array([], dtype=np.float32)
    # safe numeric conversions
    feats = feats.astype(np.float32)
    bw = feats[:,0]; lat = feats[:,1]; trust = feats[:,2]; dist = feats[:,3]
    # normalize robustly to [0,1]
    def norm(x):
        mn = np.nanmin(x); mx = np.nanmax(x)
        if mx - mn < 1e-6:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    bw_n = norm(bw)
    lat_n = norm(lat)  # higher latency -> worse
    trust_n = norm(trust)
    dist_n = norm(dist)
    # weights (tunable)
    w_bw = 0.25
    w_trust = 0.35
    w_invdist = 0.30
    w_lat = 0.10
    # compute score
    score = w_bw * bw_n + w_trust * trust_n + w_invdist * (1.0 - dist_n) - w_lat * lat_n
    return score

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("run_output","trajectories.csv"))
    parser.add_argument("--lstm", default=os.path.join("run_output","lstm_xyz_predictor.keras"))
    parser.add_argument("--scaler", default=os.path.join("run_output","scalers_xyz.pkl"))
    parser.add_argument("--pth", default="ppo_policy_only.pt")
    parser.add_argument("--vu-id", type=int, default=None, help="runtime VU id (overrides infer)")
    args = parser.parse_args()

    TRAJECTORY_CSV = args.csv
    LSTM_MODEL_PATH = args.lstm
    SCALER_PATH = args.scaler
    PPO_PTH_PATH = args.pth
    VU_ID = args.vu_id

    print("[selector] working dir:", os.getcwd())
    if not os.path.exists(TRAJECTORY_CSV):
        raise FileNotFoundError(f"{TRAJECTORY_CSV} missing")
    df = pd.read_csv(TRAJECTORY_CSV)
    req = {'sim_time','vehicle_id','x','y','z'}
    if not req.issubset(df.columns):
        raise RuntimeError(f"trajectories.csv missing {req}")
    df = df.sort_values(['vehicle_id','sim_time']).reset_index(drop=True)
    all_ids = df['vehicle_id'].unique().tolist()
    print("[selector] vehicle ids in csv:", all_ids)

    # determine VU_ID (runtime or inferred)
    if VU_ID is None:
        counts = df.groupby('vehicle_id').size().to_dict()
        if counts:
            VU_ID = max(counts.items(), key=lambda t: t[1])[0]
            print("[selector] inferred VU_ID:", VU_ID)
        else:
            VU_ID = all_ids[0]
    else:
        print("[selector] using provided VU_ID:", VU_ID)

    if VU_ID not in all_ids:
        raise RuntimeError(f"VU {VU_ID} not in CSV (available ids: {all_ids})")

    # stable ordering of CoVs
    cov_ids = sorted([vid for vid in all_ids if vid != VU_ID])
    print("[selector] detected cov ids:", cov_ids)
    if len(cov_ids) == 0:
        raise RuntimeError("No CoVs available in CSV to choose from.")

    # ---------- LSTM predict VU future position ----------
    lstm = load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[selector] loaded LSTM and scaler")
    in_shape = lstm.input_shape
    if not (in_shape and len(in_shape) == 3):
        raise RuntimeError(f"Unexpected lstm.input_shape: {in_shape}")
    _, time_steps, feat = in_shape
    df_vu = df[df['vehicle_id'] == VU_ID]
    recent = df_vu[['x','y','z']].values[-time_steps:]
    if recent.shape[0] < time_steps:
        raise RuntimeError("Not enough VU timesteps for LSTM")
    # scaler handling (fallback if dict)
    if isinstance(scaler, dict):
        scaler_X = scaler.get('scaler_X') or scaler.get('scaler_x')
        scaler_y = scaler.get('scaler_y') or scaler.get('scalerY') or None
        if scaler_X is None:
            raise KeyError("scaler_X missing in scaler dict")
    else:
        scaler_X = scaler_y = scaler
    Xs = scaler_X.transform(recent)
    X_in = np.expand_dims(Xs, axis=0)
    y_pred_scaled = lstm.predict(X_in)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    else:
        y_pred = scaler_X.inverse_transform(y_pred_scaled)[0]
    pred_x, pred_y, pred_z = float(y_pred[0]), float(y_pred[1]), float(y_pred[2])
    print(f"[selector] predicted VU pos: {pred_x:.3f},{pred_y:.3f},{pred_z:.3f}")

    # ---------- build features for CoVs ----------
    cov_feats = []
    cov_id_order = []
    for cid in cov_ids:
        df_cov = df[df['vehicle_id'] == cid]
        if len(df_cov) == 0:
            continue
        cx, cy, cz = df_cov[['x','y','z']].values[-1]
        dist = float(np.linalg.norm([cx - pred_x, cy - pred_y, cz - pred_z]))
        # guard against exact zero (could indicate overlap or data bug)
        if dist < 1e-3:
            dist = float(np.random.uniform(1.0, 5.0))
        bw = float(np.random.uniform(2, 10))
        lat = float(np.random.uniform(10, 100))
        trust = float(np.random.uniform(0.5, 1.0))
        cov_feats.append([bw, lat, trust, dist])
        cov_id_order.append(cid)

    cov_feats = np.array(cov_feats, dtype=np.float32)
    num_present = cov_feats.shape[0]
    print(f"[selector] built features for {num_present} CoVs (ordered)")

    # ---------- normalize distance column to [0,1] and other features for stability ----------
    # but keep raw distances to report if needed
    if num_present == 0:
        raise RuntimeError("No valid CoV features found.")

    # deterministic fallback scoring will normalize itself internally
    # ---------- attempt to run PPO policy ----------
    policy_forward = None
    policy_input_dim = None
    policy_output_dim = None
    policy_action = None
    try:
        policy_forward, policy_input_dim, policy_output_dim = try_load_sb3_pth(PPO_PTH_PATH)
        print(f"[selector] policy loaded (input_dim={policy_input_dim}, output_dim={policy_output_dim})")
        obs_flat = cov_feats.flatten()
        L = obs_flat.shape[0]
        if L < policy_input_dim:
            obs_final = np.concatenate([obs_flat, np.zeros(policy_input_dim - L, dtype=np.float32)])
        else:
            obs_final = obs_flat[:policy_input_dim]
        obs_tensor = torch.tensor(obs_final.reshape(1, -1), dtype=torch.float32)
        try:
            actions = policy_forward(obs_tensor)
            policy_action = int(actions[0]) if actions else None
            print(f"[selector] policy action (raw) = {policy_action}")
        except Exception as e:
            print("[selector] policy forward error:", e)
            policy_action = None
    except Exception as e:
        print("[selector] failed to load or run policy, will use deterministic scorer. Error:", e)
        policy_action = None

    # ---------- deterministic scoring (fallback or final tie-break) ----------
    scores = deterministic_score_matrix(cov_feats)  # higher better
    # show a neat debug table
    try:
        import numpy as _np
        print("[selector] CoV table: [id, bw, lat, trust, dist, score]")
        for i, cid in enumerate(cov_id_order):
            print(f"  {cid:>4}  {cov_feats[i,0]:6.2f} {cov_feats[i,1]:6.2f} {cov_feats[i,2]:5.3f} {cov_feats[i,3]:7.2f}  {scores[i]:6.4f}")
    except Exception:
        pass

    chosen_id = None
    # if policy_action points to a valid index, trust it
    if policy_action is not None and 0 <= policy_action < num_present:
        chosen_id = cov_id_order[policy_action]
        print(f"[selector] policy-selected cov_id={chosen_id} (index {policy_action})")
    else:
        # fallback: pick argmax of deterministic scorer
        best_idx = int(np.argmax(scores))
        chosen_id = cov_id_order[best_idx]
        print(f"[selector] fallback-selected cov_id={chosen_id} (score idx {best_idx})")

    # write output CSV with chosen_id (always a cov id when available)
    out = {"chosen_source": "policy" if (policy_action is not None and 0 <= policy_action < num_present) else "fallback", 
           "cov_id": int(chosen_id), 
           "predicted_vu_position": [pred_x, pred_y, pred_z],
           "vu_id_used": int(VU_ID)}
    pd.DataFrame([out]).to_csv("drl_cov_selection_result.csv", index=False)
    print("[selector] decision saved:", out)

if __name__ == "__main__":
    main()
