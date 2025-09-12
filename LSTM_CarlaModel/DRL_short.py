import os, random, math
from pathlib import Path
import numpy as np, pandas as pd, joblib, tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from tensorflow.keras.models import load_model
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

def set_seed(s):
    np.random.seed(s); random.seed(s); tf.random.set_seed(s)

def auto_detect_cols(df):
    c = [x.lower() for x in df.columns]
    m = {}
    for cand in ['time','t','sim_time','timestamp','frame','step']:
        if cand in c: m['time']=df.columns[c.index(cand)]; break
    for cand in ['vehicle_id','veh_id','id','actor_id','agent_id','vehicle']:
        if cand in c: m['id']=df.columns[c.index(cand)]; break
    for cand in ['x','pos_x','px','lon','longitude']:
        if cand in c: m['x']=df.columns[c.index(cand)]; break
    for cand in ['y','pos_y','py','lat','latitude']:
        if cand in c: m['y']=df.columns[c.index(cand)]; break
    for cand in ['z','pos_z','pz','alt','height']:
        if cand in c: m['z']=df.columns[c.index(cand)]; break
    return m

def load_traces(p):
    if not Path(p).exists(): raise FileNotFoundError(p+" not found")
    df = pd.read_csv(p); m = auto_detect_cols(df)
    for k in ['time','id','x','y','z']:
        if k not in m: raise ValueError("missing cols, detected: "+str(m))
    t,idc,x,y,z = m['time'],m['id'],m['x'],m['y'],m['z']
    df = df[[t,idc,x,y,z]].dropna().sort_values([idc,t])
    traces={}
    for vid,g in df.groupby(idc):
        g=g.sort_values(t)
        coords=np.vstack([g[x].values.astype(float),g[y].values.astype(float),g[z].values.astype(float)]).T
        times=g[t].values.astype(float)
        traces[str(vid)]={"coords":coords,"times":times}
    return traces

def load_lstm(dir_=LSTM_MODEL_DIR):
    m=os.path.join(dir_,LSTM_MODEL_NAME); s=os.path.join(dir_,LSTM_SCALER_NAME)
    if not os.path.exists(m) or not os.path.exists(s): raise FileNotFoundError("LSTM model/scaler missing")
    model=load_model(m); d=joblib.load(s)
    return model,d['scaler_X'],d['scaler_y']

def ensure_seq(seq, L):
    if seq.shape[0]>=L: return seq[-L:].copy()
    pad=np.repeat(seq[0:1], repeats=(L-seq.shape[0]), axis=0)
    return np.vstack([pad,seq]).astype(float)

def predict_next(seq, model, sX, sy, delta=True):
    arr=np.array(seq,dtype=float)
    flat=arr.reshape(-1,3)
    flat_s=sX.transform(flat)
    inp=flat_s.reshape(1,arr.shape[0],3)
    p_s=model.predict(inp,verbose=0)[0]
    p=sy.inverse_transform(p_s.reshape(1,-1))[0]
    return (arr[-1]+p) if delta else p

class VuSelectEdgeDTEnv(gym.Env):
    def __init__(self,traces,vu_id,cov_ids,lstm,sX,sy,seq_len=SEQ_LEN):
        super().__init__()
        self.traces=traces; self.vu_id=str(vu_id); self.cov_ids=[str(c) for c in cov_ids]
        self.num_cov=len(self.cov_ids); self.seq_len=seq_len
        self.lstm=lstm; self.sX=sX; self.sy=sy
        obs_dim=3+3*self.num_cov + self.num_cov*3 + 3+1
        self.observation_space=spaces.Box(low=-1e9,high=1e9,shape=(obs_dim,),dtype=np.float32)
        self.action_space=spaces.Discrete(self.num_cov+1)
        try:
            self.max_t=min(len(self.traces[self.vu_id]['coords'])-2,
                           min(len(self.traces[c]['coords'])-2 for c in self.cov_ids))
        except KeyError:
            raise KeyError("ids missing")
        self.t=self.seq_len; self._init_episode_globals()

    def _init_episode_globals(self):
        self.base_latencies=np.random.uniform(NETWORK_LATENCY_RANGE[0],NETWORK_LATENCY_RANGE[1],self.num_cov)
        self.bandwidths=np.random.uniform(BANDWIDTH_RANGE_Mbps[0],BANDWIDTH_RANGE_Mbps[1],self.num_cov)
        self.trust_scores=np.random.uniform(0.4,1.0,self.num_cov)
        self.edge_processing_delay=np.random.uniform(EDGE_PROCESSING_DELAY_RANGE[0],EDGE_PROCESSING_DELAY_RANGE[1])
        self.edge_dt_staleness=np.random.randint(0,DT_STALENESS_MAX+1)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed); random.seed(seed)
        low=self.seq_len; high=max(low+1,self.max_t)
        self.t=np.random.randint(low,high); self._init_episode_globals()
        return self._get_obs(),{}

    def _simulate_edge_dt_prediction(self):
        vu_coords=self.traces[self.vu_id]['coords']
        t_for_dt=max(self.seq_len,self.t-self.edge_dt_staleness)
        seq=ensure_seq(vu_coords[:t_for_dt+1],self.seq_len)
        dt_pred=predict_next(seq,self.lstm,self.sX,self.sy,True)
        true_next=vu_coords[self.t+1]
        err=float(np.linalg.norm(dt_pred-true_next))
        fidelity=1.0/(1.0+err)
        fidelity*=max(0.0,1.0-(self.edge_dt_staleness/max(1,DT_STALENESS_MAX)))
        return np.array(dt_pred),float(fidelity)

    def _get_obs(self):
        vu_coords=self.traces[self.vu_id]['coords'][:self.t+1]
        seq=ensure_seq(vu_coords,self.seq_len)
        vu_local_pred=predict_next(seq,self.lstm,self.sX,self.sy,True)
        cov_curr=[]; latencies=[]; bws=[]; trusts=[]
        for i,cid in enumerate(self.cov_ids):
            coords=self.traces[cid]['coords']
            cov_curr.append(coords[self.t].tolist())
            lat=max(0.0,self.base_latencies[i]+np.random.normal(0.0,0.02)); latencies.append(lat)
            bws.append(max(0.1,self.bandwidths[i]+np.random.normal(0.0,1.0)))
            trusts.append(self.trust_scores[i])
        cov_curr_flat=np.array(cov_curr).reshape(-1)
        edge_dt_pred,edge_dt_fidelity=self._simulate_edge_dt_prediction()
        obs=np.concatenate([vu_local_pred,cov_curr_flat,np.array(latencies),np.array(bws),np.array(trusts),edge_dt_pred,np.array([edge_dt_fidelity])]).astype(np.float32)
        self._cached={'vu_local_pred':np.array(vu_local_pred),
                      'cov_true_next':np.array([self.traces[c]['coords'][self.t+1] for c in self.cov_ids]),
                      'vu_true_next':np.array(self.traces[self.vu_id]['coords'][self.t+1]),
                      'latencies':np.array(latencies),'bws':np.array(bws),'trusts':np.array(trusts),
                      'edge_dt_pred':np.array(edge_dt_pred),'edge_dt_fidelity':float(edge_dt_fidelity)}
        return obs

    def step(self,action):
        assert 0<=action<(self.num_cov+1)
        i=int(action); c=self._cached
        vu_true_next=c['vu_true_next']; cov_true_next=c['cov_true_next']
        if i<self.num_cov:
            perceived=cov_true_next[i]; comm_delay=float(c['latencies'][i]); edge_proc=0.0
            bw_used=float(c['bws'][i])*0.05; trust=float(c['trusts'][i]); source=f"cov_{self.cov_ids[i]}"
        else:
            perceived=c['edge_dt_pred']; network_rr=float(np.mean(self.base_latencies))
            comm_delay=float(max(0.0,network_rr+np.random.normal(0.0,0.02)))
            edge_proc=float(self.edge_processing_delay); bw_used=0.5; trust=float(np.mean(self.trust_scores)); source="edge_dt"
        dist=float(np.linalg.norm(perceived-vu_true_next)); perception_gain=1.0/(1.0+dist)
        dt_fidelity=float(c['edge_dt_fidelity']) if i==self.num_cov else 0.0
        reward=(W_PERCEPTION*perception_gain - W_COMM_DELAY*comm_delay - W_PROC_DELAY*edge_proc + W_TRUST*trust + W_DT_FID*dt_fidelity - W_BW_COST*bw_used)
        self.t+=1; terminated=bool(self.t>=self.max_t); truncated=False
        obs=self._get_obs()
        info={'source':source,'distance':dist,'perception_gain':perception_gain,'comm_delay':comm_delay,'edge_proc_delay':edge_proc,'bw_used_mbps':bw_used,'trust':trust,'dt_fidelity':dt_fidelity}
        return obs,float(reward),terminated,truncated,info

def build_envs_and_train(traces,vu_id,lstm,sX,sy,seq_len=SEQ_LEN,num_cov=NUM_COV):
    vids=[v for v in traces.keys() if v!=str(vu_id)]
    if len(vids)<num_cov: raise ValueError("Not enough CoVs")
    random.shuffle(vids); cov_ids=vids[:num_cov]
    def make_env(): return VuSelectEdgeDTEnv(traces,vu_id,cov_ids,lstm,sX,sy,seq_len=seq_len)
    env=DummyVecEnv([make_env])
    model=PPO("MlpPolicy",env,verbose=VERBOSE,seed=RANDOM_SEED)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    model.save(OUT_PPO)
    return model,cov_ids

def evaluate_policy(model,traces,vu_id,cov_ids,lstm,sX,sy,episodes=8):
    env=VuSelectEdgeDTEnv(traces,vu_id,cov_ids,lstm,sX,sy)
    rows=[]
    for ep in range(episodes):
        obs,info=env.reset(seed=RANDOM_SEED+ep); done=False
        while not done:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(int(action))
            rows.append({'chosen_source':info['source'],'distance':info['distance'],'perception_gain':info['perception_gain'],'comm_delay':info['comm_delay'],'edge_proc_delay':info['edge_proc_delay'],'bw_used_mbps':info['bw_used_mbps'],'trust':info['trust'],'dt_fidelity':info['dt_fidelity'],'reward':reward})
            done=terminated or truncated
    df=pd.DataFrame(rows); df.to_csv(EVAL_CSV,index=False)
    print("Saved:",EVAL_CSV); print("Mean reward:",df['reward'].mean()); print("Mean distance:",df['distance'].mean()); print(df['chosen_source'].value_counts())

def main():
    set_seed(RANDOM_SEED)
    traces=load_traces(CSV_PATH)
    if str(VU_ID) not in traces: raise SystemExit("VU id not found")
    lstm,sX,sy=load_lstm(LSTM_MODEL_DIR)
    model,cov_ids=build_envs_and_train(traces,VU_ID,lstm,sX,sy)
    evaluate_policy(model,traces,VU_ID,cov_ids,lstm,sX,sy,episodes=8)

if __name__=="__main__":
    main()
