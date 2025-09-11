# ===============================
# PPO Agent for CoV Selection
# ===============================
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

# ---- Custom Environment ----
class V2VEnv(gym.Env):
    def __init__(self, n_cov=5):
        super(V2VEnv, self).__init__()
        
        self.n_cov = n_cov
        # State: [VU position, CoVs position, delay]
        self.observation_space = spaces.Box(low=-100, high=100, shape=(2*n_cov+1,), dtype=np.float32)
        # Action: choose 1 CoV out of n_cov
        self.action_space = spaces.Discrete(n_cov)

    def reset(self):
        self.vu_position = np.random.uniform(-50, 50)
        self.cov_positions = np.random.uniform(-50, 50, self.n_cov)
        self.delays = np.random.uniform(1, 10, self.n_cov)  # fixed bandwidth assumption
        return self._get_state()

    def _get_state(self):
        return np.concatenate(([self.vu_position], self.cov_positions, self.delays))

    def step(self, action):
        chosen_cov_pos = self.cov_positions[action]
        chosen_delay = self.delays[action]

        # Reward: perception gain (inverse of distance) - delay penalty
        perception_gain = 1 / (1 + abs(self.vu_position - chosen_cov_pos))
        reward = perception_gain - 0.05 * chosen_delay

        # New positions (simulate mobility)
        self.vu_position += np.random.uniform(-1, 1)
        self.cov_positions += np.random.uniform(-1, 1, self.n_cov)

        done = False  # continuous episode
        return self._get_state(), reward, done, {}

# ---- Train PPO Agent ----
env = V2VEnv(n_cov=5)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# ---- Test the Agent ----
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward:.3f}")
