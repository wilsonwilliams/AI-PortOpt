import numpy as np

import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(self, log_returns, volatilities, risk_free_rate=0.02):
        super(PortfolioEnv, self).__init__()
        self.log_returns = log_returns.values
        self.volatilities = volatilities.values
        self.n_assets = log_returns.shape[1]
        self.n_steps = len(log_returns)
        self.risk_free_rate = risk_free_rate / 252
        self.portfolio_init_value = 10_000.0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets * 2,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.portfolio_init_value
        self.weights = np.random.random(self.n_assets)
        return self._get_obs(), {}

    def _get_obs(self):
        current_returns = self.log_returns[self.current_step]
        current_vols = self.volatilities[self.current_step]
        return np.concatenate([current_returns, current_vols]).astype(np.float32)

    def step(self, action):
        self.current_step += 1  # use next day's returns to calculate portfolio return + sharpe

        action = np.clip(action, 0, None)
        self.weights = action / np.sum(action) if np.sum(action) > 0 else np.ones(self.n_assets) / self.n_assets

        port_return = np.dot(self.weights, np.exp(self.log_returns[self.current_step]) - 1)
        port_vol = np.sqrt(np.dot(self.weights**2, self.volatilities[self.current_step]**2))

        # Reward: daily sharpe ratio
        reward = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        self.portfolio_value *= (1 + port_return)

        done = self.current_step >= self.n_steps - 1
        truncated = False
        info = { "PortValue": self.portfolio_value, "PortReturn": (100 * (self.portfolio_value - self.portfolio_init_value) / self.portfolio_init_value), "PortWeights": self.weights }

        return self._get_obs(), reward, done, truncated, info
