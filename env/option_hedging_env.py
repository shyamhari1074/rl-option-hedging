import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baselines.black_scholes import call_price
from baselines.market_simulator import simulate_gbm


class OptionHedgingEnv(gym.Env):
    """
    RL environment for option hedging with transaction costs.
    One episode = lifetime of one European option.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        S0=100,
        K=100,
        T=1.0,
        r=0.0,
        mu=0.0,
        sigma=0.2,
        steps=50,
        transaction_cost=0.001,
    ):
        super().__init__()

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.steps = steps
        self.transaction_cost = transaction_cost

        # Action: hedge position in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: [price, time_to_maturity, current_hedge]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.prices = simulate_gbm(
            self.S0, self.mu, self.sigma, self.T, self.steps, seed
        )
        self.dt = self.T / self.steps
        self.t = 0

        self.hedge = 0.0
        self.cash = call_price(self.S0, self.K, self.T, self.r, self.sigma)

        return self._get_obs(), {}

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))

        S = self.prices[self.t]
        trade = action - self.hedge

        cost = self.transaction_cost * abs(trade) * S
        self.cash -= trade * S + cost
        self.hedge = action

        self.t += 1
        done = self.t == self.steps

        reward = -cost

        if done:
            payoff = max(self.prices[-1] - self.K, 0.0)
            portfolio_value = self.cash + self.hedge * self.prices[-1]
            hedging_error = portfolio_value - payoff
            reward -= hedging_error ** 2

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        time_to_maturity = self.T - self.t * self.dt
        price = self.prices[self.t]
        return np.array(
            [price, time_to_maturity, self.hedge], dtype=np.float32
        )
