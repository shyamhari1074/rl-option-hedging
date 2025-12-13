
import numpy as np


def simulate_gbm(S0, mu, sigma, T, steps, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    prices = [S0]

    for _ in range(steps):
        z = np.random.normal()
        S_next = prices[-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
        prices.append(S_next)

    return np.array(prices)
