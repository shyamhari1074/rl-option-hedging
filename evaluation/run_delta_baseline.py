import numpy as np
from baselines.delta_hedging import delta_hedging_episode


def run_monte_carlo(n_paths=1000):
    pnls, costs = [], []

    for i in range(n_paths):
        pnl, cost = delta_hedging_episode(
            S0=100,
            K=100,
            T=1.0,
            r=0.0,
            mu=0.0,
            sigma=0.2,
            steps=50,
            transaction_cost=0.001,
            seed=i,
        )
        pnls.append(pnl)
        costs.append(cost)

    print("Delta Hedging Baseline")
    print("----------------------")
    print(f"Mean PnL : {np.mean(pnls):.4f}")
    print(f"Std PnL  : {np.std(pnls):.4f}")
    print(f"Avg Cost : {np.mean(costs):.4f}")


if __name__ == "__main__":
    run_monte_carlo()
