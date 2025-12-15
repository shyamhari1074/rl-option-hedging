import numpy as np
from stable_baselines3 import PPO

from env.option_hedging_env import OptionHedgingEnv
from baselines.delta_hedging import delta_hedging_episode


def evaluate_ppo(
    model_path,
    n_episodes=500,
):
    env = OptionHedgingEnv()
    model = PPO.load(model_path)

    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def evaluate_delta(n_episodes=500):
    pnls = []
    costs = []

    for i in range(n_episodes):
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

    return np.mean(pnls), np.std(pnls), np.mean(costs)


if __name__ == "__main__":
    print("Evaluating PPO Hedger...")
    ppo_mean, ppo_std = evaluate_ppo("ppo_option_hedger.zip")


    print("\nEvaluating Delta Hedging...")
    delta_mean, delta_std, delta_cost = evaluate_delta()

    print("\nRESULTS")
    print(f"PPO Mean Reward     : {ppo_mean:.2f}")
    print(f"PPO Reward Std      : {ppo_std:.2f}")
    print("--------------------")
    print(f"Delta Mean PnL      : {delta_mean:.2f}")
    print(f"Delta PnL Std       : {delta_std:.2f}")
    print(f"Delta Avg Cost      : {delta_cost:.2f}")
