from stable_baselines3 import PPO
from env.option_hedging_env import OptionHedgingEnv


def train():
    env = OptionHedgingEnv(
        S0=100,
        K=100,
        T=1.0,
        r=0.0,
        mu=0.0,
        sigma=0.2,
        steps=50,
        transaction_cost=0.001,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
    )

    model.learn(total_timesteps=200_000)

    model.save("ppo_option_hedger")
if __name__ == "__main__":
    train()
