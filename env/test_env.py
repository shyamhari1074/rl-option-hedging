from env.option_hedging_env import OptionHedgingEnv

env = OptionHedgingEnv()

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

print("Episode reward:", total_reward)
