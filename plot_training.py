import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tensorboard_metrics.csv")

reward = df[df["metric"]=="rollout/ep_rew_mean"]

plt.figure()
plt.plot(reward["step"], reward["value"])
plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.title("PPO Training Reward")
plt.savefig("reward_curve.png")
plt.show()