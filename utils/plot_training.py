import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/tensorboard_metrics.csv")

reward = df[df["metric"]=="rollout/ep_rew_mean"]

plt.figure()
plt.plot(reward["step"], reward["value"])
plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.title("PPO Training Reward")
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/reward_curve.png")