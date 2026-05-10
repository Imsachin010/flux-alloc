import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/tensorboard_metrics.csv")

# Academic Style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300
})

reward = df[df["metric"]=="rollout/ep_rew_mean"]

plt.figure(figsize=(8, 5))
plt.plot(reward["step"], reward["value"], color='#1f77b4', linewidth=2)
plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.grid(True, linestyle=':', alpha=0.7)
os.makedirs("assets", exist_ok=True)
plt.tight_layout()
plt.savefig("assets/reward_curve.png", bbox_inches='tight')