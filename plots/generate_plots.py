import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# Paths
# -------------------------------

CSV_FILE = "tensorboard_metrics.csv"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# -------------------------------
# 1. Training Reward Curve
# -------------------------------

reward = df[df["metric"] == "rollout/ep_rew_mean"]

plt.figure()

plt.plot(reward["step"], reward["value"])

plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.title("PPO Training Reward Curve")

plt.savefig(f"{PLOT_DIR}/reward_curve.png")

plt.close()

# -------------------------------
# 2. KL Divergence Curve
# -------------------------------

kl = df[df["metric"] == "train/approx_kl"]

plt.figure()

plt.plot(kl["step"], kl["value"])

plt.xlabel("Training Steps")
plt.ylabel("KL Divergence")
plt.title("Policy KL Divergence")

plt.savefig(f"{PLOT_DIR}/kl_divergence.png")

plt.close()

# -------------------------------
# 3. Value Loss Curve
# -------------------------------

value_loss = df[df["metric"] == "train/value_loss"]

plt.figure()

plt.plot(value_loss["step"], value_loss["value"])

plt.xlabel("Training Steps")
plt.ylabel("Value Loss")
plt.title("Value Function Loss")

plt.savefig(f"{PLOT_DIR}/value_loss.png")

plt.close()

# -------------------------------
# 4. Fragmentation Comparison
# -------------------------------

methods = ["First Fit","Best Fit","Worst Fit","Random Fit","RL Agent"]

fragmentation = [0.887,0.778,0.919,0.713,0.561]

plt.figure()

plt.bar(methods, fragmentation)

plt.ylabel("Fragmentation")
plt.title("Fragmentation Comparison")

plt.savefig(f"{PLOT_DIR}/fragmentation_comparison.png")

plt.close()

# -------------------------------
# 5. Utilization Comparison
# -------------------------------

utilization = [0.879,0.868,0.759,0.830,0.904]

plt.figure()

plt.bar(methods, utilization)

plt.ylabel("Memory Utilization")
plt.title("Utilization Comparison")

plt.savefig(f"{PLOT_DIR}/utilization_comparison.png")

plt.close()

# -------------------------------
# 6. Policy Distribution
# -------------------------------

policy_labels = ["First Fit","Best Fit","Worst Fit","Random Fit"]

policy_counts = [19,4954,4,23]

plt.figure()

plt.bar(policy_labels, policy_counts)

plt.ylabel("Frequency")
plt.title("RL Policy Action Distribution")

plt.savefig(f"{PLOT_DIR}/policy_distribution.png")

plt.close()

print("\nPlots saved in:", PLOT_DIR)