from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

model = PPO.load("assets/rl_allocator")

util_range = np.linspace(0,1,50)
frag_range = np.linspace(0,1,50)

policy_grid = np.zeros((50,50))

for i,u in enumerate(util_range):
    for j,f in enumerate(frag_range):

        state = np.array([
            u,
            f,
            0.5,
            0.1,
            0.05
        ])

        action,_ = model.predict(state)

        policy_grid[j,i] = action


plt.figure(figsize=(8,6))

plt.imshow(policy_grid,
           origin="lower",
           extent=[0,1,0,1],
           aspect="auto")

plt.colorbar(label="Action")

plt.xlabel("Utilization")
plt.ylabel("Fragmentation")

plt.title("RL Allocator Policy Heatmap")

plt.savefig("policy_heatmap.png")

plt.show()