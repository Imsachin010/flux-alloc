from stable_baselines3 import PPO
from core.rl_env import MemoryEnv
import numpy as np
import matplotlib.pyplot as plt


model = PPO.load("assets/rl_allocator")

env = MemoryEnv()

state = env.reset()

action_counts = np.zeros(4)

state_log = []
action_log = []

steps = 5000

for _ in range(steps):

    action, _ = model.predict(state)

    action_counts[action] += 1

    state_log.append(state)
    action_log.append(action)

    state, reward, done, _ = env.step(action)

    if done:
        state = env.reset()


labels = ["First Fit","Best Fit","Worst Fit","Random Fit"]

print("\nAction distribution:")

for i in range(4):
    print(labels[i],":", int(action_counts[i]))


print("\nAction percentages:")

for i in range(4):
    pct = (action_counts[i] / steps) * 100
    print(labels[i],":", round(pct,2), "%")


# --------------------------
# Policy distribution plot
# --------------------------

plt.figure()

plt.bar(labels, action_counts)

plt.title("RL Allocator Policy Distribution")

plt.ylabel("Action Frequency")

plt.savefig("policy_distribution.png")

plt.show()


# --------------------------
# State → Action examples
# --------------------------

print("\nSample State → Action decisions:\n")

for i in range(10):

    s = state_log[i]

    action = labels[action_log[i]]

    print(
        "util:",round(s[0],3),
        "frag:",round(s[1],3),
        "largest:",round(s[2],3),
        "free_blocks:",round(s[3],3),
        "req:",round(s[4],3),
        "→ action:",action
    )