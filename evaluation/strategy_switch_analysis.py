from stable_baselines3 import PPO
from core.rl_env import MemoryEnv
import numpy as np

model = PPO.load("assets/rl_allocator")

env = MemoryEnv()

state = env.reset()

switches = 0
prev_action = None

for _ in range(5000):

    action,_ = model.predict(state)

    if prev_action is not None and action != prev_action:
        switches += 1

    prev_action = action

    state,_,done,_ = env.step(action)

    if done:
        state = env.reset()

print("Total strategy switches:",switches)