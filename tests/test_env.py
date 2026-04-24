from core.rl_env import MemoryEnv
import random

env = MemoryEnv()

state = env.reset()

for _ in range(20):

    action = random.randint(0,3)

    state, reward, done, _ = env.step(action)

    print("state:", state)
    print("reward:", reward)

    if done:
        break