from stable_baselines3 import PPO
import random

from core.heap import Heap
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.workload_generator import WorkloadGenerator
from core.metrics import Metrics
from core.rl_env import MemoryEnv


def evaluate_rl(model, episodes=20):

    frag_list = []
    util_list = []
    fail_list = []

    for _ in range(episodes):

        env = MemoryEnv()

        state = env.reset()

        failures = 0
        total_malloc = 0

        done = False

        while not done:

            action, _ = model.predict(state)

            state, reward, done, _ = env.step(action)

        heap = env.heap

        frag = Metrics.external_fragmentation(heap)
        util = Metrics.utilization(heap)

        frag_list.append(frag)
        util_list.append(util)

    return sum(frag_list)/len(frag_list), sum(util_list)/len(util_list)


def main():

    model = PPO.load("assets/rl_allocator")

    frag, util = evaluate_rl(model)

    print("RL Agent")
    print("Fragmentation:", frag)
    print("Utilization:", util)


if __name__ == "__main__":
    main()