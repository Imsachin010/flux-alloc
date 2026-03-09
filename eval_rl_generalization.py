from stable_baselines3 import PPO
import random

from heap import Heap
from allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from workload_generator import WorkloadGenerator
from metrics import Metrics
from rl_env import MemoryEnv


# -------------------------------
# Baseline evaluation
# -------------------------------

def run_baseline(strategy_fn, workload, heap_size=1024):

    heap = Heap(heap_size)

    failures = 0
    total_malloc = 0

    active_blocks = []

    for request in workload:

        if request[0] == "malloc":

            size = request[1]

            block_id = strategy_fn(heap, size)

            total_malloc += 1

            if block_id is None:
                failures += 1
            else:
                active_blocks.append(block_id)

        elif request[0] == "free":

            if active_blocks:
                block = random.choice(active_blocks)
                heap.free(block)
                active_blocks.remove(block)

    frag = Metrics.external_fragmentation(heap)
    util = Metrics.utilization(heap)
    fail_rate = Metrics.allocation_failure_rate(failures, total_malloc)

    return frag, util, fail_rate


# -------------------------------
# RL evaluation
# -------------------------------

def run_rl(model, workload, heap_size=1024):

    env = MemoryEnv(heap_size=heap_size)

    env.heap.reset()

    env.workload = workload
    env.ptr = 0
    env.active_blocks = []

    state = env._get_state()

    while env.ptr < len(workload):

        action, _ = model.predict(state)

        state, reward, done, _ = env.step(action)

        if done:
            break

    heap = env.heap

    frag = Metrics.external_fragmentation(heap)
    util = Metrics.utilization(heap)

    return frag, util


# -------------------------------
# Experiment runner
# -------------------------------

def evaluate():

    model = PPO.load("rl_allocator")

    generator = WorkloadGenerator(1024)

    workloads = {
        "Uniform": generator.uniform_workload(1000),
        "Bimodal": generator.bimodal_workload(1000),
        "Adversarial": generator.adversarial_workload()
    }

    strategies = {
        "First Fit": first_fit,
        "Best Fit": best_fit,
        "Worst Fit": worst_fit,
        "Random Fit": random_fit
    }

    print("\n========== BASELINE RESULTS ==========\n")

    for workload_name, workload in workloads.items():

        print("\nWorkload:", workload_name)

        for name, strategy in strategies.items():

            frag, util, fail = run_baseline(strategy, workload)

            print(f"{name:12} | Frag: {frag:.3f} | Util: {util:.3f} | Fail: {fail:.3f}")

    print("\n========== RL RESULTS ==========\n")

    for workload_name, workload in workloads.items():

        frag, util = run_rl(model, workload)

        print(f"RL Agent ({workload_name}) | Frag: {frag:.3f} | Util: {util:.3f}")


# -------------------------------

if __name__ == "__main__":
    evaluate()