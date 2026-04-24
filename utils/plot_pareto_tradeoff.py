import os
import random
import numpy as np
import matplotlib.pyplot as plt

from core.heap import Heap
from core.workload_generator import WorkloadGenerator
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv

import sys
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer
from sb3_contrib import MaskablePPO

def get_classic_metrics(strategy_fn, heap_size, workload):
    heap = Heap(heap_size)
    active_blocks = []
    
    for request in workload:
        op = request[0]
        if op == "malloc":
            size = request[1]
            block_id = strategy_fn(heap, size)
            if block_id is not None:
                active_blocks.append(block_id)
        elif op == "free":
            if active_blocks:
                block = random.choice(active_blocks)
                heap.free(block)
                active_blocks.remove(block)

    frag = Metrics.external_fragmentation(heap)
    util = Metrics.utilization(heap)
    return util, frag

def get_rl_metrics(model, heap_size, workload):
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    state = env.reset()
    env.workload = workload 
    
    while env.ptr < len(env.workload):
        action_masks = env.get_action_mask()
        action, _ = model.predict(state, action_masks=action_masks)
        state, reward, done, _ = env.step(action)
        if done:
            break

    frag = Metrics.external_fragmentation(env.heap)
    util = Metrics.utilization(env.heap)
    return util, frag

def main():
    heap_size = 1024
    num_requests = 1000
    seeds = list(range(42, 52)) # 10 different random workloads

    strategies = {
        "First Fit": first_fit,
        "Best Fit": best_fit,
        "Worst Fit": worst_fit,
        "Random Fit": random_fit
    }

    try:
        rl_model = MaskablePPO.load("assets/rl_direct_allocator")
    except Exception as e:
        print(f"Failed to load RL model: {e}")
        rl_model = None

    data = {name: {'util': [], 'frag': []} for name in strategies}
    if rl_model:
        data["RL Agent"] = {'util': [], 'frag': []}

    print("Running simulations to generate tradeoff graph...")
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        
        generator = WorkloadGenerator(heap_size)
        workload = generator.uniform_workload(num_requests)

        for name, fn in strategies.items():
            random.seed(seed) # Ensure identical free selections for fairness
            u, f = get_classic_metrics(fn, heap_size, workload)
            data[name]['util'].append(u)
            data[name]['frag'].append(f)
            
        if rl_model:
            random.seed(seed)
            u, f = get_rl_metrics(rl_model, heap_size, workload)
            data["RL Agent"]['util'].append(u)
            data["RL Agent"]['frag'].append(f)

    # Plotting Logic
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    all_points = []

    for i, (name, metrics) in enumerate(data.items()):
        plt.scatter(metrics['util'], metrics['frag'], label=name, color=colors[i % len(colors)], alpha=0.7)
        for u, f in zip(metrics['util'], metrics['frag']):
            all_points.append((u, f, name))

    # Calculate Pareto Frontier
    # Optimal allocator MAXIMIZES Utilization and MINIMIZES Fragmentation.
    # Point A dominates Point B if A.util >= B.util AND A.frag <= B.frag (and is strictly better in one)
    pareto_front = []
    for p in all_points:
        is_dominated = False
        for other in all_points:
            if other == p: continue
            if other[0] >= p[0] and other[1] <= p[1] and (other[0] > p[0] or other[1] < p[1]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(p)
            
    # Sort pareto front by utilization to draw a line connecting them
    pareto_front.sort(key=lambda x: x[0])
    
    if pareto_front:
        p_util = [p[0] for p in pareto_front]
        p_frag = [p[1] for p in pareto_front]
        plt.plot(p_util, p_frag, 'k--', label='Pareto Frontier (Optimal)', linewidth=2)
        
        # Highlight the optimal points with stars
        plt.scatter(p_util, p_frag, color='black', marker='*', s=150, zorder=5)

    plt.title("Tradeoff Graph: Utilization vs Fragmentation (10 Seeds)")
    plt.xlabel("Utilization (Higher is Better →)")
    plt.ylabel("Fragmentation (Lower is Better ↓)")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/pareto_tradeoff.png', dpi=300, bbox_inches='tight')
    print("Saved plot successfully to assets/pareto_tradeoff.png")

if __name__ == "__main__":
    main()
