import random
from sb3_contrib import MaskablePPO
from core.heap import Heap
from core.workload_generator import WorkloadGenerator
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv

# Backward compatibility for loading the model
import sys
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer

def run_classic(strategy_fn, heap_size, workload):
    random.seed(42)  # ensure deterministic frees
    heap = Heap(heap_size)
    active_blocks = []
    failures = 0
    total_malloc = 0

    for request in workload:
        op = request[0]
        if op == "malloc":
            size = request[1]
            block_id = strategy_fn(heap, size)
            total_malloc += 1
            if block_id is None:
                failures += 1
            else:
                active_blocks.append(block_id)
        elif op == "free":
            if active_blocks:
                block = random.choice(active_blocks)
                heap.free(block)
                active_blocks.remove(block)

    frag = Metrics.external_fragmentation(heap)
    util = Metrics.utilization(heap)
    fail_rate = failures / total_malloc if total_malloc else 0
    score = util - frag
    return frag, util, fail_rate, score

def run_rl(heap_size, workload):
    random.seed(42)
    model = MaskablePPO.load("assets/rl_direct_allocator")
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    
    # Initialize env and FORCE it to use the exact fixed workload
    state = env.reset()
    env.workload = workload 
    
    failures = 0
    total_malloc = 0

    while env.ptr < len(env.workload):
        req = env.workload[env.ptr]
        if req[0] == "malloc":
            total_malloc += 1
            
        action_masks = env.get_action_mask()
        action, _ = model.predict(state, action_masks=action_masks)
        state, reward, done, info = env.step(action)

        if req[0] == "malloc" and info.get("allocation_failed", False):
            failures += 1

        if done:
            break

    frag = Metrics.external_fragmentation(env.heap)
    util = Metrics.utilization(env.heap)
    fail_rate = failures / total_malloc if total_malloc else 0
    score = util - frag
    return frag, util, fail_rate, score

def main():
    # 1. Generate ONE single workload fixed array to benchmark against
    random.seed(42)
    heap_size = 1024
    num_requests = 1000
    
    generator = WorkloadGenerator(heap_size)
    fixed_workload = generator.uniform_workload(num_requests)

    results = []

    strategies = {
        "First Fit": first_fit,
        "Best Fit": best_fit,
        "Worst Fit": worst_fit,
        "Random Fit": random_fit
    }

    # 2. Run Classic Allocators
    for name, fn in strategies.items():
        frag, util, fail, score = run_classic(fn, heap_size, fixed_workload)
        results.append((name, util, frag, fail, score))

    # 3. Run RL Allocator
    rl_frag, rl_util, rl_fail, rl_score = run_rl(heap_size, fixed_workload)
    results.append(("MaskablePPO Agent", rl_util, rl_frag, rl_fail, rl_score))

    # 4. Print clean tabular output
    print("\n" + "="*80)
    print(f"{'Strategy':<20} | {'Utilization':<11} | {'Fragmentation':<13} | {'Fail Rate':<9} | {'Score (Util-Frag)'}")
    print("-" * 80)
    
    # Sort by Score (Efficiency) descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    for name, util, frag, fail, score in results:
        print(f"{name:<20} | {util:<11.4f} | {frag:<13.4f} | {fail:<9.4f} | {score:+.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
