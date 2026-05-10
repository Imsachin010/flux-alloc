import random
from sb3_contrib import MaskablePPO
from core.heap import Heap
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.workload_generator import WorkloadGenerator
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv

# Backward compatibility for loading the model
import sys
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer

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
    fail_rate = failures / total_malloc if total_malloc else 0
    score = util - frag

    return frag, util, fail_rate, score

# -------------------------------
# RL evaluation
# -------------------------------
def run_rl(model, workload, heap_size=1024):
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
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

# -------------------------------
# Experiment runner
# -------------------------------
def evaluate():
    try:
        model = MaskablePPO.load("assets/rl_direct_allocator")
    except Exception as e:
        print(f"Failed to load RL model: {e}")
        return

    random.seed(42)
    generator = WorkloadGenerator(1024)

    # Generate generalization workloads
    workloads = {
        "Uniform (Standard)": generator.uniform_workload(1000),
        "Bimodal (Mixed Sizes)": generator.bimodal_workload(1000),
        "Adversarial (Stress Test)": generator.adversarial_workload()
    }

    strategies = {
        "Best Fit": best_fit,
    }

    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS: ROBUSTNESS TO DIFFERENT WORKLOADS")
    print("="*80)

    for workload_name, workload in workloads.items():
        print(f"\n--- Testing Workload: {workload_name} ({len(workload)} ops) ---")
        
        # Test baseline
        for name, strategy in strategies.items():
            random.seed(42) # reset for fair comparison
            frag, util, fail, score = run_baseline(strategy, workload)
            print(f"{name:18} | Util: {util:.3f} | Frag: {frag:.3f} | Fail: {fail:.3f} | Score: {score:+.3f}")

        # Test RL
        random.seed(42)
        frag, util, fail, score = run_rl(model, workload)
        print(f"{'MaskablePPO Agent':18} | Util: {util:.3f} | Frag: {frag:.3f} | Fail: {fail:.3f} | Score: {score:+.3f}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    evaluate()