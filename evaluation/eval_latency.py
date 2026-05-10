import time
import random
import numpy as np
from sb3_contrib import MaskablePPO
from core.heap import Heap
from core.allocator_strategies import best_fit
from core.workload_generator import WorkloadGenerator
from core.rl_env_direct_final import DirectPlacementEnv

# Backward compatibility
import sys
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer

def measure_best_fit_latency(heap_size, workload):
    heap = Heap(heap_size)
    active_blocks = []
    
    # Pre-warm
    best_fit(heap, 10)
    
    latencies = []
    for request in workload:
        if request[0] == "malloc":
            size = request[1]
            
            start_t = time.perf_counter()
            block_id = best_fit(heap, size)
            end_t = time.perf_counter()
            
            latencies.append((end_t - start_t) * 1000) # milliseconds
            
            if block_id is not None:
                active_blocks.append(block_id)
        elif request[0] == "free":
            if active_blocks:
                block = random.choice(active_blocks)
                heap.free(block)
                active_blocks.remove(block)
                
    return np.mean(latencies), np.median(latencies), np.max(latencies)


def measure_neural_ranker_forward_ms(iters: int = 2000) -> tuple[float, float, float]:
    """Single 10D feature vector through Lookahead MLP (CPU, batch=1) — not full tree search."""
    import torch
    from lookahead.neural_ranker import NeuralRanker

    m = NeuralRanker()
    m.eval()
    x = torch.randn(1, 10, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(20):
            m(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            m(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.median(times)), float(np.max(times))


def measure_rl_latency(model, heap_size, workload):
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    state = env.reset()
    env.workload = workload 
    
    latencies = []
    
    # Pre-warm
    action_masks = env.get_action_mask()
    model.predict(state, action_masks=action_masks)
    
    while env.ptr < len(env.workload):
        req = env.workload[env.ptr]
        
        if req[0] == "malloc":
            action_masks = env.get_action_mask()
            
            start_t = time.perf_counter()
            action, _ = model.predict(state, action_masks=action_masks)
            end_t = time.perf_counter()
            
            latencies.append((end_t - start_t) * 1000) # milliseconds
            state, _, done, _ = env.step(action)
        else:
            state, _, done, _ = env.step(0) # Action is ignored on free
            
        if done:
            break
            
    return np.mean(latencies), np.median(latencies), np.max(latencies)

def main():
    random.seed(42)
    np.random.seed(42)
    
    heap_size = 1024
    generator = WorkloadGenerator(heap_size)
    workload = generator.uniform_workload(500)
    
    print("\n" + "="*80)
    print("ALLOCATION LATENCY ANALYSIS (CPU Time vs Memory Tradeoff)")
    print("="*80)

    # 1. Best Fit
    random.seed(42)
    bf_mean, bf_med, bf_max = measure_best_fit_latency(heap_size, workload)
    print(f"\n[Classic: Best Fit]")
    print(f"Mean Latency:   {bf_mean:.4f} ms")
    print(f"Median Latency: {bf_med:.4f} ms")
    print(f"Max Latency:    {bf_max:.4f} ms")

    # 2. RL Agent
    model = MaskablePPO.load("assets/rl_direct_allocator")
    random.seed(42)
    rl_mean, rl_med, rl_max = measure_rl_latency(model, heap_size, workload)
    print(f"\n[RL Agent: MaskablePPO + Transformer]")
    print(f"Mean Latency:   {rl_mean:.4f} ms")
    print(f"Median Latency: {rl_med:.4f} ms")
    print(f"Max Latency:    {rl_max:.4f} ms")
    
    # 3. Lookahead MLP (ranker) — forward pass only, not O(cands × depth × copy)
    rnk_mean, rnk_med, rnk_max = measure_neural_ranker_forward_ms()
    print(f"\n[Lookahead ranker: MLP only, one forward, CPU]")
    print(f"Mean Latency:   {rnk_mean:.4f} ms")
    print(f"Median Latency: {rnk_med:.4f} ms")
    print(f"Max Latency:    {rnk_max:.4f} ms")

    ratio = rl_mean / bf_mean if bf_mean > 0 else 0
    ratio_rnk = rnk_mean / bf_mean if bf_mean > 0 else 0
    print("\n" + "-"*80)
    print(
        f"Conclusion: MaskablePPO+Transformer is ~{ratio:.1f}x slower per decision than Best Fit "
        f"on this trace. The small ranker MLP alone is ~{ratio_rnk:.1f}x vs Best Fit. "
        "Full Lookahead+Neural adds heap copies and simulated futures (orders of magnitude "
        "more than Best Fit) when used as a planning pass."
    )
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
