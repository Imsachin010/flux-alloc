import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# SB3 load compatibility
import policy.custom_policy_transformer as custom_policy_transformer
sys.modules["custom_policy_transformer"] = custom_policy_transformer

from sb3_contrib import MaskablePPO

from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.heap import Heap
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv
from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench

def _run_classic_with_largest(strategy_fn, heap_size, workload, seed: int = 42):
    random.seed(seed)
    heap = Heap(heap_size)
    active = []
    failures = 0
    total_malloc = 0
    for i, request in enumerate(workload):
        op = request[0]
        if op == "malloc":
            size = request[1]
            block_id = strategy_fn(heap, size)
            total_malloc += 1
            if block_id is None:
                failures += 1
                print(f"    [Step {i}] malloc({size}) failed")
            else:
                active.append(block_id)
                print(f"    [Step {i}] malloc({size}) succeeded (block_id={block_id}). Heap: {heap.blocks}")
        elif op == "free" and active:
            block = random.choice(active)
            heap.free(block)
            active.remove(block)
            print(f"    [Step {i}] free(block_id={block}) succeeded. Heap: {heap.blocks}")
    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = heap.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, score, largest

def _resolve_ppo_path() -> Path | None:
    z = _ROOT / "assets" / "rl_direct_allocator.zip"
    d = _ROOT / "assets" / "rl_direct_allocator"
    if z.is_file():
        return z
    if d.is_dir() or d.is_file():
        return d
    return None

def _run_rl_with_largest(heap_size, workload, seed: int = 42):
    path = _resolve_ppo_path()
    if path is None:
        return None
    try:
        model = MaskablePPO.load(str(path))
    except Exception:
        return None
    random.seed(seed)
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    state = env.reset()
    env.workload = list(workload)
    failures = 0
    total_malloc = 0
    while env.ptr < len(env.workload):
        idx_step = env.ptr
        req = env.workload[env.ptr]
        if req[0] == "malloc":
            total_malloc += 1
        action_masks = env.get_action_mask()
        action, _ = model.predict(state, action_masks=action_masks)
        state, _, done, info = env.step(int(action))
        if req[0] == "malloc":
            if info.get("allocation_failed", False):
                failures += 1
                print(f"    [Step {idx_step}] malloc({req[1]}) failed (action chosen: {action}). Heap: {env.heap.blocks}")
            else:
                print(f"    [Step {idx_step}] malloc({req[1]}) succeeded (action chosen: {action}). Heap: {env.heap.blocks}")
        else:
            print(f"    [Step {idx_step}] free succeeded. Heap: {env.heap.blocks}")
        if done:
            break
    h = env.heap
    util = Metrics.utilization(h)
    frag = Metrics.external_fragmentation(h)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = h.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, score, largest

def _run_lookahead_with_largest(heap_size, workload, lookahead_steps=12, seed: int = 42):
    random.seed(seed)
    heap = Heap(heap_size)
    active = []
    
    from lookahead.lookahead_allocator import LookaheadAllocator
    alloc = LookaheadAllocator(
        model_path=(_ROOT / "lookahead" / "lookahead_ranker.pt"),
        heap_size=heap_size,
        lookahead_steps=lookahead_steps,
    )
    
    failures = 0
    total_malloc = 0
    for ptr, req in enumerate(workload):
        if req[0] == "malloc":
            size = req[1]
            total_malloc += 1
            idx = alloc.choose_block(heap, size, workload, ptr, active)
            if idx is None:
                failures += 1
                print(f"    [Step {ptr}] malloc({size}) failed (Lookahead returned None)")
            else:
                bid = heap.allocate(idx, size)
                if bid is None:
                    failures += 1
                    print(f"    [Step {ptr}] malloc({size}) failed to allocate at idx={idx}")
                else:
                    active.append(bid)
                    print(f"    [Step {ptr}] malloc({size}) succeeded at idx={idx}. Heap: {heap.blocks}")
        else:
            if active:
                b = random.choice(active)
                heap.free(b)
                active.remove(b)
                print(f"    [Step {ptr}] free(block_id={b}) succeeded. Heap: {heap.blocks}")
                
    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = heap.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, score, largest

def main():
    heap_size = 16
    generator = WorkloadGenerator(heap_size)
    workload = generator.adversarial_workload()
    
    print("Workload details:")
    for i, req in enumerate(workload):
        print(f"  Op {i}: {req}")
    print()

    print("--- 1. Best Fit ---")
    u, f, fl, sc, lg = _run_classic_with_largest(best_fit, heap_size, workload)
    print(f"Results -> Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f}\n")

    print("--- 2. First Fit ---")
    u, f, fl, sc, lg = _run_classic_with_largest(first_fit, heap_size, workload)
    print(f"Results -> Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f}\n")

    print("--- 3. MaskablePPO Agent ---")
    rl_res = _run_rl_with_largest(heap_size, workload)
    if rl_res:
        u, f, fl, sc, lg = rl_res
        print(f"Results -> Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f}\n")
    else:
        print("MaskablePPO Agent skipped.\n")

    print("--- 4. Lookahead+Neural (L3) ---")
    u, f, fl, sc, lg = _run_lookahead_with_largest(heap_size, workload)
    print(f"Results -> Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f}\n")

if __name__ == "__main__":
    main()
