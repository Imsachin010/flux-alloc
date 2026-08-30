import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.heap import Heap
from core.metrics import Metrics
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit

def run_heuristic(strategy_fn, workload, heap_size=1024):
    random.seed(42)
    heap = Heap(heap_size)
    active = []
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
                active.append(block_id)
        elif op == "free" and active:
            block = random.choice(active)
            heap.free(block)
            active.remove(block)
    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    return util, frag, fail_rate, score

def generate_alternating_workload(num_requests=1000):
    # Alternating small and large allocations, with occasional frees
    random.seed(42)
    workload = []
    active_count = 0
    for _ in range(num_requests):
        if active_count > 10 and random.random() < 0.3:
            workload.append(("free", None))
            active_count -= 1
        else:
            # 50% small, 50% large
            if random.random() < 0.5:
                size = random.randint(2, 8)
            else:
                size = random.randint(64, 128)
            workload.append(("malloc", size))
            active_count += 1
    return workload

def main():
    heap_size = 1024
    workload = generate_alternating_workload(1000)

    print("Evaluating Alternating Workload (1000 requests):")
    for name, fn in [
        ("Best Fit", best_fit),
        ("First Fit", first_fit),
        ("Worst Fit", worst_fit),
        ("Random Fit", random_fit)
    ]:
        u, f, fl, sc = run_heuristic(fn, workload, heap_size)
        print(f"  {name:<12} | Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f}")

if __name__ == "__main__":
    main()
