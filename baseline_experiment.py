from heap import Heap
from allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from workload_generator import WorkloadGenerator
from metrics import Metrics
import random


def run_allocator(strategy_fn, heap_size=1024, num_requests=1000):

    heap = Heap(heap_size)
    generator = WorkloadGenerator(heap_size)

    workload = generator.uniform_workload(num_requests)

    failures = 0
    total_malloc = 0

    active_blocks = []

    for request in workload:

        op = request[0]

        # -------------------
        # MALLOC
        # -------------------
        if op == "malloc":

            size = request[1]

            block_id = strategy_fn(heap, size)

            total_malloc += 1

            if block_id is None:
                failures += 1
            else:
                active_blocks.append(block_id)

        # -------------------
        # FREE
        # -------------------
        elif op == "free":

            if active_blocks:
                block = random.choice(active_blocks)
                heap.free(block)
                active_blocks.remove(block)

    frag = Metrics.external_fragmentation(heap)
    util = Metrics.utilization(heap)
    fail_rate = Metrics.allocation_failure_rate(failures, total_malloc)

    return frag, util, fail_rate


def main():

    strategies = {
        "First Fit": first_fit,
        "Best Fit": best_fit,
        "Worst Fit": worst_fit,
        "Random Fit": random_fit
    }

    for name, fn in strategies.items():

        frag, util, fail = run_allocator(fn)

        print(name)
        print(f"Fragmentation: {frag:.3f}")
        print(f"Utilization: {util:.3f}")
        print(f"Failure Rate: {fail:.3f}")
        print("-"*40)


if __name__ == "__main__":
    main()