"""Shared one-pass lookahead simulation on a fixed workload (for eval + comparisons)."""

from __future__ import annotations

import random
from pathlib import Path

from core.heap import Heap
from core.metrics import Metrics
from lookahead.lookahead_allocator import LookaheadAllocator


def run_lookahead_bench(
    heap_size: int,
    workload: list,
    *,
    model_path: str | Path | None = None,
    lookahead_steps: int = 12,
    neural_weight: float = 0.2,
    sim_weight: float = 0.8,
    seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """
    Returns (utilization, fragmentation, fail_rate, util - frag, largest_free / heap_size).
    """
    random.seed(seed)
    heap = Heap(heap_size)
    active: list = []

    root = Path(__file__).resolve().parent
    if model_path is not None:
        p = Path(model_path)
    else:
        p = root / "lookahead_ranker.pt"
    load = str(p) if p.is_file() else None

    alloc = LookaheadAllocator(
        model_path=load,
        heap_size=heap_size,
        lookahead_steps=lookahead_steps,
        neural_weight=neural_weight,
        sim_weight=sim_weight,
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
            else:
                bid = heap.allocate(idx, size)
                if bid is not None:
                    active.append(bid)
        else:
            if active:
                b = random.choice(active)
                heap.free(b)
                active.remove(b)

    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = heap.largest_free_block() / float(heap_size)
    return float(util), float(frag), float(fail_rate), float(score), float(largest)
