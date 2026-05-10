"""Short rollouts on a heap copy: future workload + best-fit mallocs, FIFO frees."""

from __future__ import annotations

import copy

from core.allocator_strategies import best_fit
from core.metrics import Metrics


def outcome_score(heap) -> float:
    return Metrics.utilization(heap) - Metrics.external_fragmentation(heap)


def simulate_after_malloc(
    heap,
    free_index: int,
    size: int,
    workload: list,
    request_ptr: int,
    active_block_ids: list,
    lookahead_steps: int,
) -> float:
    """
    After hypothetically placing the current malloc at `free_index`, run up to
    `lookahead_steps` *future* workload ops (from request_ptr+1) on a heap copy.
    Future mallocs use best-fit; frees pop FIFO from active alloc list.
    Returns a scalar score (higher = better expected future state).
    """
    h = copy.deepcopy(heap)
    act = list(active_block_ids)
    bid = h.allocate(free_index, size)
    if bid is None:
        return -2.0
    act.append(bid)

    future_mallocs = 0
    failures = 0

    for t in range(1, lookahead_steps + 1):
        p = request_ptr + t
        if p >= len(workload):
            break
        w = workload[p]
        if w[0] == "malloc":
            future_mallocs += 1
            r = best_fit(h, w[1])
            if r is None:
                failures += 1
            else:
                act.append(r)
        else:
            if act:
                b = act.pop(0)
                h.free(b)

    base = outcome_score(h)
    if future_mallocs:
        base -= 0.25 * (failures / future_mallocs)
    elif failures:
        base -= 0.25 * failures
    return float(base)
