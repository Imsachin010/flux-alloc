import random


def first_fit(heap, request_size):

    for i, block in heap.free_blocks():
        if block.size >= request_size:
            return heap.allocate(i, request_size)

    return None


def best_fit(heap, request_size):

    candidates = [
        (i, b) for i, b in heap.free_blocks() if b.size >= request_size
    ]

    if not candidates:
        return None

    best = min(candidates, key=lambda x: x[1].size)

    return heap.allocate(best[0], request_size)


def worst_fit(heap, request_size):

    candidates = [
        (i, b) for i, b in heap.free_blocks() if b.size >= request_size
    ]

    if not candidates:
        return None

    worst = max(candidates, key=lambda x: x[1].size)

    return heap.allocate(worst[0], request_size)


def random_fit(heap, request_size):

    candidates = [
        (i, b) for i, b in heap.free_blocks() if b.size >= request_size
    ]

    if not candidates:
        return None

    i, _ = random.choice(candidates)

    return heap.allocate(i, request_size)