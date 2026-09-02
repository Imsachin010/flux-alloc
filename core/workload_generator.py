import random
from typing import List, Tuple, Optional


class WorkloadGenerator:

    def __init__(self, heap_size: int = 1024):
        self.heap_size = heap_size

    def _get_rng(self, seed: Optional[int]) -> random.Random:
        if seed is not None:
            return random.Random(seed)
        return random

    def uniform_workload(
        self,
        num_requests: int = 1000,
        seed: Optional[int] = None,
        scale: int = 1,
    ) -> List[Tuple[str, Optional[int]]]:
        rng = self._get_rng(seed)
        requests = []

        for _ in range(num_requests):
            # 70% malloc, 30% free
            if rng.random() < 0.7:
                size = rng.randint(1, 64) * scale
                requests.append(("malloc", size))
            else:
                requests.append(("free", None))

        return requests

    def bimodal_workload(
        self,
        num_requests: int = 1000,
        seed: Optional[int] = None,
        scale: int = 1,
    ) -> List[Tuple[str, Optional[int]]]:
        rng = self._get_rng(seed)
        requests = []

        for _ in range(num_requests):
            if rng.random() < 0.7:
                if rng.random() < 0.7:
                    size = rng.randint(1, 8) * scale
                else:
                    size = rng.randint(32, 64) * scale
                requests.append(("malloc", size))
            else:
                requests.append(("free", None))

        return requests

    def adversarial_workload(self) -> List[Tuple[str, Optional[int]]]:
        return [
            ("malloc", 8),
            ("malloc", 8),
            ("malloc", 8),
            ("free", None),
            ("malloc", 4),
            ("malloc", 4),
        ]

    def scaled_adversarial_workload(
        self,
        num_requests: int = 1000,
        seed: Optional[int] = None,
        scale: int = 1,
    ) -> List[Tuple[str, Optional[int]]]:
        from core.heap import Heap

        def best_fit_idx(h, request_size):
            candidates = [(i, b) for i, b in h.free_blocks() if b.size >= request_size]
            if not candidates:
                return None
            best = min(candidates, key=lambda x: x[1].size)
            return best[0]

        rng = self._get_rng(seed)
        sim_heap = Heap(self.heap_size)
        sim_active = {}

        requests = []
        sz_small = 8 * scale
        sz_large = 56 * scale

        # Initial fill: 16 pairs of (small, large) -> 32 operations
        for _ in range(16):
            requests.append(("malloc", sz_small))
            idx_8 = best_fit_idx(sim_heap, sz_small)
            if idx_8 is not None:
                bid_8 = sim_heap.allocate(idx_8, sz_small)
                sim_active[bid_8] = sz_small

            requests.append(("malloc", sz_large))
            idx_56 = best_fit_idx(sim_heap, sz_large)
            if idx_56 is not None:
                bid_56 = sim_heap.allocate(idx_56, sz_large)
                sim_active[bid_56] = sz_large

        # Initial free: free all 16 small blocks of size sz_small -> 16 operations
        bids_to_free = [bid for bid, sz in sim_active.items() if sz == sz_small]
        for bid in bids_to_free:
            requests.append(("free", sz_small))
            sim_heap.free(bid)
            del sim_active[bid]

        # Dynamic phase: maintain 12-14 large blocks under Best Fit
        dynamic_limit = num_requests - 40
        while len(requests) < dynamic_limit:
            active_large = [bid for bid, sz in sim_active.items() if sz == sz_large]
            active_small = [bid for bid, sz in sim_active.items() if sz == sz_small]

            if rng.random() < 0.70:
                # Small block operation
                if not active_small or rng.random() < 0.50:
                    requests.append(("malloc", sz_small))
                    idx = best_fit_idx(sim_heap, sz_small)
                    if idx is not None:
                        bid = sim_heap.allocate(idx, sz_small)
                        sim_active[bid] = sz_small
                else:
                    requests.append(("free", sz_small))
                    bid = rng.choice(active_small)
                    sim_heap.free(bid)
                    del sim_active[bid]
            else:
                # Large block operation (feedback-controlled)
                if len(active_large) < 12 or (len(active_large) < 15 and rng.random() < 0.50):
                    requests.append(("malloc", sz_large))
                    idx = best_fit_idx(sim_heap, sz_large)
                    if idx is not None:
                        bid = sim_heap.allocate(idx, sz_large)
                        sim_active[bid] = sz_large
                else:
                    requests.append(("free", sz_large))
                    bid = rng.choice(active_large)
                    sim_heap.free(bid)
                    del sim_active[bid]

        # Final exposure phase: free all small blocks
        for _ in range(40):
            requests.append(("free", sz_small))

        return requests

    def adversarial_rand(
        self,
        num_requests: int = 1000,
        seed: Optional[int] = None,
        scale: int = 1,
    ) -> List[Tuple[str, Optional[int]]]:
        """
        Randomized Adversarial Workload (Batch C):
        - small ~ U{6, 10} * scale
        - large ~ U{42, 70} * scale
        - cycle length ~ Geometric(mean 8)
        - each small block freed w.p. 0.8
        - n = num_requests
        """
        from core.heap import Heap

        def best_fit_idx(h, request_size):
            candidates = [(i, b) for i, b in h.free_blocks() if b.size >= request_size]
            if not candidates:
                return None
            best = min(candidates, key=lambda x: x[1].size)
            return best[0]

        rng = self._get_rng(seed)
        sim_heap = Heap(self.heap_size)
        sim_active = {}
        requests = []

        while len(requests) < num_requests:
            # Draw cycle length from Geometric distribution with mean 8 (p = 1/8 = 0.125)
            # numpy geometric or inverse transform with rng.random()
            # Geometric(p): number of trials until first success = 1 + int(log(1-u) / log(1-p))
            u = rng.random()
            if u == 0:
                u = 1e-9
            cycle_len = max(1, min(32, int(1 + (random.Random(seed).uniform(0, 1) if False else 0)) + int(1 + (0 if u >= 1 else int(__import__('math').log(1.0 - u) / __import__('math').log(1.0 - 0.125))))))

            cycle_smalls = []
            for _ in range(cycle_len):
                if len(requests) >= num_requests:
                    break
                sz_s = rng.randint(6, 10) * scale
                requests.append(("malloc", sz_s))
                idx_s = best_fit_idx(sim_heap, sz_s)
                if idx_s is not None:
                    bid_s = sim_heap.allocate(idx_s, sz_s)
                    sim_active[bid_s] = sz_s
                    cycle_smalls.append((bid_s, sz_s))

                if len(requests) >= num_requests:
                    break
                sz_l = rng.randint(42, 70) * scale
                requests.append(("malloc", sz_l))
                idx_l = best_fit_idx(sim_heap, sz_l)
                if idx_l is not None:
                    bid_l = sim_heap.allocate(idx_l, sz_l)
                    sim_active[bid_l] = sz_l

            # Free each small block in cycle with probability 0.8
            for bid_s, sz_s in cycle_smalls:
                if len(requests) >= num_requests:
                    break
                if rng.random() < 0.80 and bid_s in sim_active:
                    requests.append(("free", sz_s))
                    sim_heap.free(bid_s)
                    del sim_active[bid_s]

        return requests[:num_requests]