import random


class WorkloadGenerator:

    def __init__(self, heap_size):
        self.heap_size = heap_size


    def uniform_workload(self, num_requests=1000):

        requests = []

        for _ in range(num_requests):

            # 70% malloc, 30% free
            if random.random() < 0.7:
                size = random.randint(1, 64)
                requests.append(("malloc", size))
            else:
                requests.append(("free", None))

        return requests


    def bimodal_workload(self, num_requests=1000):

        requests = []

        for _ in range(num_requests):

            if random.random() < 0.7:

                if random.random() < 0.7:
                    size = random.randint(1, 8)
                else:
                    size = random.randint(32, 64)

                requests.append(("malloc", size))

            else:
                requests.append(("free", None))

        return requests


    def adversarial_workload(self):

        return [
            ("malloc", 8),
            ("malloc", 8),
            ("malloc", 8),
            ("free", None),
            ("malloc", 4),
            ("malloc", 4),
        ]

    def scaled_adversarial_workload(self, num_requests=1000):
        from core.heap import Heap

        def best_fit_idx(h, request_size):
            candidates = [(i, b) for i, b in h.free_blocks() if b.size >= request_size]
            if not candidates:
                return None
            best = min(candidates, key=lambda x: x[1].size)
            return best[0]

        sim_heap = Heap(self.heap_size)
        sim_active = {}

        requests = []
        random.seed(42)

        # Initial fill: 16 pairs of (8, 56) -> 32 operations
        for _ in range(16):
            requests.append(("malloc", 8))
            idx_8 = best_fit_idx(sim_heap, 8)
            if idx_8 is not None:
                bid_8 = sim_heap.allocate(idx_8, 8)
                sim_active[bid_8] = 8

            requests.append(("malloc", 56))
            idx_56 = best_fit_idx(sim_heap, 56)
            if idx_56 is not None:
                bid_56 = sim_heap.allocate(idx_56, 56)
                sim_active[bid_56] = 56

        # Initial free: free all 16 small blocks of size 8 -> 16 operations
        bids_to_free = [bid for bid, sz in sim_active.items() if sz == 8]
        for bid in bids_to_free:
            requests.append(("free", 8))
            sim_heap.free(bid)
            del sim_active[bid]

        # Dynamic phase: maintain 12-14 large blocks under Best Fit
        dynamic_limit = num_requests - 40
        while len(requests) < dynamic_limit:
            active_56 = [bid for bid, sz in sim_active.items() if sz == 56]
            active_8 = [bid for bid, sz in sim_active.items() if sz == 8]

            if random.random() < 0.70:
                # Small block operation
                if not active_8 or random.random() < 0.50:
                    requests.append(("malloc", 8))
                    idx = best_fit_idx(sim_heap, 8)
                    if idx is not None:
                        bid = sim_heap.allocate(idx, 8)
                        sim_active[bid] = 8
                else:
                    requests.append(("free", 8))
                    bid = random.choice(active_8)
                    sim_heap.free(bid)
                    del sim_active[bid]
            else:
                # Large block operation (feedback-controlled)
                if len(active_56) < 12 or (len(active_56) < 15 and random.random() < 0.50):
                    requests.append(("malloc", 56))
                    idx = best_fit_idx(sim_heap, 56)
                    if idx is not None:
                        bid = sim_heap.allocate(idx, 56)
                        sim_active[bid] = 56
                else:
                    requests.append(("free", 56))
                    bid = random.choice(active_56)
                    sim_heap.free(bid)
                    del sim_active[bid]

        # Final exposure phase: free all size-8 blocks
        for _ in range(40):
            requests.append(("free", 8))

        return requests