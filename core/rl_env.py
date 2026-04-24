import numpy as np
import random

from core.heap import Heap
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.workload_generator import WorkloadGenerator
from core.metrics import Metrics


class MemoryEnv:

    def __init__(self, heap_size=1024, episode_length=1000):

        self.heap_size = heap_size
        self.episode_length = episode_length

        self.heap = Heap(heap_size)
        self.generator = WorkloadGenerator(heap_size)

        self.strategies = [
            first_fit,
            best_fit,
            worst_fit,
            random_fit
        ]

        self.reset()

    # ------------------------
    # Reset environment
    # ------------------------
    def reset(self):

        self.heap.reset()

        self.workload = self.generator.uniform_workload(self.episode_length)

        self.active_blocks = []
        self.ptr = 0

        return self._get_state()

    # ------------------------
    # State representation
    # ------------------------
    def _get_state(self):

        util = self.heap.utilization()
        frag = Metrics.external_fragmentation(self.heap)

        largest = self.heap.largest_free_block() / self.heap_size
        free_blocks = self.heap.num_free_blocks() / 100

        if self.ptr < len(self.workload) and self.workload[self.ptr][0] == "malloc":
            req_size = self.workload[self.ptr][1] / self.heap_size
        else:
            req_size = 0

        return np.array([
            util,
            frag,
            largest,
            free_blocks,
            req_size
        ], dtype=np.float32)

    # ------------------------
    # Step function
    # ------------------------
    def step(self, action):

        done = False
        reward = 0

        request = self.workload[self.ptr]

        if request[0] == "malloc":

            size = request[1]

            strategy = self.strategies[action]

            block_id = strategy(self.heap, size)

            if block_id is None:
                reward -= 2
            else:
                self.active_blocks.append(block_id)

        else:

            if self.active_blocks:
                block = random.choice(self.active_blocks)
                self.heap.free(block)
                self.active_blocks.remove(block)

        util = self.heap.utilization()
        frag = Metrics.external_fragmentation(self.heap)

        reward += util - frag

        self.ptr += 1

        if self.ptr >= self.episode_length:
            done = True

        next_state = self._get_state()

        return next_state, reward, done, {}