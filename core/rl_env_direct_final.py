import numpy as np
import random

from core.heap import Heap
from core.workload_generator import WorkloadGenerator
from core.metrics import Metrics


class DirectPlacementEnv:

    def __init__(self, heap_size=1024, episode_length=1000, max_blocks=20, lookahead=3):

        self.heap_size = heap_size
        self.episode_length = episode_length
        self.max_blocks = max_blocks
        self.lookahead = lookahead

        self.heap = Heap(heap_size)
        self.generator = WorkloadGenerator(heap_size)

        self.reset()

    def reset(self):

        self.heap.reset()
        self.workload = self.generator.uniform_workload(self.episode_length)

        self.ptr = 0
        self.active_blocks = []
        self.block_age = {}
        self.prev_largest = self.heap.largest_free_block()

        return self._get_state()

    # -------------------------
    # STATE (FINAL VERSION)
    # -------------------------
    def _get_state(self):

        util = self.heap.utilization()
        frag = Metrics.external_fragmentation(self.heap)
        largest = self.heap.largest_free_block() / self.heap_size
        num_free = self.heap.num_free_blocks() / self.max_blocks

        # current request
        if self.ptr < len(self.workload) and self.workload[self.ptr][0] == "malloc":
            req = self.workload[self.ptr][1] / self.heap_size
        else:
            req = 0

        # lookahead
        future = []
        for i in range(1, self.lookahead + 1):
            if self.ptr + i < len(self.workload) and self.workload[self.ptr + i][0] == "malloc":
                future.append(self.workload[self.ptr + i][1] / self.heap_size)
            else:
                future.append(0)

        free_blocks = self.heap.free_blocks()

        block_features = []
        mask = []

        for i, (idx, b) in enumerate(free_blocks[:self.max_blocks]):

            left = free_blocks[i-1][1].size if i > 0 else 0
            right = free_blocks[i+1][1].size if i < len(free_blocks)-1 else 0

            age = self.block_age.get(idx, 0)

            block_features.extend([
                b.size / self.heap_size,
                b.start / self.heap_size,
                left / self.heap_size,
                right / self.heap_size,
                age / 100
            ])

            mask.append(1 if b.size >= req * self.heap_size else 0)

        # padding
        while len(mask) < self.max_blocks:
            block_features.extend([0, 0, 0, 0, 0])
            mask.append(0)

        state = np.array(
            [util, frag, largest, num_free, req] + future + block_features + mask,
            dtype=np.float32
        )

        return state

    def get_action_mask(self):
        return self._get_state()[-self.max_blocks:]

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):

        reward = 0
        done = False
        allocation_failed = False
        bad_split = False

        request = self.workload[self.ptr]

        if request[0] == "malloc":

            size = request[1]
            free_blocks = self.heap.free_blocks()

            if action >= len(free_blocks):
                allocation_failed = True
            else:
                idx, block = free_blocks[action]

                if block.size < size:
                    allocation_failed = True
                else:
                    remaining = block.size - size
                    if 0 < remaining < 16:
                        bad_split = True

                    block_id = self.heap.allocate(idx, size)

                    if not block_id:
                        allocation_failed = True
                    else:
                        self.active_blocks.append(block_id)

        else:
            if self.active_blocks:
                b = random.choice(self.active_blocks)
                self.heap.free(b)
                self.active_blocks.remove(b)

        # update age
        for idx, _ in self.heap.free_blocks():
            self.block_age[idx] = self.block_age.get(idx, 0) + 1

        # -------------------------
        # REWARD (FROM DOC)
        # -------------------------
        util = self.heap.utilization()
        frag = Metrics.external_fragmentation(self.heap)

        free_blocks = self.heap.free_blocks()

        tiny_frag = sum(1 / (b.size + 1e-5) for _, b in free_blocks)

        reward += (
            1.5 * util
            - 1.2 * frag
            - 0.05 * tiny_frag
        )

        reward /= 5.0  # normalization

        # first-class: not diluted by global normalization
        if request[0] == "malloc" and allocation_failed:
            reward -= 2.0
        if request[0] == "malloc" and bad_split and not allocation_failed:
            reward -= 2.0

        largest_after = self.heap.largest_free_block()
        reward += 0.5 * (largest_after - self.prev_largest) / self.heap_size
        self.prev_largest = largest_after

        self.ptr += 1

        if self.ptr >= len(self.workload):
            done = True

        info: dict = {}
        if request[0] == "malloc":
            info["allocation_failed"] = bool(allocation_failed)
        else:
            info["allocation_failed"] = False

        return self._get_state(), reward, done, info