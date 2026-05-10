import torch
import numpy as np

from core.metrics import Metrics
from lookahead.neural_ranker import NeuralRanker
from lookahead.lookahead_sim import simulate_after_malloc

_DEFAULT_LOOKAHEAD = 12
_NEURAL_W = 0.2
_SIM_W = 0.8


class LookaheadAllocator:
    def __init__(
        self,
        model_path=None,
        heap_size=1024,
        lookahead_steps: int = _DEFAULT_LOOKAHEAD,
        neural_weight: float = _NEURAL_W,
        sim_weight: float = _SIM_W,
    ):

        self.heap_size = heap_size
        self.lookahead_steps = lookahead_steps
        self.neural_weight = neural_weight
        self.sim_weight = sim_weight
        self.model = NeuralRanker()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()

    def build_features(self, heap, block, request_size, sim_hint: float = 0.0):
        util = heap.utilization()
        frag = Metrics.external_fragmentation(heap)

        largest = heap.largest_free_block() / self.heap_size
        num_free = len(heap.free_blocks()) / 20

        block_size = block.size / self.heap_size
        block_pos = block.start / self.heap_size

        remaining = (block.size - request_size) / self.heap_size
        horiz = min(1.0, self.lookahead_steps / 100.0)
        sh = min(1.0, max(0.0, float(sim_hint)))

        features = np.array(
            [
                util,
                frag,
                largest,
                num_free,
                request_size / self.heap_size,
                block_size,
                block_pos,
                remaining,
                sh,
                horiz,
            ],
            dtype=np.float32,
        )

        return torch.tensor(features)

    def choose_block(
        self,
        heap,
        request_size,
        workload=None,
        request_index: int = 0,
        active_block_ids=None,
    ):

        free_blocks = heap.free_blocks()
        active_block_ids = active_block_ids or []

        use_lookahead = (
            workload is not None
            and self.lookahead_steps > 0
            and 0 <= request_index < len(workload)
        )

        candidates = []

        for idx, block in free_blocks:
            if block.size < request_size:
                continue

            features = self.build_features(heap, block, request_size)
            with torch.no_grad():
                nscore = self.model(features).item()

            if use_lookahead:
                sim = simulate_after_malloc(
                    heap,
                    idx,
                    request_size,
                    workload,
                    request_index,
                    active_block_ids,
                    self.lookahead_steps,
                )
                # Blend: simulation is the primary "future-safe" signal.
                tscore = self.neural_weight * nscore + self.sim_weight * sim
            else:
                tscore = nscore

            candidates.append((tscore, idx))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
