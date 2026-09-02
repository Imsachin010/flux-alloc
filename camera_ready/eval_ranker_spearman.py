import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

from core.heap import Heap
from core.workload_generator import WorkloadGenerator
from core.allocator_strategies import best_fit
from lookahead.neural_ranker import NeuralRanker
from lookahead.lookahead_allocator import LookaheadAllocator
from lookahead.lookahead_sim import simulate_after_malloc


def compute_spearman_ranker():
    model_path = _ROOT / "lookahead" / "lookahead_ranker.pt"
    model = NeuralRanker()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Held-out trace generation: seed 999 (completely held-out from eval seeds {0, 1, 2, 3, 4, 7, 10, 18, 42, 123})
    gen = WorkloadGenerator(1024)
    workload = gen.uniform_workload(1000, seed=999)

    alloc = LookaheadAllocator(heap_size=1024, lookahead_steps=12)
    heap = Heap(1024)
    active = []

    predictions = []
    actual_sim_scores = []

    for ptr, req in enumerate(workload):
        if req[0] == "malloc":
            size = req[1]
            for idx, block in heap.free_blocks():
                if block.size < size:
                    continue
                sim = simulate_after_malloc(
                    heap, idx, size, workload, ptr, list(active), 12
                )
                feat = alloc.build_features(heap, block, size, 0.0)
                with torch.no_grad():
                    pred = model(feat).item()
                predictions.append(pred)
                actual_sim_scores.append(sim)

            r = best_fit(heap, size)
            if r is not None:
                active.append(r)
        else:
            if active:
                b = active.pop(0)
                heap.free(b)

    rho, pval = spearmanr(predictions, actual_sim_scores)
    print(f"Held-out Candidate Pairs Evaluated: {len(predictions)}")
    print(f"Spearman rho: {rho:.4f} (p-value: {pval:.4e})")

    # Architecture facts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")

    return rho, pval, total_params


if __name__ == "__main__":
    compute_spearman_ranker()
