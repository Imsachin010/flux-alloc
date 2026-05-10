import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from core.heap import Heap
from core.workload_generator import WorkloadGenerator
from core.allocator_strategies import best_fit
from lookahead.neural_ranker import NeuralRanker
from lookahead.lookahead_allocator import LookaheadAllocator
from lookahead.lookahead_sim import simulate_after_malloc


def collect_dataset(
    num_episode_steps: int = 2500,
    heap_size: int = 1024,
    lookahead_steps: int = 12,
    seed: int = 0,
    workload: str = "uniform",
):
    random.seed(seed)
    gen = WorkloadGenerator(heap_size)
    if workload == "bimodal":
        workload_list = gen.bimodal_workload(num_episode_steps)
    else:
        workload_list = gen.uniform_workload(num_episode_steps)
    alloc = LookaheadAllocator(heap_size=heap_size, lookahead_steps=lookahead_steps)
    heap = Heap(heap_size)
    active: list = []

    xs = []
    ys = []

    for ptr, req in enumerate(workload_list):
        if req[0] != "malloc":
            if active:
                b = active.pop(random.randrange(len(active)))
                heap.free(b)
            continue

        size = req[1]
        for idx, block in heap.free_blocks():
            if block.size < size:
                continue
            sim = simulate_after_malloc(
                heap, idx, size, workload_list, ptr, list(active), lookahead_steps
            )
            feat = alloc.build_features(heap, block, size, 0.0)
            xs.append(feat)
            ys.append([sim])

        r = best_fit(heap, size)
        if r is not None:
            active.append(r)
        # else: allocation failed; heap unchanged for a broken scenario — best_fit failed

    if not xs:
        raise RuntimeError("No training examples collected; check workload/heap settings.")

    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)
    return X, y


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workload", choices=("uniform", "bimodal"), default="uniform"
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lookahead", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "lookahead_ranker.pt"
    X, y = collect_dataset(
        num_episode_steps=args.steps,
        lookahead_steps=args.lookahead,
        seed=42,
        workload=args.workload,
    )

    model = NeuralRanker()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    n = X.size(0)
    batch = 256
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        loss_m = 0.0
        n_batches = 0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb = X[idx]
            yb = y[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_m += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                mse = criterion(model(X), y).item()
            print(f"Epoch {epoch+1}  batch_loss_avg={loss_m / max(1, n_batches):.6f}  full_mse={mse:.6f}")

    torch.save(model.state_dict(), out)
    with torch.no_grad():
        mse = criterion(model(X), y).item()
    print("Saved", out, "MSE on training set", mse)


if __name__ == "__main__":
    main()
