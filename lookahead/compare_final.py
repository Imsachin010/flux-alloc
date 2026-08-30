"""
Apples-to-apples: classic heuristics + optional MaskablePPO + Lookahead/Neural ranker.

  python -m lookahead.compare_final
  python -m lookahead.compare_final --workload bimodal
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# SB3 load compatibility (same as evaluation.compare_allocators)
import policy.custom_policy_transformer as custom_policy_transformer
sys.modules["custom_policy_transformer"] = custom_policy_transformer

from sb3_contrib import MaskablePPO

from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.heap import Heap
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv
from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench


def _run_classic_with_largest(strategy_fn, heap_size, workload, seed: int = 42):
    random.seed(seed)
    heap = Heap(heap_size)
    active = []
    failures = 0
    total_malloc = 0
    for request in workload:
        op = request[0]
        if op == "malloc":
            size = request[1]
            block_id = strategy_fn(heap, size)
            total_malloc += 1
            if block_id is None:
                failures += 1
            else:
                active.append(block_id)
        elif op == "free" and active:
            target_size = request[1] if len(request) > 1 else None
            if target_size is not None:
                candidates = []
                for b_id in active:
                    for block in heap.blocks:
                        if block.block_id == b_id and block.allocated and block.size == target_size:
                            candidates.append(b_id)
                            break
                if candidates:
                    block = random.choice(candidates)
                    heap.free(block)
                    active.remove(block)
            else:
                block = random.choice(active)
                heap.free(block)
                active.remove(block)
    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    largest = heap.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, util - frag, largest


def _resolve_ppo_path() -> Path | None:
    z = _ROOT / "assets" / "rl_direct_allocator.zip"
    d = _ROOT / "assets" / "rl_direct_allocator"
    if z.is_file():
        return z
    if d.is_dir() or d.is_file():
        return d
    return None


def _run_rl_with_largest(
    heap_size, workload
) -> tuple[float, float, float, float, float] | None:
    """Same as evaluation.compare_allocators.run_rl, plus largest free block (normalized)."""
    path = _resolve_ppo_path()
    if path is None:
        return None
    try:
        model = MaskablePPO.load(str(path))
    except Exception:
        return None
    random.seed(42)
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    state = env.reset()
    env.workload = list(workload)
    failures = 0
    total_malloc = 0
    while env.ptr < len(env.workload):
        req = env.workload[env.ptr]
        if req[0] == "malloc":
            total_malloc += 1
        action_masks = env.get_action_mask()
        action, _ = model.predict(state, action_masks=action_masks)
        state, _, done, info = env.step(int(action))
        if req[0] == "malloc" and info.get("allocation_failed", False):
            failures += 1
        if done:
            break
    h = env.heap
    util = Metrics.utilization(h)
    frag = Metrics.external_fragmentation(h)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = h.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, score, largest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classics + optional PPO + Lookahead+Neural on the same seed-42 trace."
    )
    ap.add_argument(
        "--workload",
        choices=("uniform", "bimodal", "adversarial"),
        default="uniform",
    )
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--heap-size", type=int, default=1024)
    ap.add_argument("--lookahead", type=int, default=12)
    ap.add_argument(
        "--oracle",
        choices=("best_fit", "first_fit", "next_fit"),
        default="best_fit",
        help="Oracle strategy for L3 Lookahead",
    )
    ap.add_argument(
        "--no-ppo",
        action="store_true",
        help="Do not run MaskablePPO (faster, only classics + Lookahead).",
    )
    args = ap.parse_args()

    random.seed(42)
    gen = WorkloadGenerator(args.heap_size)
    if args.workload == "uniform":
        workload = gen.uniform_workload(args.requests)
    elif args.workload == "bimodal":
        workload = gen.bimodal_workload(args.requests)
    else:
        workload = gen.scaled_adversarial_workload(args.requests)

    hsz = args.heap_size
    rows: list[tuple[str, float, float, float, float, float]] = []

    for name, fn in [
        ("First Fit", first_fit),
        ("Best Fit", best_fit),
        ("Worst Fit", worst_fit),
        ("Random Fit", random_fit),
    ]:
        u, f, fl, sc, lg = _run_classic_with_largest(fn, hsz, workload, seed=42)
        rows.append((name, u, f, fl, sc, lg))

    if not args.no_ppo:
        ppo = _run_rl_with_largest(hsz, workload)
        if ppo is not None:
            u, f, fl, sc, lg = ppo
            rows.append(("MaskablePPO Agent", u, f, fl, sc, lg))
        else:
            print("MaskablePPO: no checkpoint under assets/rl_direct_allocator. Skipping.", file=sys.stderr)

    la = run_lookahead_bench(
        hsz,
        workload,
        model_path=(_ROOT / "lookahead" / "lookahead_ranker.pt"),
        lookahead_steps=args.lookahead,
        oracle_strategy=args.oracle,
    )
    u, f, fl, sc, lg = la
    rows.append((f"Lookahead+Neural ({args.oracle})", u, f, fl, sc, lg))

    rows.sort(key=lambda x: x[4], reverse=True)

    print()
    wtype = f"{args.workload}  n={args.requests}  heap={hsz}  lookahead={args.lookahead}"
    print("== FINAL COMPARISON  ", wtype)
    print("=" * 100)
    hdr = f"{'Strategy':<22} | {'Util.':<8} | {'Frag.':<8} | {'Fail':<8} | {'Lg.fr/H':<8} | {'Score':<8}"
    print(hdr)
    print("-" * 100)
    for name, u, f, fl, sc, lg in rows:
        print(
            f"{name:<22} | {u:<8.4f} | {f:<8.4f} | {fl:<8.4f} | {lg:<8.4f} | {sc:+.4f}"
        )
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()