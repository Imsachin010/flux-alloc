import random
import sys
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# SB3 load compatibility
import policy.custom_policy_transformer as custom_policy_transformer
sys.modules["custom_policy_transformer"] = custom_policy_transformer

from sb3_contrib import MaskablePPO

from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit
from core.heap import Heap
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv
from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench

def _run_classic_with_largest(strategy_fn, heap_size, workload, seed: int):
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
            block = random.choice(active)
            heap.free(block)
            active.remove(block)
    util = Metrics.utilization(heap)
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0.0
    score = util - frag
    largest = heap.largest_free_block() / float(heap_size)
    return util, frag, fail_rate, score, largest

def _resolve_ppo_path() -> Path | None:
    z = _ROOT / "assets" / "rl_direct_allocator.zip"
    d = _ROOT / "assets" / "rl_direct_allocator"
    if z.is_file():
        return z
    if d.is_dir() or d.is_file():
        return d
    return None

def _run_rl_with_largest(heap_size, workload, seed: int):
    path = _resolve_ppo_path()
    if path is None:
        return None
    try:
        model = MaskablePPO.load(str(path))
    except Exception:
        return None
    random.seed(seed)
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

def main():
    seeds = [42, 10, 18]
    heap_size = 1024
    num_requests = 1000
    lookahead_steps = 12

    # Strategy -> list of tuples (util, frag, fail, score, largest)
    strategies_results = {
        "Best Fit": [],
        "Lookahead+Neural": [],
        "MaskablePPO Agent": [],
        "First Fit": [],
        "Random Fit": [],
        "Worst Fit": []
    }

    # Run for each seed
    for seed in seeds:
        print(f"--- Running seed {seed} ---")
        random.seed(seed)
        gen = WorkloadGenerator(heap_size)
        workload = gen.bimodal_workload(num_requests)

        # Classics
        for name, fn in [
            ("First Fit", first_fit),
            ("Best Fit", best_fit),
            ("Worst Fit", worst_fit),
            ("Random Fit", random_fit),
        ]:
            u, f, fl, sc, lg = _run_classic_with_largest(fn, heap_size, workload, seed=seed)
            strategies_results[name].append((u, f, fl, sc, lg))
            print(f"  {name:<18} | Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f} | Lg.fr: {lg:.4f}")

        # RL
        ppo = _run_rl_with_largest(heap_size, workload, seed=seed)
        if ppo is not None:
            u, f, fl, sc, lg = ppo
            strategies_results["MaskablePPO Agent"].append((u, f, fl, sc, lg))
            print(f"  {'MaskablePPO Agent':<18} | Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f} | Lg.fr: {lg:.4f}")
        else:
            print("  MaskablePPO Agent not run (no model).")

        # Lookahead
        la = run_lookahead_bench(
            heap_size,
            workload,
            model_path=(_ROOT / "lookahead" / "lookahead_ranker.pt"),
            lookahead_steps=lookahead_steps,
            seed=seed
        )
        u, f, fl, sc, lg = la
        strategies_results["Lookahead+Neural"].append((u, f, fl, sc, lg))
        print(f"  {'Lookahead+Neural':<18} | Util: {u:.4f} | Frag: {f:.4f} | Fail: {fl:.4f} | Score: {sc:+.4f} | Lg.fr: {lg:.4f}")
        print()

    # Now calculate mean and std for each strategy
    print("=" * 115)
    print(f"SUMMARY RESULTS OVER SEEDS {seeds} (BIMODAL WORKLOAD)")
    print("=" * 115)
    hdr = f"{'Strategy':<20} | {'Util.':<18} | {'Frag.':<18} | {'Fail':<18} | {'Lg.fr/H':<18} | {'Score':<18}"
    print(hdr)
    print("-" * 115)

    # Sort strategies by mean Score (descending)
    sorted_strategies = []
    for name, res_list in strategies_results.items():
        if not res_list:
            continue
        arr = np.array(res_list) # shape: (len(seeds), 5)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        sorted_strategies.append((name, means, stds))

    sorted_strategies.sort(key=lambda x: x[1][3], reverse=True)

    for name, means, stds in sorted_strategies:
        u_str = f"{means[0]:.4f} \u00b1 {stds[0]:.4f}"
        f_str = f"{means[1]:.4f} \u00b1 {stds[1]:.4f}"
        fl_str = f"{means[2]:.4f} \u00b1 {stds[2]:.4f}"
        sc_str = f"{means[3]:+.4f} \u00b1 {stds[3]:.4f}"
        lg_str = f"{means[4]:.4f} \u00b1 {stds[4]:.4f}"
        print(f"{name:<20} | {u_str:<18} | {f_str:<18} | {fl_str:<18} | {lg_str:<18} | {sc_str:<18}")
    print("=" * 115)

if __name__ == "__main__":
    main()
