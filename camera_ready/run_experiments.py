"""
FluxAlloc Camera-Ready Experiment Runner (Batches A, B, C, D)
Outputs all run records to camera_ready/results.csv and detailed execution logs to camera_ready/logs/
"""

import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# SB3 load compatibility
import policy.custom_policy_transformer as custom_policy_transformer
sys.modules["custom_policy_transformer"] = custom_policy_transformer

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit, next_fit
from core.heap import Heap
from core.metrics import Metrics
from core.rl_env_direct_final import DirectPlacementEnv
from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench
from lookahead.lookahead_allocator import LookaheadAllocator
from lookahead.neural_ranker import NeuralRanker


def resolve_ppo_path() -> Optional[Path]:
    z = _ROOT / "assets" / "rl_direct_allocator.zip"
    d = _ROOT / "assets" / "rl_direct_allocator"
    if z.is_file():
        return z
    if d.is_dir() or d.is_file():
        return d
    return None


def run_classic_strategy(
    strategy_fn,
    heap_size: int,
    workload: list,
    seed: int,
) -> tuple[float, float, float, float, float]:
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
    score = util - frag
    largest = heap.largest_free_block() / float(heap_size)
    return float(util), float(frag), float(fail_rate), float(largest), float(score)


def run_ppo_strategy(
    heap_size: int,
    workload: list,
    seed: int,
) -> Optional[tuple[float, float, float, float, float]]:
    path = resolve_ppo_path()
    if path is None:
        return None
    try:
        model = MaskablePPO.load(str(path))
    except Exception as e:
        print(f"Failed to load PPO model: {e}")
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
    return float(util), float(frag), float(fail_rate), float(largest), float(score)


def run_lookahead_strategy(
    heap_size: int,
    workload: list,
    seed: int,
    lookahead_steps: int = 12,
    neural_weight: float = 0.2,
    sim_weight: float = 0.8,
    oracle_strategy: str = "best_fit",
    free_policy: str = "FIFO",
    mismatched_hypothesis: bool = False,
    scale: int = 1,
) -> tuple[float, float, float, float, float]:
    model_path = _ROOT / "lookahead" / "lookahead_ranker.pt"
    util, frag, fail_rate, score, largest = run_lookahead_bench(
        heap_size=heap_size,
        workload=workload,
        model_path=model_path,
        lookahead_steps=lookahead_steps,
        neural_weight=neural_weight,
        sim_weight=sim_weight,
        seed=seed,
        oracle_strategy=oracle_strategy,
        free_policy=free_policy,
        mismatched_hypothesis=mismatched_hypothesis,
        scale=scale,
    )
    return float(util), float(frag), float(fail_rate), float(largest), float(score)


def measure_latencies(heap_size: int, workload: list, seed: int):
    # 1. Best Fit Latency
    random.seed(seed)
    heap_bf = Heap(heap_size)
    active_bf = []
    bf_latencies = []
    for req in workload:
        if req[0] == "malloc":
            sz = req[1]
            t0 = time.perf_counter()
            bid = best_fit(heap_bf, sz)
            t1 = time.perf_counter()
            bf_latencies.append((t1 - t0) * 1000.0)
            if bid is not None:
                active_bf.append(bid)
        elif req[0] == "free" and active_bf:
            b = random.choice(active_bf)
            heap_bf.free(b)
            active_bf.remove(b)

    # 2. Lookahead MLP-only Latency (single forward pass)
    model = NeuralRanker()
    model_path = _ROOT / "lookahead" / "lookahead_ranker.pt"
    if model_path.is_file():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    x = torch.randn(1, 10, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(20):
            model(x)
    mlp_latencies = []
    for _ in range(len(bf_latencies)):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        t1 = time.perf_counter()
        mlp_latencies.append((t1 - t0) * 1000.0)

    # 3. Lookahead Full Planning Latency
    random.seed(seed)
    heap_fa = Heap(heap_size)
    active_fa = []
    alloc = LookaheadAllocator(
        model_path=model_path if model_path.is_file() else None,
        heap_size=heap_size,
        lookahead_steps=12,
    )
    fa_latencies = []
    for ptr, req in enumerate(workload):
        if req[0] == "malloc":
            sz = req[1]
            t0 = time.perf_counter()
            idx = alloc.choose_block(heap_fa, sz, workload, ptr, active_fa)
            t1 = time.perf_counter()
            fa_latencies.append((t1 - t0) * 1000.0)
            if idx is not None:
                bid = heap_fa.allocate(idx, sz)
                if bid is not None:
                    active_fa.append(bid)
        elif req[0] == "free" and active_fa:
            b = random.choice(active_fa)
            heap_fa.free(b)
            active_fa.remove(b)

    return {
        "Best Fit": (float(np.mean(bf_latencies)), float(np.median(bf_latencies)), float(np.max(bf_latencies))),
        "FluxAlloc MLP-only": (float(np.mean(mlp_latencies)), float(np.median(mlp_latencies)), float(np.max(mlp_latencies))),
        "FluxAlloc Full Planning": (float(np.mean(fa_latencies)), float(np.median(fa_latencies)), float(np.max(fa_latencies))),
    }


def main():
    logs_dir = _ROOT / "camera_ready" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_csv = _ROOT / "camera_ready" / "results.csv"

    # Seed Sets
    seed_10 = [0, 1, 2, 3, 4, 7, 10, 18, 42, 123]
    seed_5 = [0, 7, 10, 18, 42]

    results_rows = []

    def record_row(config, workload_name, seed, strategy, util, frag, fail, lgfr, score, lat=(None, None, None)):
        hit = 1.0 - fail
        miss = fail
        lat_mean, lat_med, lat_max = lat
        row = {
            "config": config,
            "workload": workload_name,
            "seed": seed,
            "strategy": strategy,
            "util": f"{util:.4f}",
            "frag": f"{frag:.4f}",
            "fail": f"{fail:.4f}",
            "lgfr": f"{lgfr:.4f}",
            "score": f"{score:+.4f}",
            "hit": f"{hit:.4f}",
            "miss": f"{miss:.4f}",
            "lat_mean_ms": f"{lat_mean:.4f}" if lat_mean is not None else "",
            "lat_med_ms": f"{lat_med:.4f}" if lat_med is not None else "",
            "lat_max_ms": f"{lat_max:.4f}" if lat_max is not None else "",
        }
        results_rows.append(row)
        log_file = logs_dir / f"{config}_{workload_name}_s{seed}_{strategy.replace(' ', '_').replace('+', '_')}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
        print(f"[{config}] {workload_name} s{seed} | {strategy:<20} -> Util: {row['util']}, Frag: {row['frag']}, Fail: {row['fail']}, Score: {row['score']}")

    print("=" * 80)
    print("STARTING FLUXALLOC EXPERIMENTAL REGENERATION")
    print("=" * 80)

    gen_1024 = WorkloadGenerator(1024)

    # =========================================================================
    # BATCH A: CORE REGENERATION
    # =========================================================================
    print("\n>>> BATCH A: Core Regeneration <<<")

    # A-unif-s42
    w_unif_42 = gen_1024.uniform_workload(1000, seed=42)
    for name, fn in [("Best Fit", best_fit), ("Random Fit", random_fit), ("First Fit", first_fit), ("Worst Fit", worst_fit)]:
        u, f, fl, lg, sc = run_classic_strategy(fn, 1024, w_unif_42, seed=42)
        record_row("A-unif-s42", "uniform", 42, name, u, f, fl, lg, sc)
    ppo_res = run_ppo_strategy(1024, w_unif_42, seed=42)
    if ppo_res:
        u, f, fl, lg, sc = ppo_res
        record_row("A-unif-s42", "uniform", 42, "Baseline PPO", u, f, fl, lg, sc)
    u, f, fl, lg, sc = run_lookahead_strategy(1024, w_unif_42, seed=42)
    record_row("A-unif-s42", "uniform", 42, "FluxAlloc", u, f, fl, lg, sc)

    # A-bim-s42
    w_bim_42 = gen_1024.bimodal_workload(1000, seed=42)
    for name, fn in [("Best Fit", best_fit), ("Random Fit", random_fit), ("First Fit", first_fit), ("Worst Fit", worst_fit)]:
        u, f, fl, lg, sc = run_classic_strategy(fn, 1024, w_bim_42, seed=42)
        record_row("A-bim-s42", "bimodal", 42, name, u, f, fl, lg, sc)
    ppo_res = run_ppo_strategy(1024, w_bim_42, seed=42)
    if ppo_res:
        u, f, fl, lg, sc = ppo_res
        record_row("A-bim-s42", "bimodal", 42, "Baseline PPO", u, f, fl, lg, sc)
    u, f, fl, lg, sc = run_lookahead_strategy(1024, w_bim_42, seed=42)
    record_row("A-bim-s42", "bimodal", 42, "FluxAlloc", u, f, fl, lg, sc)

    # A-bim-10
    for s in seed_10:
        w_bim_s = gen_1024.bimodal_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 1024, w_bim_s, seed=s)
        record_row("A-bim-10", "bimodal", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_bim_s, seed=s)
        record_row("A-bim-10", "bimodal", s, "FluxAlloc", u, f, fl, lg, sc)

    # A-unif-10
    for s in seed_10:
        w_unif_s = gen_1024.uniform_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 1024, w_unif_s, seed=s)
        record_row("A-unif-10", "uniform", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_unif_s, seed=s)
        record_row("A-unif-10", "uniform", s, "FluxAlloc", u, f, fl, lg, sc)

    # A-adv-canonical
    w_adv_42 = gen_1024.scaled_adversarial_workload(1000, seed=42)
    for name, fn in [("Best Fit", best_fit), ("Random Fit", random_fit), ("First Fit", first_fit), ("Worst Fit", worst_fit)]:
        u, f, fl, lg, sc = run_classic_strategy(fn, 1024, w_adv_42, seed=42)
        record_row("A-adv-canonical", "adversarial", 42, name, u, f, fl, lg, sc)
    for oracle_name in ["best_fit", "next_fit", "first_fit"]:
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_adv_42, seed=42, oracle_strategy=oracle_name)
        record_row("A-adv-canonical", "adversarial", 42, f"FluxAlloc (oracle={oracle_name})", u, f, fl, lg, sc)

    # A-adv-10
    for s in seed_10:
        w_adv_s = gen_1024.scaled_adversarial_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 1024, w_adv_s, seed=s)
        record_row("A-adv-10", "adversarial", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_adv_s, seed=s, oracle_strategy="best_fit")
        record_row("A-adv-10", "adversarial", s, "FluxAlloc", u, f, fl, lg, sc)

    # A-abl (Ablations on Bimodal Seed 42)
    w_abl = gen_1024.bimodal_workload(1000, seed=42)
    # Blend: alpha=0.0 (sim only), alpha=1.0 (neural only)
    u, f, fl, lg, sc = run_lookahead_strategy(1024, w_abl, seed=42, lookahead_steps=12, neural_weight=0.0, sim_weight=1.0)
    record_row("A-abl", "bimodal", 42, "FluxAlloc (alpha=0.0, k=12)", u, f, fl, lg, sc)
    u, f, fl, lg, sc = run_lookahead_strategy(1024, w_abl, seed=42, lookahead_steps=12, neural_weight=1.0, sim_weight=0.0)
    record_row("A-abl", "bimodal", 42, "FluxAlloc (alpha=1.0, k=12)", u, f, fl, lg, sc)
    # Depth k in {4, 8, 16} with alpha=0.2
    for k_val in [4, 8, 16]:
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_abl, seed=42, lookahead_steps=k_val, neural_weight=0.2, sim_weight=0.8)
        record_row("A-abl", "bimodal", 42, f"FluxAlloc (k={k_val}, alpha=0.2)", u, f, fl, lg, sc)

    # A-ppo-eval: PPO Policy Distribution Analysis
    ppo_path = resolve_ppo_path()
    if ppo_path:
        ppo_model = MaskablePPO.load(str(ppo_path))
        env_ppo = DirectPlacementEnv(heap_size=1024, episode_length=5000)
        state_p = env_ppo.reset()
        gen_eval = WorkloadGenerator(1024)
        env_ppo.workload = gen_eval.uniform_workload(5000, seed=42)
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0} # 0: First Fit, 1: Best Fit, 2: Worst Fit, 3: Random Fit
        while env_ppo.ptr < len(env_ppo.workload):
            masks = env_ppo.get_action_mask()
            act_p, _ = ppo_model.predict(state_p, action_masks=masks)
            if env_ppo.workload[env_ppo.ptr][0] == "malloc":
                action_counts[int(act_p) % 4] = action_counts.get(int(act_p) % 4, 0) + 1
            state_p, _, done_p, _ = env_ppo.step(int(act_p))
            if done_p:
                break
        print(f"PPO Policy Evaluation Action Counts (5000 steps): {action_counts}")

    # A-latency
    w_lat = gen_1024.uniform_workload(500, seed=42)
    lat_dict = measure_latencies(1024, w_lat, seed=42)
    for strat_name, lat_stats in lat_dict.items():
        if strat_name == "Best Fit":
            u, f, fl, lg, sc = run_classic_strategy(best_fit, 1024, w_lat, seed=42)
        elif strat_name == "FluxAlloc MLP-only":
            u, f, fl, lg, sc = run_lookahead_strategy(1024, w_lat, seed=42, lookahead_steps=0, neural_weight=1.0, sim_weight=0.0)
        else:
            u, f, fl, lg, sc = run_lookahead_strategy(1024, w_lat, seed=42, lookahead_steps=12, neural_weight=0.2, sim_weight=0.8)
        record_row("A-latency", "uniform", 42, strat_name, u, f, fl, lg, sc, lat=lat_stats)

    # =========================================================================
    # BATCH B: SCALE & REGIME
    # =========================================================================
    print("\n>>> BATCH B: Scale & Regime <<<")

    # B-x64: heap=65536, sizes * 64, bimodal, Seed-5
    gen_65k = WorkloadGenerator(65536)
    for s in seed_5:
        w_b64 = gen_65k.bimodal_workload(1000, seed=s, scale=64)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 65536, w_b64, seed=s)
        record_row("B-x64", "bimodal", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(65536, w_b64, seed=s, scale=64)
        record_row("B-x64", "bimodal", s, "FluxAlloc", u, f, fl, lg, sc)

    # B-h2: heap=2048, sizes unchanged, bimodal, Seed-5
    gen_2048 = WorkloadGenerator(2048)
    for s in seed_5:
        w_h2 = gen_2048.bimodal_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 2048, w_h2, seed=s)
        record_row("B-h2", "bimodal", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(2048, w_h2, seed=s)
        record_row("B-h2", "bimodal", s, "FluxAlloc", u, f, fl, lg, sc)

    # B-h4: heap=4096, sizes unchanged, bimodal, Seed-5
    gen_4096 = WorkloadGenerator(4096)
    for s in seed_5:
        w_h4 = gen_4096.bimodal_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 4096, w_h4, seed=s)
        record_row("B-h4", "bimodal", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(4096, w_h4, seed=s)
        record_row("B-h4", "bimodal", s, "FluxAlloc", u, f, fl, lg, sc)

    # B-unif-h4: heap=4096, uniform, Seed-5
    for s in seed_5:
        w_uh4 = gen_4096.uniform_workload(1000, seed=s)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 4096, w_uh4, seed=s)
        record_row("B-unif-h4", "uniform", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(4096, w_uh4, seed=s)
        record_row("B-unif-h4", "uniform", s, "FluxAlloc", u, f, fl, lg, sc)

    # =========================================================================
    # BATCH C: ADVERSARIAL ROBUSTNESS & MISMATCHED ORACLE
    # =========================================================================
    print("\n>>> BATCH C: Adversarial Robustness & Mismatched Oracle <<<")

    for s in seed_10:
        w_rand = gen_1024.adversarial_rand(1000, seed=s)
        # C-rand: BF and FA (matched hypothesis)
        u, f, fl, lg, sc = run_classic_strategy(best_fit, 1024, w_rand, seed=s)
        record_row("C-rand", "adversarial_rand", s, "Best Fit", u, f, fl, lg, sc)
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_rand, seed=s, mismatched_hypothesis=False)
        record_row("C-rand", "adversarial_rand", s, "FluxAlloc (matched)", u, f, fl, lg, sc)

        # C-mism: FA with mismatched hypothesis (perturbing peeked future to canonical 8/56)
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_rand, seed=s, mismatched_hypothesis=True)
        record_row("C-mism", "adversarial_rand", s, "FluxAlloc (mismatched)", u, f, fl, lg, sc)

    # =========================================================================
    # BATCH D: FREE-POLICY SENSITIVITY
    # =========================================================================
    print("\n>>> BATCH D: Free-Policy Sensitivity <<<")

    # Bimodal Seed 42: LIFO and random (FIFO is already in A-bim-s42)
    w_d_bim = gen_1024.bimodal_workload(1000, seed=42)
    for f_pol in ["LIFO", "random"]:
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_d_bim, seed=42, free_policy=f_pol)
        record_row("D-freepol", "bimodal", 42, f"FluxAlloc (free={f_pol})", u, f, fl, lg, sc)

    # Adversarial Seed 42: LIFO and random (FIFO is already in A-adv-canonical)
    w_d_adv = gen_1024.scaled_adversarial_workload(1000, seed=42)
    for f_pol in ["LIFO", "random"]:
        u, f, fl, lg, sc = run_lookahead_strategy(1024, w_d_adv, seed=42, free_policy=f_pol)
        record_row("D-freepol", "adversarial", 42, f"FluxAlloc (free={f_pol})", u, f, fl, lg, sc)

    # Write results.csv
    fieldnames = [
        "config", "workload", "seed", "strategy", "util", "frag", "fail", "lgfr",
        "score", "hit", "miss", "lat_mean_ms", "lat_med_ms", "lat_max_ms"
    ]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_rows:
            writer.writerow(r)

    print("\n" + "=" * 80)
    print(f"EXPERIMENTS COMPLETED! Results saved to {results_csv}")
    print(f"Total experiment runs recorded: {len(results_rows)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
