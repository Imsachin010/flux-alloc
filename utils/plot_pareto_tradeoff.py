import os
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.workload_generator import WorkloadGenerator
from core.allocator_strategies import first_fit, best_fit, worst_fit, random_fit

# Use the exact benchmark loops from compare_final to guarantee correctness
from lookahead.compare_final import _run_classic_with_largest, _run_rl_with_largest
from lookahead.bench import run_lookahead_bench

def main():
    heap_size = 1024
    num_requests = 1000
    seeds = list(range(42, 52)) # 10 seeds to match original density

    strategies = {
        "First Fit": first_fit,
        "Best Fit": best_fit,
        "Worst Fit": worst_fit,
        "Random Fit": random_fit
    }

    data = {name: {'util': [], 'frag': []} for name in strategies}
    data["MaskablePPO"] = {'util': [], 'frag': []}
    data["Lookahead+Neural"] = {'util': [], 'frag': []}

    print("Running simulations to generate tradeoff graph...")
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        
        generator = WorkloadGenerator(heap_size)
        workload = generator.uniform_workload(num_requests)

        for name, fn in strategies.items():
            u, f, _, _, _ = _run_classic_with_largest(fn, heap_size, workload, seed=seed)
            data[name]['util'].append(u)
            data[name]['frag'].append(f)
            
        rl_res = _run_rl_with_largest(heap_size, workload)
        if rl_res:
            u, f, _, _, _ = rl_res
            data["MaskablePPO"]['util'].append(u)
            data["MaskablePPO"]['frag'].append(f)
            
        try:
            u, f, _, _, _ = run_lookahead_bench(heap_size, workload, model_path=_ROOT / "lookahead" / "lookahead_ranker.pt", lookahead_steps=12)
            data["Lookahead+Neural"]['util'].append(u)
            data["Lookahead+Neural"]['frag'].append(f)
        except Exception as e:
            print(f"Lookahead failed: {e}")

    # Plotting Logic
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.dpi': 300
    })

    plt.figure(figsize=(9, 6))
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    all_points = []

    for i, (name, metrics) in enumerate(data.items()):
        if not metrics['util']: continue
        plt.scatter(metrics['util'], metrics['frag'], label=name, color=colors[i % len(colors)], alpha=0.8, s=60, edgecolor='black', linewidth=0.5)
        for u, f in zip(metrics['util'], metrics['frag']):
            all_points.append((u, f, name))

    # Calculate strict Pareto Frontier (Maximize Util, Minimize Frag)
    pareto_front = []
    for p in all_points:
        is_dominated = False
        for other in all_points:
            if other == p: continue
            if other[0] >= p[0] and other[1] <= p[1] and (other[0] > p[0] or other[1] < p[1]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(p)
            
    pareto_front.sort(key=lambda x: x[0])
    
    if pareto_front:
        p_util = [p[0] for p in pareto_front]
        p_frag = [p[1] for p in pareto_front]
        
        # Calculate Hypervolume using reference point Util=0, Frag=1 
        # (Area bounded by the pareto curve where higher util and lower frag is better)
        hypervolume = 0.0
        prev_u = 0.0
        for u, f in zip(p_util, p_frag):
            # width = current util - previous util
            # height = 1.0 - current frag (since lower frag is better)
            hypervolume += (u - prev_u) * (1.0 - f)
            prev_u = u
            
        plt.plot(p_util, p_frag, 'k--', label=f'Pareto Frontier (HV: {hypervolume:.3f})', linewidth=2, zorder=4)
        plt.scatter(p_util, p_frag, color='black', marker='*', s=200, zorder=5)

    plt.xlabel("Utilization (Higher is Better)")
    plt.ylabel("Fragmentation (Lower is Better)")
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/pareto_tradeoff.png', bbox_inches='tight')
    print("Saved fixed Pareto plot with Hypervolume to assets/pareto_tradeoff.png")

if __name__ == "__main__":
    main()
