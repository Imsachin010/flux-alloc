import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Academic style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif',
        'figure.dpi': 300
    })

    methods = ["Best Fit", "Random Fit", "First Fit", "Lookahead\n+Neural", "Maskable\nPPO", "Worst Fit"]
    
    # Uniform
    score_u = [0.1813, 0.0937, 0.0333, 0.0153, -0.0463, -0.1846]
    # Bimodal
    score_b = [0.2046, 0.0988, 0.0707, 0.2266, 0.0745, -0.0862]

    # Reorder methods: Lookahead, Best, PPO, First, Random, Worst
    order = [3, 0, 4, 2, 1, 5] 
    methods = [methods[i] for i in order]
    score_u = [score_u[i] for i in order]
    score_b = [score_b[i] for i in order]

    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Muted academic colors
    color_u = '#5D8AA8' # Air Force Blue
    color_b = '#A52A2A' # Brown/Red
    
    rects1 = ax.bar(x - width/2, score_u, width, label='Uniform Workload', color=color_u, edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, score_b, width, label='Bimodal Workload', color=color_b, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Efficiency Score (Util - Frag)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/claim1_workloads.png', bbox_inches='tight')
    print("Saved Claim 1 plot to assets/claim1_workloads.png")

if __name__ == "__main__":
    main()
