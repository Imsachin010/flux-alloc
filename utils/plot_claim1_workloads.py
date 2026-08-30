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
    # Adversarial
    score_a = [0.1990, 0.1704, -0.0010, 0.4990, np.nan, -0.2807]

    # Reorder methods: Lookahead, Best, PPO, First, Random, Worst
    order = [3, 0, 4, 2, 1, 5] 
    methods = [methods[i] for i in order]
    score_u = [score_u[i] for i in order]
    score_b = [score_b[i] for i in order]
    score_a = [score_a[i] for i in order]

    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    # Muted academic colors
    color_u = '#5D8AA8' # Air Force Blue
    color_b = '#A52A2A' # Brown/Red
    color_a = '#2E8B57' # Sea Green
    
    rects1 = ax.bar(x - width, score_u, width, label='Uniform Workload', color=color_u, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, score_b, width, label='Bimodal Workload', color=color_b, edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, score_a, width, label='Adversarial Workload', color=color_a, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Efficiency Score (Util - Frag)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)

    # Note that MaskablePPO is not evaluated on Adversarial workload
    ax.text(2, -0.05, 'N/A', ha='center', va='bottom', color='gray', fontsize=10, fontstyle='italic')

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/claim1_workloads.png', bbox_inches='tight')
    print("Saved Claim 1 plot with Adversarial workload to assets/claim1_workloads.png")

if __name__ == "__main__":
    main()
