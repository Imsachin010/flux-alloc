import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Academic style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif',
        'figure.dpi': 300
    })

    methods = ["Best Fit", "Lookahead (Ranker Only)", "Lookahead (Full L3)", "MaskablePPO"]
    latencies = [0.0143, 0.1238, 5.2610, 7.2341]
    scores = [0.2046, 0.2266, 0.2266, 0.0745]
    
    fail_rates = [0.3431, 0.3207, 0.3207, 0.3515]
    success_rates = [1 - f for f in fail_rates]
    
    # Scale bubbles for visual contrast, but not overly massive
    bubble_sizes = [(s - 0.6) * 10000 for s in success_rates] 
    
    # Standard academic colors
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    
    plt.figure(figsize=(8, 5))
    
    for i in range(len(methods)):
        plt.scatter(latencies[i], scores[i], s=bubble_sizes[i], color=colors[i], alpha=0.75, label=methods[i], edgecolors='black', linewidth=1)
        
    plt.xscale('log')
    plt.xlabel("Mean Inference Latency (ms)")
    plt.ylabel("Efficiency Score (Util - Frag)")
    
    plt.grid(True, which="major", linestyle=':', alpha=0.7)
    
    # Clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    lgnd = plt.legend(handles, labels, loc='lower left', title="Allocator Strategy", frameon=True)
    for handle in lgnd.legend_handles:
        handle.set_sizes([100])
        
    # Subtle labels instead of massive annotations
    for i in range(len(methods)):
        plt.annotate(f"{success_rates[i]:.1%} Success", (latencies[i], scores[i] + 0.01), ha='center', fontsize=10)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/claim2_latency.png', bbox_inches='tight')
    print("Saved Claim 2 plot to assets/claim2_latency.png")

if __name__ == "__main__":
    main()
