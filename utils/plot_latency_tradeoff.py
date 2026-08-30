import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Data from Section 5 and Section 4 of EXPERIMENT_REPORT.md
    methods = ["Best Fit", "Lookahead Ranker (Forward)", "Lookahead+Neural (Full L3)", "MaskablePPO (Transformer)"]
    
    # X-axis: Mean Latency (ms) - Section 5
    latencies = [0.0143, 0.1238, 5.2610, 7.2341]
    
    # Y-axis: Efficiency Score (Util - Frag) - Section 4 Bimodal
    scores = [0.2046, 0.2266, 0.2266, 0.0745]
    
    # Bubble Size: Success Rate (1 - Fail Rate) * scalar
    fail_rates = [0.3431, 0.3207, 0.3207, 0.3515]
    success_rates = [1 - f for f in fail_rates]
    bubble_sizes = [s * 1000 for s in success_rates]
    
    colors = ['green', 'blue', 'purple', 'orange']
    
    plt.figure(figsize=(10, 6))
    
    for i in range(len(methods)):
        plt.scatter(latencies[i], scores[i], s=bubble_sizes[i], color=colors[i], alpha=0.6, label=methods[i], edgecolors='black', linewidth=1.5)
        
    plt.xscale('log')
    
    plt.title("Latency vs Efficiency Tradeoff (Bimodal Workload)")
    plt.xlabel("Mean Inference Latency (ms, log scale)")
    plt.ylabel("Efficiency Score (Utilization - Fragmentation)")
    
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Adjust legend size
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create legend with uniform marker size
    lgnd = plt.legend(handles, labels, scatterpoints=1, title="Allocation Strategy", loc='upper left')
    for handle in lgnd.legend_handles:
        handle.set_sizes([100])
    
    # Annotations
    for i in range(len(methods)):
        plt.annotate(f"Success Rate: {success_rates[i]:.1%}", (latencies[i], scores[i]), textcoords="offset points", xytext=(0,20), ha='center', fontsize=9)

    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/latency_tradeoff.png', dpi=300, bbox_inches='tight')
    print("Saved latency tradeoff plot to assets/latency_tradeoff.png")

if __name__ == "__main__":
    main()
