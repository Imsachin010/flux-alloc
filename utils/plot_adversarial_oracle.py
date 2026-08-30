import os
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

    oracles = ["First Fit Oracle\n(first_fit)", "Next Fit Oracle\n(next_fit)", "Best Fit Oracle\n(best_fit)"]
    scores = [0.0353, 0.3866, 0.4990]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#ffbb78', '#aec7e8', '#1f77b4'] # Light Orange, Light Blue, Deep Blue
    
    bars = ax.bar(oracles, scores, color=colors, edgecolor='black', width=0.4)
    ax.set_ylabel("L3 Efficiency Score (Util - Frag)")
    ax.set_xlabel("Internal Simulation Rollout Oracle")
    ax.set_title("L3 Rollout Oracle Ablation (Adversarial Workload)")
    ax.set_ylim(0.0, 0.6)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:+.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/adversarial_oracle.png', bbox_inches='tight')
    print("Saved L3 Rollout Oracle Ablation plot to assets/adversarial_oracle.png")

if __name__ == "__main__":
    main()
