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

    depths = [4, 8, 12, 16]
    depth_scores = [0.2046, 0.1310, 0.2266, 0.2102]
    
    blend_labels = ["Neural Only", "Sim Only", "Blend (0.2/0.8)"]
    blend_scores = [0.1315, 0.0515, 0.2266]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot A: Depth (Line)
    ax1.plot(depths, depth_scores, marker='o', markersize=8, linestyle='-', linewidth=2, color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel("Lookahead Horizon (Depth)")
    ax1.set_ylabel("Efficiency Score")
    ax1.set_xticks(depths)
    ax1.set_ylim(0.10, 0.25)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Plot B: Blend (Bar)
    bars = ax2.bar(blend_labels, blend_scores, color=['#d62728', '#7f7f7f', '#1f77b4'], edgecolor='black', linewidth=1, width=0.5)
    ax2.set_xlabel("Ranker Configuration")
    ax2.set_ylim(0.0, 0.25)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/claim3_ablations.png', bbox_inches='tight')
    print("Saved Claim 3 plot to assets/claim3_ablations.png")

if __name__ == "__main__":
    main()
