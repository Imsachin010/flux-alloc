import os
import matplotlib.pyplot as plt

def main():
    # Data from Section 6 of EXPERIMENT_REPORT.md
    
    # 6.1 Neural / Rollout Blend
    blend_labels = ["MLP Rank Only\n(neural=1, sim=0)", "Score Only\n(neural=0, sim=1)", "Blend\n(0.2 / 0.8)"]
    blend_scores = [0.1315, 0.0515, 0.2266]
    
    # 6.2 Lookahead Depth
    depth_labels = ["Depth 4", "Depth 8", "Depth 12\n(Default)", "Depth 16"]
    depth_scores = [0.2046, 0.1310, 0.2266, 0.2102]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Blend Ratios
    bars1 = ax1.bar(blend_labels, blend_scores, color=['lightcoral', 'lightblue', 'royalblue'])
    ax1.set_title("Ablation: Neural vs. Simulation Blend")
    ax1.set_ylabel("Efficiency Score (Util - Frag)")
    ax1.set_ylim(0, 0.25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    # Plot 2: Lookahead Depths
    bars2 = ax2.bar(depth_labels, depth_scores, color=['silver', 'silver', 'royalblue', 'silver'])
    ax2.set_title("Ablation: Lookahead Depth (Horizon)")
    ax2.set_ylabel("Efficiency Score (Util - Frag)")
    ax2.set_ylim(0, 0.25)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.suptitle("Lookahead + Neural Ablation Studies (Bimodal Workload)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/ablation_study.png', dpi=300, bbox_inches='tight')
    print("Saved ablation study plot to assets/ablation_study.png")

if __name__ == "__main__":
    main()
