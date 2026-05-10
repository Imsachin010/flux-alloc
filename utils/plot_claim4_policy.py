import os
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.dpi': 300
    })

    labels = ['First Fit', 'Best Fit', 'Worst Fit', 'Random Fit']
    counts = [19, 4954, 4, 23]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Use a simple bar chart instead of a pie chart (which fails for 99% vs 0.1%)
    colors = ['#aec7e8', '#1f77b4', '#ffbb78', '#98df8a']
    
    bars = ax.barh(labels, counts, color=colors, edgecolor='black', height=0.6)
    ax.set_xlabel('Number of Times Selected by Agent')
    ax.set_xscale('log') # Log scale because Best Fit is 5000 and others are ~20
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    
    # Add text annotations
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.1, bar.get_y() + bar.get_height()/2, f"{int(width)}", 
                va='center', ha='left', fontsize=11)
                
    # Extend x-axis slightly so text fits
    ax.set_xlim(1, 20000)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/claim4_policy.png', bbox_inches='tight')
    print("Saved Claim 4 (Bar Chart) to assets/claim4_policy.png")

if __name__ == "__main__":
    main()
