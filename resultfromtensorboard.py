import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
# Replace this with the path to your PPO_1 log folder
LOG_DIR = "./ppo_allocator_logs/PPO_1" 
# The specific metric you want to extract (from your screenshot)
TAG_TO_EXTRACT = 'train/approx_kl' 
OUTPUT_NAME = "ppo_approx_kl"
# ---------------------

def export_tensorboard_data(log_dir, tag):
    if not os.path.exists(log_dir):
        print(f"Error: Folder {log_dir} not found.")
        return

    # Load the event file
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Check available tags if the requested one isn't found
    tags = event_acc.Tags()['scalars']
    if tag not in tags:
        print(f"Tag '{tag}' not found. Available tags are: {tags}")
        return

    # Extract steps and values
    w_times, step_nums, vals = zip(*event_acc.Scalars(tag))

    # Create DataFrame
    df = pd.DataFrame({'step': step_nums, 'value': vals})
    
    # 1. Save to CSV
    csv_file = f"{OUTPUT_NAME}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Successfully saved raw data to: {csv_file}")

    # 2. Save as PNG Graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['value'], label=tag, color='blue', alpha=0.8)
    plt.title(f"TensorBoard Export: {tag}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    png_file = f"{OUTPUT_NAME}.png"
    plt.savefig(png_file, dpi=300)
    print(f"Successfully saved graph to: {png_file}")
    plt.show()

if __name__ == "__main__":
    export_tensorboard_data(LOG_DIR, TAG_TO_EXTRACT)