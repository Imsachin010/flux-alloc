import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from core.workload_generator import WorkloadGenerator
from core.rl_env_direct_final import DirectPlacementEnv

# Backward compatibility
import sys
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer

def main():
    try:
        model = MaskablePPO.load("assets/rl_direct_allocator")
    except Exception as e:
        print(f"Failed to load RL model: {e}")
        return

    random.seed(42)
    heap_size = 1024
    num_requests = 1000
    
    generator = WorkloadGenerator(heap_size)
    workload = generator.uniform_workload(num_requests)
    
    env = DirectPlacementEnv(heap_size=heap_size, episode_length=len(workload))
    state = env.reset()
    env.workload = workload 
    
    selected_sizes = []
    
    while env.ptr < len(env.workload):
        req = env.workload[env.ptr]
        
        if req[0] == "malloc":
            action_masks = env.get_action_mask()
            action, _ = model.predict(state, action_masks=action_masks)
            
            # Action corresponds to the block index in the heap free list
            free_blocks = env.heap.free_blocks()
            if action < len(free_blocks):
                # block[1] is the actual Block object, block[1].size is its size
                chosen_block_size = free_blocks[action][1].size
                selected_sizes.append(chosen_block_size)
                
            state, _, done, _ = env.step(action)
        else:
            state, _, done, _ = env.step(0)
            
        if done:
            break

    # Plotting Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(selected_sizes, bins=30, color='teal', edgecolor='black', alpha=0.7)
    
    plt.title("Agent Block Size Selection Preference")
    plt.xlabel("Selected Free Block Size")
    plt.ylabel("Frequency (Number of Times Chosen)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/block_preference.png', dpi=300, bbox_inches='tight')
    print("Saved block preference visualization to assets/block_preference.png")

if __name__ == "__main__":
    main()
