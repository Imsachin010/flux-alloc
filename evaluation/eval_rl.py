import sys
import random
from sb3_contrib import MaskablePPO
from core.rl_env_direct_final import DirectPlacementEnv
from core.metrics import Metrics

# Backward compatibility for loading the model
from policy import custom_policy_transformer
sys.modules['custom_policy_transformer'] = custom_policy_transformer

def evaluate():
    # Set a fixed seed so your results are consistent between evaluation runs
    random.seed(42)
    
    # Load the trained MaskablePPO model
    model = MaskablePPO.load("assets/rl_direct_allocator")
    
    # Initialize the Environment
    env = DirectPlacementEnv()
    state = env.reset()
    
    failures = 0
    total_malloc = 0

    # Step through the episode
    while env.ptr < len(env.workload):
        req = env.workload[env.ptr]
        if req[0] == "malloc":
            total_malloc += 1

        # Must pass action_masks to ensure the agent doesn't pick invalid locations
        action_masks = env.get_action_mask()
        action, _ = model.predict(state, action_masks=action_masks)
        
        state, reward, done, info = env.step(action)
        if req[0] == "malloc" and info.get("allocation_failed", False):
            failures += 1

        if done:
            break

    # Calculate final metrics
    heap = env.heap
    util = heap.utilization()
    frag = Metrics.external_fragmentation(heap)
    fail_rate = failures / total_malloc if total_malloc else 0

    # Print the clean output
    print("\n=== RL AGENT EVALUATION ===")
    print(f"Utilization:   {util:.4f}")
    print(f"Fragmentation: {frag:.4f}")
    print(f"Largest Block: {heap.largest_free_block()}")
    print(f"Free Blocks:   {len(heap.free_blocks())}")
    print(f"Failure Rate:  {fail_rate:.4f}")
    print("===========================\n")

if __name__ == "__main__":
    evaluate()
