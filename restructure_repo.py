import os
import shutil
import re

directories = {
    'core': ['rl_env_direct_final.py', 'rl_env.py', 'heap.py', 'workload_generator.py', 'allocator_strategies.py', 'metrics.py'],
    'policy': ['custom_policy_transformer.py'],
    'training': ['train_direct_final.py', 'train_agent.py'],
    'evaluation': ['eval_direct_final.py', 'analyze_heap_behavior.py', 'eval_rl_generalization.py', 'evaluate.py', 'evaluate_rl.py', 'analyze_policy.py', 'baseline_experiment.py', 'strategy_switch_analysis.py'],
    'utils': ['export_tb_data.py', 'plot_training.py', 'policy_heatmap.py', 'resultfromtensorboard.py'],
    'tests': ['test_env.py', 'test_heap.py', 'test_workload.py'],
    'assets': ['policy_distribution.png', 'policy_heatmap.png', 'reward_curve.png', 'tensorboard_metrics.csv', 'rl_allocator.zip', 'rl_direct_allocator.zip']
}

# 1. Create Directories and __init__.py
for d, files in directories.items():
    if not os.path.exists(d):
        os.makedirs(d)
    if d != 'assets':
        init_path = os.path.join(d, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass

# 2. Move files
for d, files in directories.items():
    for f in files:
        if os.path.exists(f):
            print(f"Moving {f} to {d}/")
            shutil.move(f, os.path.join(d, f))

# 3. Update imports
# Mapping module name to its new package prefix
module_packages = {
    'rl_env_direct_final': 'core',
    'rl_env': 'core',
    'heap': 'core',
    'workload_generator': 'core',
    'allocator_strategies': 'core',
    'metrics': 'core',
    'custom_policy_transformer': 'policy',
    # Assume training/evaluation etc aren't imported much, but add if needed
}

def update_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for mod, pkg in module_packages.items():
        # Match 'from mod import X' -> 'from pkg.mod import X'
        new_content = re.sub(rf'^from {mod} import', f'from {pkg}.{mod} import', new_content, flags=re.MULTILINE)
        
        # Match 'import mod' -> 'from pkg import mod'
        # Be careful not to replace 'import model' when looking for 'import mod'
        new_content = re.sub(rf'^import {mod}(\s|$)', f'from {pkg} import {mod}\\1', new_content, flags=re.MULTILINE)

    # Some scripts load "rl_direct_allocator" or zip files. Let's fix those paths too since they moved to assets/
    new_content = new_content.replace('"rl_direct_allocator"', '"assets/rl_direct_allocator"')
    new_content = new_content.replace("'rl_direct_allocator'", "'assets/rl_direct_allocator'")
    new_content = new_content.replace('"rl_allocator"', '"assets/rl_allocator"')
    new_content = new_content.replace("'rl_allocator'", "'assets/rl_allocator'")
    
    # Also adjust tensorboard paths if needed? Not strictly necessary unless hardcoded.

    if new_content != content:
        print(f"Updated imports in {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

# Walk through all python files in the new directories to update their imports
for d in directories.keys():
    if d == 'assets': continue
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.py'):
                update_imports(os.path.join(root, f))

print("Restructuring complete!")
