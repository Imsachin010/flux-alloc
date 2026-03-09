# FluxAlloc: Reinforcement Learning-Based Memory Allocation

FluxAlloc is a research-oriented project that explores the use of **Reinforcement Learning (RL)** to improve dynamic memory allocation strategies.

Traditional memory allocators rely on static heuristics such as **First Fit**, **Best Fit**, and **Worst Fit**. In this project, we design a **reinforcement learning agent** that learns to dynamically choose the best allocation strategy depending on the current heap state.

The RL agent is trained using **Proximal Policy Optimization (PPO)** and evaluated against classical heuristics across different workload patterns.

---

# Project Goals

The project aims to:

- Simulate a memory allocator environment
- Implement classical allocation heuristics
- Train a reinforcement learning agent to select allocation strategies
- Compare RL performance with traditional allocators
- Analyze the learned policy

---

# Features

- Heap memory simulator
- Classical memory allocation algorithms
- Reinforcement Learning environment
- PPO training pipeline
- TensorBoard training visualization
- Policy analysis tools
- Workload generalization experiments
- Automated result plotting

---

# Project Structure

```

flux-alloc/
│
├── heap.py
├── allocator_strategies.py
├── workload_generator.py
├── metrics.py
│
├── baseline_experiment.py
├── rl_env.py
├── train_agent.py
├── evaluate_rl.py
│
├── analyze_policy.py
├── policy_heatmap.py
├── strategy_switch_analysis.py
│
├── eval_rl_generalization.py
├── generate_plots.py
│
├── tensorboard_metrics.csv
├── plots/
│
└── rl_allocator.zip

```

---

# Installation

Create a Python environment:

```

python -m venv fluxAlloc

```

Activate environment:

**Windows**

```

fluxAlloc\Scripts\activate

```

```

Install dependencies:

```

python -m pip install -r requirements.txt

```

---

# Running the Project

## 1. Run Baseline Allocators

Evaluate classical allocation strategies.

```

python baseline_experiment.py

```

Output example:

```

First Fit
Fragmentation: 0.617
Utilization: 0.883
Failure Rate: 0.515

```

---

# 2. Train the Reinforcement Learning Agent

Train PPO agent to learn allocator policy.

```

python train_agent.py

```

This will:

- train the RL model
- log training data
- save the trained model

Output model:

```

rl_allocator.zip

```

---

# 3. View Training with TensorBoard

```

tensorboard --logdir=ppo_allocator_logs

```

Open browser:

```

[http://localhost:6006](http://localhost:6006)

```

---

# 4. Evaluate the RL Agent

```

python evaluate_rl.py

```

Example output:

```

RL Agent
Fragmentation: 0.781
Utilization: 0.866

```

---

# 5. Run Generalization Experiments

Evaluate RL on multiple workload types.

```

python eval_rl_generalization.py

```

Example output:

```

Workload: Uniform
First Fit | Frag: 0.617 | Util: 0.883

RL Agent | Frag: 0.843 | Util: 0.900

```

---

# 6. Analyze Learned Policy

```

python analyze_policy.py

```

Example output:

```

Action distribution:

First Fit : 19
Best Fit : 4954
Worst Fit : 4
Random Fit : 23

```

This reveals which allocator strategy the RL agent prefers.

---

# 7. Generate Visualization Plots

```

python generate_plots.py

```

Plots generated:

```

plots/
reward_curve.png
kl_divergence.png
value_loss.png
fragmentation_comparison.png
utilization_comparison.png
policy_distribution.png

```

---

# RL Environment Design

The memory allocator is formulated as a **Markov Decision Process (MDP)**.

### State Representation

The state vector contains:

```

[utilization,
fragmentation,
largest_free_block,
num_free_blocks,
request_size]

```

### Action Space

The agent chooses among allocation strategies:

```

0 → First Fit
1 → Best Fit
2 → Worst Fit
3 → Random Fit

```

### Reward Function

The reward encourages efficient memory usage:

```

Reward = Utilization − Fragmentation − AllocationFailurePenalty

```

---

# Workload Types

Three workload generators are used:

### Uniform workload
Random allocation sizes.

### Bimodal workload
Combination of small and large allocations.

### Adversarial workload
Designed to trigger fragmentation.

---

# Results Summary

| Method | Fragmentation | Utilization |
|------|------|------|
First Fit | 0.617 | 0.883 |
Best Fit | 0.701 | 0.876 |
Worst Fit | 0.896 | 0.765 |
Random Fit | 0.868 | 0.852 |
RL Agent | 0.843 | 0.900 |

---

# Key Observations

- The RL agent learns a policy that approximates the **Best Fit** heuristic.
- RL adapts allocation strategy based on heap state.
- Policy analysis shows **Best Fit selected in ~99% of decisions**.

---

# Future Improvements

Possible extensions:

- richer state representation
- mixed workload training
- larger heap sizes
- learning direct memory placement instead of heuristic selection

---

# Author

Sachin Mishra

Research interest:  
Reinforcement Learning • Systems Optimization • Machine Learning

```

