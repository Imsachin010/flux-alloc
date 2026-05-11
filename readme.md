# FluxAlloc: Transformer-Based Reinforcement Learning Memory Allocator

FluxAlloc is an advanced research project exploring the use of **Reinforcement Learning (RL)** to solve dynamic memory allocation natively. 

Moving beyond traditional meta-allocators (which simply choose between heuristics like Best-Fit or First-Fit), FluxAlloc implements a **Direct Memory Placement** strategy. It utilizes a **Transformer Feature Extractor** to analyze the memory heap in real-time and explicitly selects the exact free block to allocate, natively masked by `MaskablePPO` to ensure mathematical safety.

---

## Models

This repository evaluates two distinct neural approaches against standard heuristics (Best-Fit, First-Fit, etc.), each serving a specific research objective:

### 1. MaskablePPO (Transformer-Based Direct Placement)
- **Description:** An end-to-end RL agent using a multi-head self-attention mechanism (`nn.TransformerEncoder`) to process a 128-dimensional state space (Utilization, Fragmentation, Age, Size, Position). It directly outputs the index of the memory block to allocate, natively masked by `sb3-contrib`'s `MaskablePPO` to ensure mathematical safety.
- **Justification:** Explores whether a pure, reactive RL policy can natively learn spatial memory packing without relying on hardcoded heuristic rules. 
- **Tradeoff:** Achieves generalization but suffers from extreme latency overhead (~500x slower than Best-Fit) due to the heavy inference cost of the Transformer on every single allocation step.

### 2. Level-3 Lookahead + Neural Ranker (Hybrid Planning)
- **Description:** A hybrid search-based allocator that blends a fast Neural Ranker (MLP) with a short-horizon simulation rollout (default depth = 12). It scores candidate blocks by simulating future allocations (using `lookahead_ranker.pt` with a 0.2 Neural / 0.8 Sim blend).
- **Justification:** Designed to overcome the latency and sample-inefficiency of pure PPO. By looking into the future via simulation, it avoids the fragmentation traps that greedy heuristics and purely reactive RL agents fall into.
- **Tradeoff:** Achieves **State-of-the-Art** efficiency on complex Bimodal workloads, while operating significantly faster than the heavy Transformer model (only ~8.7x Best-Fit latency).

---

## Project Structure (Modular Clean Architecture)

```text
flux-alloc/
├── core/
│   ├── heap.py
│   ├── rl_env_direct_final.py  # The Direct Placement MDP Env
│   ├── allocator_strategies.py
│   ├── workload_generator.py
│   └── metrics.py
├── policy/
│   └── custom_policy_transformer.py # Transformer Neural Network
├── training/
│   ├── train_direct_final.py
│   └── train_agent.py
├── evaluation/
│   ├── compare_allocators.py   # Main Fairness Benchmark
│   ├── eval_rl.py              # Fast RL Evaluation
│   ├── eval_latency.py         # CPU vs Memory Tradeoff
│   └── eval_rl_generalization.py # Robustness Checks
├── utils/
│   ├── plot_training.py
│   ├── plot_block_preference.py # Visualizes Agent Psychology
│   └── plot_pareto_tradeoff.py  # Generates Pareto optimal visualizations
└── assets/
    ├── rl_direct_allocator.zip # Saved Model
    └── ... (Generated plots, graphs, and logs)
```

---

## Installation

Create and activate a Python environment (Windows):
```bash
python -m venv fluxAlloc
fluxAlloc\Scripts\activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

Because the repository is organized into a clean Python module structure, you must run scripts using the `-m` flag from the root directory.

### 1. The Ultimate Benchmark (Apples-to-Apples Comparison)
Run a heavily seeded, identically-matched workload sequence against Best-Fit, First-Fit, Worst-Fit, Random-Fit, and your trained MaskablePPO Agent.
```bash
python -m evaluation.compare_allocators
```

### 2. Fast RL Agent Evaluation
Quickly evaluate the agent's performance on a single fixed sequence.
```bash
python -m evaluation.eval_rl
```

### 3. CPU vs Memory Latency Tradeoff
Prove the exact execution time cost (in milliseconds) of using a Neural Network over a standard heuristic `for-loop`.
```bash
python -m evaluation.eval_latency
```

### 4. Generalization / Robustness
Test how the RL agent handles unexpected situations, such as heavy Bimodal and Adversarial workloads.
```bash
python -m evaluation.eval_rl_generalization
```

### 5. Generate Visualizations & Graphs
To regenerate the specific `.png` graphs used in the report:
```bash
python -m utils.plot_pareto_tradeoff
python -m utils.plot_block_preference
python -m utils.plot_training
```

### 6. Train the Agent from Scratch
Launch the MaskablePPO trainer with the custom Transformer.
```bash
python -m training.train_direct_final
```

### 7. Run Lookahead & Ablation Benchmarks
Test the Level-3 Lookahead model against PPO and heuristics, or run the depth/blend ablation studies.
```bash
python -m lookahead.compare_final --workload bimodal --requests 1000
python -m lookahead.ablation_bench --workload bimodal --requests 1000
```

---

## Academic Results Summary

Based on our `EXPERIMENT_REPORT.md` benchmarks:
1. **Lookahead Supremacy:** While Best-Fit edges out pure RL on simple uniform workloads, **Lookahead+Neural strictly dominates on complex Bimodal workloads** (achieving a +0.2266 Efficiency Score vs Best-Fit's +0.2046 and PPO's +0.0745), successfully pushing the Pareto optimal boundary.
2. **Computational Cost (Math vs Compute):** The Transformer MaskablePPO model provides a proof-of-concept for direct placement but costs **~500x more CPU latency** than Best-Fit. In contrast, the Lookahead MLP Ranker offers a sweet spot, requiring only **~8.7x latency** while yielding the highest performance.
3. **Synergy of Blended Planning:** Ablation studies prove that blending the Neural Ranker with simulation (0.2 / 0.8 ratio) at a Lookahead Depth of 12 produces synergistic results superior to using either method alone.

---

## Author
**Sachin Mishra**  
Reinforcement Learning • Systems Optimization • Machine Learning
