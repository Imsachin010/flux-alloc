# FluxAlloc: Transformer-Based Reinforcement Learning Memory Allocator

FluxAlloc is an advanced research project exploring the use of **Reinforcement Learning (RL)** to solve dynamic memory allocation natively. 

Moving beyond traditional meta-allocators (which simply choose between heuristics like Best-Fit or First-Fit), FluxAlloc implements a **Direct Memory Placement** strategy. It utilizes a **Transformer Feature Extractor** to analyze the memory heap in real-time and explicitly selects the exact free block to allocate, natively masked by `MaskablePPO` to ensure mathematical safety.

---

## Key Architectural Upgrades
- **Direct Placement Environment:** Formulated a custom MDP where the agent selects discrete block indices, moving beyond heuristic switching.
- **Transformer Feature Extractor:** Employs a multi-head self-attention mechanism (`nn.TransformerEncoder`) to process a 128-dimensional state space encompassing both global metrics (Utilization, Fragmentation) and localized free-block embeddings (Age, Size, Position).
- **Native Action Masking:** Integrates `sb3-contrib`'s `MaskablePPO` to dynamically restrict the policy from selecting invalid or non-existent free blocks, preventing fatal allocation errors during training.
- **Advanced Reward Function:** A highly tuned, multi-objective reward structure that natively penalizes utilization loss, small fragment generation, and fragmentation density while rewarding largest block preservation.

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

---

## Academic Results Summary

Our findings from evaluating the Transformer-based RL agent:
1. **Generalization Supremacy:** While Best-Fit barely edges out RL on standard uniform workloads, **MaskablePPO crushes Best-Fit on complex Bimodal workloads** (+0.212 vs +0.131 Efficiency Score).
2. **Computational Cost:** The Transformer architecture provides superior mathematical memory packing but costs roughly **800x more CPU latency** per decision than traditional Best-Fit.
3. **Behavioral Discovery:** By analyzing block selection preferences, we proved the agent develops targeted preferences (favoring smaller blocks to preserve large, contiguous gaps for the future).

---

## Author
**Sachin Mishra**  
Reinforcement Learning • Systems Optimization • Machine Learning
