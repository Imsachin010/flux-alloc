# FluxAlloc: Model Architectures & Data Flow

This document outlines the theoretical design, structural flow, and algorithmic architectures of the two primary neural allocators developed in the FluxAlloc repository.

---

## 1. MaskablePPO (Transformer-Based Direct Placement)

Traditional RL allocators function as "Meta-Allocators," merely selecting between predefined heuristics (e.g., Best-Fit vs. First-Fit). FluxAlloc's MaskablePPO agent represents a **Direct Memory Placement** formulation. It processes the raw geometry of the memory heap and directly outputs the exact block index for allocation.

### Architecture Flowchart

<img src="MaskablePPO%20(Transformer-Based%20Direct%20Placement).png" height="600" width="400" alt="MaskablePPO Architecture" />

### Key Components

*   **128-Dimensional State Extraction:** Instead of passing raw bytes, the heap is converted into a structured 128D vector. It includes global metrics (current utilization, global fragmentation) and localized embeddings of the free blocks (size, spatial age, position in memory).
*   **Transformer Encoder:** To handle the varying sizes and relationships between distinct free memory blocks, we use an `nn.TransformerEncoder`. Multi-head self-attention allows the neural network to learn the spatial relationships and gaps between free blocks without being tied to a rigid grid.
*   **Native Action Masking:** A significant challenge in Direct Placement MDPs is the agent selecting invalid, occupied, or undersized blocks. By integrating `MaskablePPO` from `sb3-contrib`, we dynamically pass a boolean mask during training and inference. Invalid actions are assigned $-\infty$ logits, making it mathematically impossible for the agent to make an illegal move.
*   **Reward Function:** The agent is guided by a highly tuned multi-objective reward function. It penalizes fragmentation (weighted heavily) and utilization loss, while specifically rewarding the preservation of the "largest free contiguous block" to prevent premature fragmentation failure.

---

## 2. Level-2 Lookahead + Neural Ranker (Hybrid Planning)

While pure PPO provides excellent generalization, its reactive nature and heavy inference latency (~500x slower than Best-Fit) make it computationally expensive. To bridge the gap between "dumb but fast heuristics" and "smart but slow neural networks," we designed the **Level-3 Lookahead Neural Ranker**.

### Architecture Flowchart

<img src="NN+%20Lookahead.png" height="600" width="400" alt="Lookahead Architecture" />

### Key Components

*   **Deepcopy Planning:** Instead of reacting to the immediate state, the allocator evaluates *all* valid candidate blocks for the current request. It creates isolated clones of the heap for each candidate and performs a virtual allocation.
*   **The Neural Ranker (0.2 Blend):** The virtual state is passed into a lightweight MLP (`lookahead_ranker.pt`). Because it is a simple MLP rather than a full Transformer, it evaluates almost instantly. It scores the mathematical "health" of the heap state based on utilization, fragmentation, and block distribution.
*   **12-Step Rollout Simulation (0.8 Blend):** To avoid getting trapped in local optima, the allocator simulates the next 12 requests in the workload sequence. It uses rapid heuristic stand-ins (Best-Fit for future mallocs, FIFO for future frees) to project the heap 12 steps into the future.
*   **Synergistic Combination:** The final decision is a weighted blend (20% immediate neural intuition, 80% empirical future simulation). Ablation studies proven this exact configuration drastically outperforms both pure Neural evaluation and pure Simulation rollouts, successfully pushing the Pareto optimal boundary on complex bimodal workloads.


