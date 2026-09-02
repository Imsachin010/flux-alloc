# FluxAlloc Repository Map (Phase 0 Reconnaissance)

This document provides the complete architectural and file-level map of the FluxAlloc repository for the camera-ready experimental regeneration.

---

## 1. Heap Environment & Metrics Computation

* **File:** `core/heap.py`
  * Class: `Heap(size: int)`
  * Core methods:
    * `allocate(free_block_index: int, size: int) -> int | None`: Splits target free block if `block.size >= size`. Allocates block, tags `allocated=True`, assigns `block_id`, records `self.last_allocated_addr`.
    * `free(block_id: int) -> bool`: Frees allocated block and triggers `_coalesce()`.
    * `_coalesce()`: Iterates adjacent blocks and merges consecutive free blocks.
    * `free_blocks() -> list[tuple[int, Block]]`: Returns enumerated list of unallocated free blocks.
    * `largest_free_block() -> int`: Returns maximum size among free blocks (0 if full).
    * `utilization() -> float`: `allocated_bytes / total_heap_size`.
* **File:** `core/metrics.py`
  * Class: `Metrics`
  * Methods:
    * `utilization(heap: Heap) -> float`: `allocated_bytes / total_heap_size`.
    * `external_fragmentation(heap: Heap) -> float`: `1.0 - (largest_free_block / total_free_bytes)` if `total_free_bytes > 0` else `0.0`.
    * `largest_free_block(heap: Heap) -> int`.
    * `free_block_count(heap: Heap) -> int`.

---

## 2. Workload Generators & Seeding Audit

* **File:** `core/workload_generator.py`
  * Class: `WorkloadGenerator(heap_size)`
  * Generator Methods & RNG Sources:
    1. `uniform_workload(num_requests=1000)`:
       * Uses global `random.random()` for operation choice (70% malloc, 30% free) and `random.randint(1, 64)` for allocation size.
       * *Seeding*: Dependent on ambient global `random.seed(...)`.
    2. `bimodal_workload(num_requests=1000)`:
       * Uses global `random.random()` for operation choice (70% malloc, 30% free), `random.random() < 0.7` for small vs large mode, `random.randint(1, 8)` (small) vs `random.randint(32, 64)` (large).
       * *Seeding*: Dependent on ambient global `random.seed(...)`.
    3. `adversarial_workload()`: Static mini-trace (toy example).
    4. `scaled_adversarial_workload(num_requests=1000)`:
       * Simulates Best Fit on a companion heap to generate adversarial allocation/free sequences.
       * *Seeding Flaw*: Contains hardcoded `random.seed(42)` on line 72! This causes all seeds to generate identical traces.
       * *RNG Sources*: `random.random()` and `random.choice()`.

---

## 3. Strategy Implementations

* **Classical Heuristics (`core/allocator_strategies.py`):**
  * `first_fit(heap, request_size)`: First free block with `size >= request_size`.
  * `best_fit(heap, request_size)`: Minimum-sized free block with `size >= request_size`.
  * `worst_fit(heap, request_size)`: Maximum-sized free block with `size >= request_size`.
  * `random_fit(heap, request_size)`: `random.choice` among valid candidates.
  * `next_fit(heap, request_size)`: First valid free block at or after `heap.last_allocated_addr`.
* **Baseline PPO (`policy/` & `assets/`):**
  * Checkpoint: `assets/rl_direct_allocator.zip` (MaskablePPO with Transformer/MLP policy).
  * Environment: `core/rl_env_direct_final.py` (`DirectPlacementEnv`).
* **FluxAlloc / Lookahead Hybrid (`lookahead/`):**
  * Allocator: `lookahead/lookahead_allocator.py` (`LookaheadAllocator`).
  * Rollout Simulator: `lookahead/lookahead_sim.py` (`simulate_after_malloc`).
  * Neural Ranker: `lookahead/neural_ranker.py` (`NeuralRanker`).

---

## 4. FluxAlloc Internals & Rollout Peeking Verification

### Rollout Peek vs. Generator Verification (CRITICAL for Batch C):
* **Answer:** **(b) Peeks at the true upcoming requests.**
* *Code verification in `lookahead/lookahead_sim.py`:*
  ```python
  for t in range(1, lookahead_steps + 1):
      p = request_ptr + t
      if p >= len(workload):
          break
      w = workload[p]  # Peeks directly into upcoming workload sequence!
  ```
* **Rollout Assumptions:**
  * Future `malloc`s: Simulated using the designated oracle strategy (default: `best_fit`, or `next_fit`, `first_fit`).
  * Future `free`s: If request specifies a `target_size`, searches for an active block matching that size; otherwise defaults to FIFO (`act.pop(0)`).

### Ranker MLP Architecture & Checkpoint:
* **Architecture:** 3-layer MLP: Linear(10, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 1).
* **Parameters:** $(10 \times 128 + 128) + (128 \times 128 + 128) + (128 \times 1 + 1) = 1,408 + 16,512 + 129 = 18,049$ parameters.
* **Input Features (10 dimensions):**
  1. `utilization` (heap utilization)
  2. `external_fragmentation` (heap fragmentation)
  3. `largest_free_block / heap_size`
  4. `num_free_blocks / 20`
  5. `request_size / heap_size`
  6. `candidate_block_size / heap_size`
  7. `candidate_block_start / heap_size`
  8. `(candidate_block_size - request_size) / heap_size`
  9. `sim_hint` (bounded $[0, 1]$)
  10. `lookahead_steps / 100.0`
* **Checkpoint:** `lookahead/lookahead_ranker.pt`
* **Training Script:** `lookahead/train_ranker.py` (trained on uniform workload, seed 42, 3000 steps, Adam lr=1e-3, MSELoss, 40 epochs).

---

## 5. Benchmarking, Ablations, and Policy Distribution Scripts

* **Main Allocator Benchmark / CLI:** `lookahead/compare_final.py` (`python -m lookahead.compare_final --workload [uniform|bimodal|adversarial] --requests 1000 --oracle [best_fit|first_fit|next_fit]`).
* **Multi-Seed Benchmark:** `lookahead/multi_seed_bench.py`.
* **Lookahead Ablations:** `lookahead/ablation_bench.py` (ablates depth $k \in \{4, 8, 12, 16\}$ and blend $\alpha \in \{0.0, 0.2, 1.0\}$).
* **Latency Evaluation:** `evaluation/eval_latency.py` (Best Fit, MaskablePPO, Ranker MLP forward pass).
* **PPO Policy Distribution Analysis:** `utils/plot_claim4_policy.py` / `paper_plots/plot4_ppo_distribution.py` (analyzes action distribution: 4954 Best Fit, 23 Random Fit, 19 First Fit, 4 Worst Fit).
* **Paper Plots Scripts:** `paper_plots/plot1_workloads.py` through `plot6_multi_seed_reliability.py` and `paper_plots/generate_all.py`.
