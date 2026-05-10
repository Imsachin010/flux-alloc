# FluxAlloc: Experiment Report (Level-3 Lookahead + Baselines)

This document records **reproducible** benchmark numbers from the current repository. All multi-strategy runs use a **single fixed trace** per table (same workload array, `random.seed(42)` before generation), heap size **1024**, and **1000** requests unless noted.

**Generated environment:** `python` runs from the repository root (`flux-alloc/`), on the machine where these commands were last run. Re-run the listed commands after retraining or code changes.

---

## 1. Metric definitions

| Term | Definition (this codebase) |
|------|----------------------------|
| **Utilization** | Allocated bytes / heap size (`Metrics.utilization`). |
| **Fragmentation** | External fragmentation, \(1 - \max_{free} / total\_free\), from `Metrics.external_fragmentation` (0–1, higher = more fragmented by this measure). |
| **Failure rate (unified)** | Failed mallocs / total malloc operations. Heuristics: `allocate` returns `None`. **MaskablePPO / `DirectPlacementEnv`:** the env sets `info["allocation_failed"]` in `core/rl_env_direct_final.py` when the chosen free block is invalid, too small, or `heap.allocate` returns `None`. (Earlier builds used a free-memory proxy; that is **removed** for a consistent definition with classics.) **Lookahead:** no valid free index to score / allocate. |
| **Score (Util − Frag)** | `utilization - external_fragmentation` (higher = better for this project scalar). |
| **Lg.fr/H** | `largest_free_block() / heap_size` at trace end. |

### Level-3 (Lookahead + Neural)

- **Lookahead steps (default):** 12.  
- **Blend:** `0.2` neural MLP + `0.8` short rollout (see `lookahead/lookahead_sim.py` + `lookahead_allocator.py`).  
- **Sim rollouts:** best-fit for future **mallocs**; FIFO for **frees** on a heap **copy**; `lookahead/lookahead_ranker.pt` when present.

---

## 2. Main allocator benchmark (uniform workload)

*Command:* `python -m evaluation.compare_allocators`  
*Trace:* `uniform_workload(1000)`, `seed=42`.

| Strategy | Utilization | Fragmentation | Fail rate | Score (Util − Frag) |
|----------|------------:|--------------:|---------:|--------------------:|
| Best Fit | 0.8438 | 0.6625 | 0.5134 | +0.1813 |
| Random Fit | 0.8613 | 0.7676 | 0.5219 | +0.0937 |
| First Fit | 0.8018 | 0.7685 | 0.5177 | +0.0333 |
| **MaskablePPO Agent** | 0.7754 | 0.8217 | 0.5205 | **−0.0463** |
| Worst Fit | 0.7402 | 0.9248 | 0.5233 | −0.1846 |

*With explicit allocation failure, PPO’s end-state score on this uniform trace **falls between** First Fit and Worst Fit in this run; do not over-claim without cross-seeds and confidence intervals.*

---

## 3. Fast RL single-episode check (not Section-2–comparable)

*Command:* `python -m evaluation.eval_rl`  
Uses the default env-generated workload; **not** the Section 2 array.

| Stat | Value (last run) |
|------|------:|
| Utilization | 0.8857 |
| Fragmentation | 0.8718 |
| Failure rate (unified) | 0.5367 |
| Largest block (bytes index) | 15 |
| Free blocks | 24 |

---

## 4. `lookahead.compare_final` (same trace per row)

All rows share one workload; includes **Lg.fr/H** and optional **PPO** when a checkpoint exists.

### 4a. Uniform, PPO not run

`python -m lookahead.compare_final --workload uniform --no-ppo --requests 1000`

| Strategy | Util. | Frag. | Fail | Lg.fr/H | Score |
|----------|------:|------:|-----:|--------:|------:|
| Best Fit | 0.8438 | 0.6625 | 0.5134 | 0.0527 | +0.1813 |
| Random Fit | 0.8613 | 0.7676 | 0.5219 | 0.0322 | +0.0937 |
| First Fit | 0.8018 | 0.7685 | 0.5177 | 0.0459 | +0.0333 |
| Lookahead+Neural | 0.8643 | 0.8489 | 0.5078 | 0.0205 | +0.0153 |
| Worst Fit | 0.7402 | 0.9248 | 0.5233 | 0.0195 | −0.1846 |

### 4b. Bimodal, PPO not run

`python -m lookahead.compare_final --workload bimodal --no-ppo --requests 1000`

| Strategy | Util. | Frag. | Fail | Lg.fr/H | Score |
|----------|------:|------:|-----:|--------:|------:|
| **Lookahead+Neural** | 0.9766 | 0.7500 | 0.3207 | 0.0059 | +0.2266 |
| Best Fit | 0.9824 | 0.7778 | 0.3431 | 0.0039 | +0.2046 |
| Random Fit | 0.9697 | 0.8710 | 0.3501 | 0.0039 | +0.0988 |
| First Fit | 0.9570 | 0.8864 | 0.3417 | 0.0049 | +0.0707 |
| Worst Fit | 0.8877 | 0.9739 | 0.3459 | 0.0029 | −0.0862 |

### 4c. Bimodal **with** MaskablePPO (unified PPO fail)

`python -m lookahead.compare_final --workload bimodal --requests 1000`

| Strategy | Util. | Frag. | Fail | Lg.fr/H | Score |
|----------|------:|------:|-----:|--------:|------:|
| **Lookahead+Neural** | 0.9766 | 0.7500 | 0.3207 | 0.0059 | **+0.2266** |
| Best Fit | 0.9824 | 0.7778 | 0.3431 | 0.0039 | +0.2046 |
| Random Fit | 0.9697 | 0.8710 | 0.3501 | 0.0039 | +0.0988 |
| MaskablePPO Agent | 0.9316 | 0.8571 | 0.3515 | 0.0098 | +0.0745 |
| First Fit | 0.9570 | 0.8864 | 0.3417 | 0.0049 | +0.0707 |
| Worst Fit | 0.8877 | 0.9739 | 0.3459 | 0.0029 | −0.0862 |

*On this single bimodal trace, Lookahead+Neural is **first** in Score among all listed methods, including the trained PPO, with the **lowest** failure rate.*

---

## 5. Latency (per-allocation, CPU, `n=500` uniform trace in script)

*Command:* `python -m evaluation.eval_latency`  
*Workload inside script:* `uniform_workload(500)` with `heap_size=1024` (not identical to 1000-request tables, but the official latency lab).  
*MaskablePPO* latency is the **inference** time of `model.predict` on malloc steps. **Lookahead MLP** is a **single** 10D forward of `NeuralRanker` (not full candidate × depth simulation).

| Method | Mean (ms) | Median (ms) | Max (ms) | × vs Best Fit (mean) |
|--------|----------:|------------:|---------:|--------------------:|
| Best Fit (heuristic) | 0.0143 | 0.0125 | 0.0504 | 1.0 |
| **MaskablePPO + Transformer** (predict) | 7.2341 | 6.9191 | 15.0094 | **~506.6×** |
| Lookahead **ranker MLP only** (1 forward) | 0.1238 | 0.0846 | 1.8345 | **~8.7×** |

*Full Lookahead+Neural decisions run many rollouts and heap **copies**; wall time dominates vs Best Fit. Report planning cost separately from “one neural forward”.*

---

## 6. Ablations (Level-3, bimodal `n=1000`, `seed=42`, `lookahead/lookahead_ranker.pt`)

*Command:* `python -m lookahead.ablation_bench --workload bimodal --requests 1000`

### 6.1. Neural / rollout blend (lookahead depth = 12)

| Label | Util. | Frag. | Fail | Score | Lg.fr/H |
|-------|------:|------:|-----:|------:|--------:|
| neural=1, sim=0 (MLP rank only) | 0.9648 | 0.8333 | 0.3263 | +0.1315 | 0.0059 |
| neural=0, sim=1 (rollout score only) | 0.9746 | 0.9231 | 0.3417 | +0.0515 | 0.0020 |
| **default 0.2 / 0.8** | **0.9766** | **0.7500** | **0.3207** | **+0.2266** | 0.0059 |

*The default blend has the best Score in this ablation: pure neural or pure sim alone are **not** Pareto-dominant on this one trace.*

### 6.2. Lookahead depth (0.2 / 0.8 blend)

| Depth | Util. | Frag. | Fail | Score | Lg.fr/H |
|------:|------:|------:|-----:|------:|--------:|
| 4 | 0.9824 | 0.7778 | 0.3585 | +0.2046 | 0.0039 |
| 8 | 0.9697 | 0.8387 | 0.3221 | +0.1310 | 0.0049 |
| **12** (default) | 0.9766 | 0.7500 | 0.3207 | +0.2266 | 0.0059 |
| 16 | 0.9678 | 0.7576 | 0.3417 | +0.2102 | 0.0078 |

*Depth 12 is best for Score and failure in this small sweep; very short horizons (e.g. 4) change the outcome substantially, so the planning depth is a real hyperparameter.

---

## 7. Reproduce

```text
python -m evaluation.compare_allocators
python -m evaluation.eval_rl
python -m evaluation.eval_latency
python -m lookahead.compare_final --workload uniform  --no-ppo --requests 1000
python -m lookahead.compare_final --workload bimodal   --no-ppo --requests 1000
python -m lookahead.compare_final --workload bimodal   --requests 1000
python -m lookahead.ablation_bench --workload bimodal --requests 1000
python -m lookahead.eval_lookahead --workload bimodal  --requests 1000
```

---

## 8. Honest paper framing

- **Unify** failure: use the same `allocation_failed` story for PPO, RL eval, and comparisons (implemented in this repo).  
- **Same-trace** tables: Sections 2, 4, and 6 use fixed generators and seeds.  
- **Do not** compare Section 3 to Section 2 as “the same run.”  
- **Ablations** and **latency** are **one machine, one pass**; report **multiple seeds** in the paper for statistical claims.  
- **PPO** vs **Lookahead+Neural** are different designs (policy gradient vs. scored planning); compare on agreed traces and SLOs (e.g. fail + latency + score).

*Reproducibility — we can reproduce the tables after retraining or changing `rl_direct_allocator` or `lookahead_ranker.pt`.*

---
<!-- 
## 9. Contributors

**Contributors:**
- **Sachin Mishra**
- **Lomesh Soni**
- **Abhay Gotmare**

**Institute:**  
International Institute of Information Technology (IIIT), Bengaluru

**License:**  
This project is licensed under the **MIT License**. -->
