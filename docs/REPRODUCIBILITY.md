# Reproducibility Guide

This guide provides the complete experimental environment specifications, workload parameters, statistical evaluation protocols, and end-to-end execution steps required to reproduce all findings and figures reported in the paper.

---

## 1. Experimental Environment

* **Heap Size:** 1024 bytes (canonical baseline); 2048, 4096, and 65536 bytes (scale/headroom benchmarks)
* **Requests per Trace:** 1000 requests (5000 for PPO policy evaluation)
* **Evaluation Seeds (10-Seed Suite):**  
  `{0, 1, 2, 3, 4, 7, 10, 18, 42, 123}`
* **Scale / Headroom Seeds (5-Seed Subset):**  
  `{0, 7, 10, 18, 42}`
* **Hardware & Runtime:**  
  Benchmarked on x86-64 CPU (Python 3.10+ / PyTorch 2.0+ / NumPy / SciPy / Gymnasium / Stable-Baselines3).

---

## 2. Workload Definitions

1. **Uniform Workload (`uniform_workload`)**:
   * Request sizes drawn uniformly from $[8, 128]$ bytes.
   * Alternating allocation and deallocation operations.
2. **Bimodal Workload (`bimodal_workload`)**:
   * Simulates real-world software allocations: 80% small blocks ($[8, 32]$ bytes), 20% large blocks ($[128, 256]$ bytes).
   * Generates severe fragmentation under greedy heuristic strategies.
3. **Adversarial Canonical Workload (`scaled_adversarial_workload`)**:
   * Cyclic alternating pattern of small ($8$ bytes) and large ($56$ bytes) allocations designed to trap first-fit/worst-fit allocators.
4. **Randomized Adversarial Workload (`adversarial_rand`)**:
   * Small blocks $\sim \mathcal{U}\{6, 10\}$, large blocks $\sim \mathcal{U}\{42, 70\}$, cycle lengths $\sim \text{Geometric}(\text{mean } 8)$, with small blocks freed with probability $0.8$.

---

## 3. Statistical Evaluation Protocol

For all multi-seed comparisons:
* **Paired Evaluation:** All allocators are evaluated on identical, deterministic workload traces generated from the same PRNG seed.
* **Hypothesis Testing:** Two-sided paired Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`) comparing paired failure differences $\Delta = \text{Fail}_{\text{BF}} - \text{Fail}_{\text{FA}}$.
* **Confidence Intervals:** 10,000-resample non-parametric bootstrap 95% percentile confidence intervals for the mean paired difference $\Delta$.
* **Robust Scoring:** $\beta$-family composite score metric $\text{Score}_\beta = \text{Util} - \beta \cdot \text{Frag}$ evaluated for $\beta \in \{0.5, 1.0, 2.0\}$.

---

## 4. Trace Integrity & Manifest

Every workload trace generated during benchmarking is verified using SHA-256 hashing to guarantee that PRNG streams are independent and non-colliding across seeds.

The complete SHA-256 hash manifest is recorded in:
* [`traces/SHA256SUMS`](../traces/SHA256SUMS)
* [`camera_ready/trace_manifest.csv`](../camera_ready/trace_manifest.csv)

To audit and verify trace uniqueness:
```bash
python camera_ready/trace_audit.py
```

---

## 5. End-to-End Reproduction Steps

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Imsachin010/flux-alloc.git
cd flux-alloc

# Create and activate virtual environment
python -m venv fluxAlloc
.\fluxAlloc\Scripts\activate      # On Windows
# source fluxAlloc/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Full Experimental Suite
Execute all experimental batches (Batch A: Core, Batch B: Scale, Batch C: Adversarial Robustness, Batch D: Free-Policy Sensitivity):
```bash
python camera_ready/run_experiments.py
```
This logs every individual decision and run record to `camera_ready/logs/` and outputs the unified results table to `camera_ready/results.csv`.

### Step 3: Compute Statistical Significance & Metrics
```bash
python camera_ready/compute_stats.py
```
This generates the full statistical report with Wilcoxon $p$-values and bootstrap CIs in `camera_ready/stats.md`.

### Step 4: Regenerate Publication Figures
```bash
python -m paper_plots.generate_all
```
This produces all 6 publication figures in `paper_plots/` at $\ge 300$ DPI.
