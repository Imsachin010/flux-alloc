# FluxAlloc: Adaptive Dynamic Memory Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/IEEE%20DSAA-2026-blue)](https://dsaa2026.dsaa.co/)

**FluxAlloc** is an adaptive dynamic memory allocation framework designed to mitigate external heap fragmentation in memory-constrained systems. Moving beyond traditional meta-allocators (which heuristically switch between classical rules like Best-Fit or First-Fit), FluxAlloc implements:

1. **Direct Memory Placement (MaskablePPO)**: An end-to-end Reinforcement Learning (RL) agent utilizing a **Transformer Feature Extractor** to process heap block geometry in real time and directly select allocation targets under strict action masking.
2. **Hybrid Lookahead Planning (FluxAlloc Ranker)**: A high-efficiency planning allocator that blends a 10-dimensional **Neural Ranker (MLP)** with short-horizon simulation rollouts ($k=12$), avoiding fragmentation traps with minimal latency overhead.

---

## Repository Structure

```text
flux-alloc/
├── LICENSE                   # MIT License
├── CITATION.cff              # Machine-readable citation metadata
├── CITATION.bib              # BibTeX citation entry
├── THIRD_PARTY_NOTICES.md    # Upstream software and dependency notices
├── CHANGELOG.md              # Project version history
├── CONTRIBUTING.md           # Development and contribution guidelines
├── CODE_OF_CONDUCT.md        # Contributor Covenant Code of Conduct
├── requirements.txt          # Python package dependencies
│
├── core/                     # Heap simulation, heuristics, and MDP environment
│   ├── heap.py               # Heap memory model, block splitting, and coalescing
│   ├── allocator_strategies.py # Classical heuristics (Best-Fit, First-Fit, etc.)
│   ├── rl_env_direct_final.py # Gymnasium Direct Placement MDP environment
│   ├── workload_generator.py # Workload generators (Uniform, Bimodal, Adversarial)
│   └── metrics.py            # Utilization and external fragmentation metrics
│
├── lookahead/                # Lookahead planning & Neural Ranker
│   ├── lookahead_allocator.py # Hybrid Lookahead allocator
│   ├── lookahead_sim.py      # Rollout simulation engine
│   ├── neural_ranker.py      # 10-feature MLP ranker network
│   ├── train_ranker.py       # Supervised ranker training harness
│   └── bench.py              # Single-pass lookahead benchmark
│
├── policy/                   # Neural network architectures
│   └── custom_policy_transformer.py # Multi-head attention policy extractor
│
├── camera_ready/             # Camera-ready experimental regeneration suite
│   ├── run_experiments.py    # Master runner for Batches A, B, C, D
│   ├── compute_stats.py      # Wilcoxon tests & bootstrap CI computations
│   ├── trace_audit.py        # SHA-256 trace verification audit
│   ├── trace_manifest.csv    # SHA-256 hash manifest per (workload, seed)
│   ├── results.csv           # Unified benchmark output records
│   ├── stats.md              # Statistical significance report
│   ├── RESULTS.md            # Detailed per-batch tables & token values
│   └── DELTA.md              # New vs paper-current delta tracking
│
├── paper_plots/              # Publication figures and plotting scripts (>= 300 DPI)
│   ├── generate_all.py       # Master script regenerating all 6 paper figures
│   └── ...                   # Individual figure generation scripts
│
├── docs/                     # Extended documentation
│   ├── REPRODUCIBILITY.md    # Detailed reproduction protocol & seed specs
│   ├── ARCHITECTURE.md       # Technical design and subsystem architecture
│   └── EXPERIMENT_REPORT.md  # Comprehensive experimental analysis
│
└── traces/                   # Checksums and trace manifests
    └── SHA256SUMS            # Workload trace SHA-256 checksums
```

---

## Getting Started

### 1. Installation

Create and activate a virtual environment:
```bash
# Clone repository
git clone https://github.com/Imsachin010/flux-alloc.git
cd flux-alloc

# Create virtual environment
python -m venv fluxAlloc

# Activate environment (Windows)
.\fluxAlloc\Scripts\activate
# Activate environment (Linux/macOS)
# source fluxAlloc/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Paper Results

All experimental runs are completely deterministic and paired across evaluation seeds $S = \{0, 1, 2, 3, 4, 7, 10, 18, 42, 123\}$.

### 1. Run All Experiment Batches
Execute Batches A (core benchmarks & latency), B (scale $\times 64$ & headroom), C (randomized adversarial & mismatched oracle), and D (free-policy sensitivity):
```bash
python camera_ready/run_experiments.py
```
Outputs are written to `camera_ready/results.csv` and logged to `camera_ready/logs/`.

### 2. Compute Statistical Tests
Compute paired two-sided Wilcoxon signed-rank tests and 10,000-resample bootstrap 95% confidence intervals:
```bash
python camera_ready/compute_stats.py
```
Summary report is generated in `camera_ready/stats.md`.

### 3. Regenerate Publication Figures
Regenerate all publication-ready figures in `paper_plots/`:
```bash
python -m paper_plots.generate_all
```

For detailed experimental specifications and workload parameters, refer to [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Running Individual Modules

### Classical Heuristic vs Lookahead Benchmark
```bash
python -m lookahead.compare_final --workload bimodal --requests 1000
```

### Hyperparameter Ablation Benchmarks
```bash
python -m lookahead.ablation_bench --workload bimodal --requests 1000
```

### Direct Placement PPO Training
```bash
python -m training.train_direct_final
```

---

## Citation

If you use FluxAlloc in your research, please cite our paper:

```bibtex
@inproceedings{mishra2026fluxalloc,
  title        = {FluxAlloc: Adaptive Dynamic Memory Allocation},
  author       = {Mishra, Sachin and Soni, Lomesh and Gotmare, Abhay},
  booktitle    = {2026 IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
  year         = {2026},
  organization = {IEEE}
}
```

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff).

---

## License & Third-Party Notices

* FluxAlloc source code is licensed under the [MIT License](LICENSE).
* Third-party software dependencies and their licenses are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Authors & Contributors

* **Sachin Mishra** — *International Institute of Information Technology Bangalore*
* **Lomesh Soni** — *International Institute of Information Technology Bangalore*
* **Abhay Gotmare** — *International Institute of Information Technology Bangalore*
