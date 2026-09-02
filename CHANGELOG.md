# Changelog

All notable changes to the **FluxAlloc** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-30

### Added
- **Direct Placement Environment (`core/rl_env_direct_final.py`)**: Custom Gymnasium MDP simulating dynamic memory heaps with action masking for valid allocations.
- **Transformer Policy Extractor (`policy/custom_policy_transformer.py`)**: Multi-head self-attention feature extractor for raw memory block states.
- **Lookahead + Neural Ranker Hybrid (`lookahead/`)**:
  - `NeuralRanker` 10-dimensional MLP scoring network (`lookahead/neural_ranker.py`).
  - Simulation rollout engine with configurable lookahead horizon $k \in \{4, 8, 12, 16\}$ and blend weight $\alpha \in [0.0, 1.0]$ (`lookahead/lookahead_sim.py`).
  - Support for custom free policies (`FIFO`, `LIFO`, `random`) and oracle strategies (`best_fit`, `first_fit`, `next_fit`).
- **Reproducibility & Verification Harness (`camera_ready/`)**:
  - `run_experiments.py` executing Batches A (core benchmarks & latency), B (scale $\times 64$ & headroom), C (randomized & mismatched adversarial), and D (free policies).
  - SHA-256 trace verification audit (`camera_ready/trace_audit.py` and `camera_ready/trace_manifest.csv`).
  - Comprehensive statistical suite (`camera_ready/compute_stats.py`) computing paired two-sided Wilcoxon signed-rank tests and 10,000-resample bootstrap 95% confidence intervals.
- **Academic Plots Suite (`paper_plots/`)**: Publication-ready vector graphics and charts at $\ge 300$ DPI.
- **Standardized Documentation**: Comprehensive architecture documentation, experiment reports, reproducibility guides, and citation metadata (`CITATION.cff`, `CITATION.bib`, `THIRD_PARTY_NOTICES.md`).

### Changed
- Refactored `core/workload_generator.py` to ensure per-seed isolated PRNG streams with complete deterministic trace reproducibility across all evaluation seeds.
- Updated requirements to fully specify `sb3-contrib` and `scipy`.
