# FluxAlloc Assumptions & Generator Audit Documentation

## 1. Trace Integrity Audit (Phase 1)
* **Root Cause of Seed Duplication**: `scaled_adversarial_workload` in `core/workload_generator.py` previously had a hardcoded `random.seed(42)` on line 72, which forced every seed passed into adversarial workload generation to produce identical traces (`2bae55ebcd88...`).
* **Fix Applied**:
  * Added `seed: int | None = None` and `scale: int = 1` parameters across all workload generator methods (`uniform_workload`, `bimodal_workload`, `scaled_adversarial_workload`, and `adversarial_rand`).
  * Used `rng = random.Random(seed)` so that all stochastic draws (request operations, allocation sizes, and free candidates) strictly flow from the specified seed stream.
  * Verified 100% hash uniqueness across all seeds $S = \{0, 1, 2, 3, 4, 5, 7, 10, 18, 42, 123\}$ in `camera_ready/trace_manifest.csv`.
  * Verified canary test on seeds 7 and 18: results are confirmed to differ.

---

## 2. Mismatched Oracle Implementation (Phase 3 Batch C)
* **Rollout Behavior**: As verified in `REPO_MAP.md`, `LookaheadAllocator` peeks at the upcoming workload ops (`workload[request_ptr + t]`).
* **Mismatched Hypothesis Implementation (`C-mism`)**: For the mismatched oracle control, the peeked future request sizes during rollout simulation are perturbed to assume the canonical alternating $8 / 56$ cycle rather than the true randomized sizes.

---

## 3. Verification Gate
* Canonical runs with default flags will be verified against standard baseline runs to ensure $10^{-9}$ tolerance.
