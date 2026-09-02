# FluxAlloc Execution Failures & Blockers Log

This file records any run failures, exceptions, or blockers encountered during the camera-ready experimental regeneration.

---

## Phase 4 Ranker Evaluation Note: Training/Eval Seed Overlap
* **Finding**: The ranker checkpoint `lookahead_ranker.pt` was trained on seed 42 (uniform workload, 3000 steps). Evaluation seed set $S = \{0, 1, 2, 3, 4, 7, 10, 18, 42, 123\}$ includes seed 42.
* **Status**: Condition $\text{training seeds} \cap \text{eval seeds} = \emptyset$ is violated specifically for seed 42 ($\{42\} \cap S = \{42\}$).
* **Action**: Per Phase 4 instructions, recorded in `FAILURES.md` and `RESULTS.md` without retraining the model checkpoint. Held-out Spearman correlation was evaluated on seed 999 ($\rho = 0.0708$, $p = 0.0947$).
