# IDE AGENT TASK SPECIFICATION — FluxAlloc Camera-Ready Regeneration

You are executing the experimental regeneration for the camera-ready of the
accepted IEEE DSAA paper "FluxAlloc: Adaptive Dynamic Memory Allocation".
Reviewer comments require: (1) fixing a suspected trace-generator seeding bug
(seeds 7 and 18 produced identical results), (2) multi-seed evaluation of ALL
workloads with significance tests, (3) scale/headroom sensitivity runs,
(4) randomized-adversarial and mismatched-oracle controls, (5) free-policy
sensitivity, (6) regenerated figures. **Your job ends at delivering data
artifacts. You do NOT edit the paper .tex file.**

---

## 0. OPERATING RULES (violations = failed task)

1. **No fabricated numbers.** Every value in every deliverable must come from
   an executed run logged in `camera_ready/logs/`. If a run fails, record it
   in `FAILURES.md` and move on.
2. **Do not modify method semantics.** You may ADD optional config flags
   (scale, headroom, free-policy, randomized adversarial, oracle hypothesis)
   whose defaults reproduce current behaviour exactly. You may FIX the trace
   generator seeding (Phase 1). Nothing else in the allocator/RL code changes.
3. **Determinism & provenance.** Pin and record Python/numpy/torch versions in
   `camera_ready/env.txt`. Every run logs its full config into `results.csv`.
4. **Version control.** Work on branch `camera-ready`. One commit per phase:
   `[cr] phase0 repo map`, `[cr] phase1 trace audit`, etc.
5. **All artifacts live in `camera_ready/`** (create it at repo root).
6. **If blocked > 30 min**, write the blocker to `FAILURES.md` with what you
   tried, then continue with the next priority item.
7. Priority order is strict: Phase 1 → Batch A → Phase 5 stats → Batches B–D
   → Phase 6 plots. Never starve a higher-priority item.

---

## 1. PHASE 0 — RECONNAISSANCE (output: `camera_ready/REPO_MAP.md`)

Explore the repo and record exact file paths + function names for:
- [ ] Heap environment (split/coalescing, metrics computation)
- [ ] Workload generators: uniform, bimodal, adversarial (note EVERY place a
      RNG is drawn from and how `seed` enters)
- [ ] Strategy implementations: BestFit, FirstFit, WorstFit, RandomFit,
      NextFit, BaselinePPO (checkpoint path), FluxAlloc/LookaheadAllocator
- [ ] FluxAlloc internals: rollout loop (what it assumes about future mallocs
      and frees — critical for Phase 3 Batch C), ranker MLP (architecture,
      checkpoint `lookahead_ranker.pt`, training script, training seeds)
- [ ] Latency benchmark script; ablation scripts; PPO policy-analysis script
      (the one that produced "4,954 of 5,000"); plotting scripts
- [ ] Entry point/CLI used to run one strategy on one trace

Also verify: does the rollout (a) sample future requests from a hypothesised
generator, or (b) peek at the true upcoming requests? Record the answer in
`REPO_MAP.md` — it determines the mismatched-oracle implementation in Batch C.

## 2. PHASE 1 — TRACE INTEGRITY AUDIT (critical)

1. Write `camera_ready/trace_audit.py`: for seeds
   S = {0,1,2,3,4,5,7,10,18,42,123}, generate uniform/bimodal/adversarial
   traces EXACTLY as the current generator does; write
   `sha256(request_array.bytes)` per (workload, seed) to
   `camera_ready/trace_manifest.csv`.
2. **Assert uniqueness per workload.** Expected failure of this assertion:
   seed 18 == seed 7 (reviewers observed identical table rows).
3. If duplicates exist: fix the generator so ALL stochastic draws (sizes,
   cycle lengths, free patterns, any internal choices) flow from one
   `rng = np.random.default_rng(seed)` (or equivalent per-seed stream).
   Keep the same seed values. Save the diff; document in `ASSUMPTIONS.md`.
4. Re-run the audit; assertion must pass. Then run a canary:
   BF and FA on bimodal seeds 7 and 18 — results MUST now differ.
5. NOTE: fixing the generator may change traces for ALL seeds. This is
   expected. Everything is regenerated downstream; no old/new mixing.

## 3. PHASE 2 — FREEZE

`git checkout -b camera-ready && git tag cr-freeze` AFTER the Phase-1 fix.
Record versions in `env.txt`.

## 4. PHASE 3 — EXPERIMENT RUNS

Schema for `camera_ready/results.csv` (one row per run):
```
config,workload,seed,strategy,util,frag,fail,lgfr,score,hit,miss,lat_mean_ms,lat_med_ms,lat_max_ms
```
`hit = 1 - fail`. Latency columns filled only for `A-latency`.
Parallelise across seeds (independent); use ≤ (cores − 1) workers.

Seed-10 set: {0,1,2,3,4,7,10,18,42,123}. Seed-5 set: {0,7,10,18,42}.

### Batch A — core regeneration (HIGHEST priority)
| config | workload | seeds | strategies |
|---|---|---|---|
| A-unif-s42 | uniform | {42} | BF, RF, FF, WF, PPO, FA |
| A-bim-s42 | bimodal | {42} | BF, RF, FF, WF, PPO, FA |
| A-bim-10 | bimodal | Seed-10 | BF, FA |
| A-unif-10 | uniform | Seed-10 | BF, FA |
| A-adv-canonical | adversarial | {42} | BF, RF, FF, WF + FA(oracle∈{BF,Next,First}) |
| A-adv-10 | adversarial | Seed-10 | BF, FA |
| A-abl | bimodal | {42} | FA with α∈{0.0,1.0} (k=12); k∈{4,8,16} (α=0.2) |
| A-ppo-eval | (as original script) | — | re-run PPO policy-analysis script unchanged; log per-action counts |
| A-latency | uniform | {42}, n=500 | BF; FA-MLP-only; FA-full; per-decision wall clock |

### Batch B — scale & regime (new flags; defaults = old behaviour)
| config | change | workload | seeds | strategies |
|---|---|---|---|---|
| B-x64 | heap=65536; ALL sizes ×64 | bimodal | Seed-5 | BF, FA |
| B-x1024 (OPTIONAL) | heap=1048576; sizes ×1024 | bimodal | Seed-5 | BF, FA |
| B-h2 / B-h4 | heap=2048 / 4096; sizes unchanged | bimodal | Seed-5 | BF, FA |
| B-unif-h4 | heap=4096 | uniform | Seed-5 | BF, FA |

### Batch C — adversarial robustness
1. Add generator variant `adversarial_rand` (new flag, do not touch old one):
   small ~U{6,10}, large ~U{42,70}, cycle length ~Geometric(mean 8),
   each small block freed w.p. 0.8; n=1000; seeded per Seed-10.
2. `C-rand`: on these traces run BF and FA(matched hypothesis).
3. `C-mism`: FA with MISMATCHED hypothesis on the same traces.
   - If rollout samples a hypothesised generator (REPO_MAP answer a):
     hypothesis = canonical (deterministic 8/56 alternating cycle).
   - If rollout peeks at true future requests (answer b): perturb the peeked
     window (replace sizes with canonical 8/56 cycle).
   Document the chosen implementation in `ASSUMPTIONS.md`.

### Batch D — free-policy sensitivity
`D-freepol`: FA with rollout free-policy ∈ {LIFO, random} (FIFO = existing
canonical runs). Workloads: bimodal seed 42; adversarial seed 42.

### Verification gate (after Batch A)
Add flags must not change canonical outputs: run B-configs with default flags
and assert equality (tol 1e-9) with A outputs for the same (workload,seed,
strategy). Log pass/fail in `ASSUMPTIONS.md`.

## 5. PHASE 4 — RANKER FACTS (no env runs)

Extract from code/training logs into `RESULTS.md` §Ranker: the 10 input
features (exact names/order); hidden layer sizes & param count; activation;
loss; optimizer+lr+epochs; training-trace seed set; checkpoint filename.
Compute held-out Spearman ρ between MLP predictions and rollout scores on a
held-out pair set (pure forward pass; minutes). **Assert training seeds ∩
eval seeds = ∅**; if violated, record in `FAILURES.md` (do not retrain).

## 6. PHASE 5 — STATISTICS (output: `camera_ready/stats.md`)

Use scipy (install if missing). For each of {bimodal-10, uniform-10,
adv-10, adv-rand-10}: per-strategy mean±sd (fail); paired Δ = BF−FA per seed;
**two-sided paired Wilcoxon signed-rank p** (`scipy.stats.wilcoxon`);
bootstrap 95% percentile CI of mean Δ (10,000 resamples).
β-family: from bimodal-10 per-seed (util,frag): Score_β = util − β·frag,
β∈{0.5,1,2}; report #seeds with FA>BF per β.
Shield ratios: canonical = frag_BF/frag_FA (seed 42); randomized = mean of
per-seed ratios (C-rand); mismatched likewise (C-mism vs BF).
Scale/headroom: per-config mean Δfail over Seed-5 + Wilcoxon p (note n=5).

## 7. PHASE 6 — PLOTS (output: regenerated figures)

Back up current figures to `paper_plots_old/`, then regenerate IN PLACE
(same filenames, ≥300 dpi, same style as existing scripts) from
`results.csv`:
plot1_workloads.png (seed-42 scores, 6 strategies, uniform+bimodal);
plot2_latency_accuracy.png; plot3_ablations.png; plot4_ppo_distribution.png;
plot5_adversarial_sweep.png; plot6_multi_seed_reliability.png (**10 seeds**,
bands = ±1 sd). No new figure types.

## 8. PHASE 7 — DELIVERABLES

1. `camera_ready/RESULTS.md` — full per-batch tables (every row of
   results.csv rendered as markdown) + §Ranker facts + §Headline tokens:
   fill EXACTLY these tokens (Writer will search-replace into the paper):
   <COLLAPSE> <COLLAPSE_N> | <BM_FA> <BM_BF> <BM_P> <BM_CI> |
   <U10_FA> <U10_BF> <U10_P> <U10_CI> | <A10_FA> <A10_BF> <A10_P> <A10_CI> |
   <SHIELD_C> <SHIELD_R> <SHIELD_M> | <LAT_MED> <LAT_MEAN> <LAT_MAX> |
   <X64_DF> <H2_DF> <H4_DF> <H4U_DF> | <SENS_FIFO> <SENS_LIFO> <SENS_RAND> |
   <RHO>
2. `camera_ready/DELTA.md` — new vs. paper-current values (table below) with
   CHANGED/UNCHANGED flags per number.
3. `ASSUMPTIONS.md`, `FAILURES.md`, `REPO_MAP.md`, `env.txt`,
   `trace_manifest.csv`, `results.csv`, `stats.md`, `logs/`.
4. Final chat summary: 10 lines max — what ran, what failed, headline tokens.

### Paper-current values for DELTA.md
PPO: 99.08% (4954/5000); RF 23; FF 19; WF 4.
Uniform s42: BF .8438/.6625/.5134/+.1813 · RF .8613/.7676/.5219/+.0937 ·
FF .8018/.7685/.5177/+.0333 · FA .8643/.8489/.5078/+.0153 ·
PPO .7754/.8217/.5205/−.0463 · WF .7402/.9248/.5233/−.1846.
Bimodal s42: FA .9766/.7500/.3207/.0059/+.2266 · BF .9824/.7778/.3431/+.2046.
Multi-seed means: FA .3185±.0326 · BF .3403±.0276.
Adversarial: BF +.1990 (frag .5667) · FA-BForacle +.4990 (frag .2667) ·
NF +.3866 · FF +.0353 · WF −.2807 · FFfit −.0010 · RF +.1704.
Latency: BF .0143/.0125/.0504 · MLP .1238/.0846/1.8345 ·
full 5.2610/.9149/80.0109.
Ablations: α0 +.0515 · α1 +.1315 · α.2 +.2266 · k4 +.2046 · k8 +.1310 ·
k12 +.2266 · k16 +.2102.

## 9. FALLBACK LADDER (if compute/time overruns)

Drop in this order: B-x1024 → C-mism to Seed-5 → B seeds to 5 (already) →
A-unif-10 to Seed-5. NEVER drop: Phase 1, A-bim-10 (≥10 seeds), A-adv-10,
C-rand, Phase 5 stats.

## 10. DEFINITION OF DONE

- [ ] trace_manifest hashes unique; canary (7≠18) passed
- [ ] results.csv contains every config above (or FAILURES.md explains)
- [ ] stats.md has Wilcoxon p + CI for all four 10-seed comparisons
- [ ] six figures regenerated; backups exist
- [ ] RESULTS.md tokens filled; DELTA.md complete
