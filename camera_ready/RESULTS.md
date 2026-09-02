# FluxAlloc Camera-Ready Results & Deliverables

This document contains the complete per-batch experimental tables, ranker facts, and headline tokens for the IEEE DSAA camera-ready submission of *FluxAlloc: Adaptive Dynamic Memory Allocation*.

## 1. Headline Search-Replace Tokens

```text
<COLLAPSE> 99.08%
<COLLAPSE_N> 4954/5000
<BM_FA> 0.3241 ± 0.0285
<BM_BF> 0.3373 ± 0.0308
<BM_P> 6.4453e-02
<BM_CI> [+0.0022, +0.0241]
<U10_FA> 0.5063 ± 0.0278
<U10_BF> 0.5061 ± 0.0305
<U10_P> 8.4375e-01
<U10_CI> [-0.0037, +0.0035]
<A10_FA> 0.0588 ± 0.0459
<A10_BF> 0.0630 ± 0.0504
<A10_P> 9.3750e-02
<A10_CI> [+0.0010, +0.0077]
<SHIELD_C> 2.1249
<SHIELD_R> 0.9788
<SHIELD_M> 0.9951
<LAT_MED> 1.2605
<LAT_MEAN> 4.4785
<LAT_MAX> 37.9817
<X64_DF> +0.0218
<H2_DF> +0.0202
<H4_DF> +0.0079
<H4U_DF> -0.0037
<SENS_FIFO> +0.2266
<SENS_LIFO> +0.0900
<SENS_RAND> +0.1325
<RHO> 0.0708
```

## 2. Neural Ranker Facts (§Ranker)

* **Model Checkpoint**: `lookahead/lookahead_ranker.pt`
* **Architecture**: 3-layer MLP (`Linear(10, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 1)`)
* **Total Parameters**: 18,049 parameters
* **Input Feature Dimensions (10D)**:
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
* **Training Hyperparameters**: Adam optimizer, learning rate = 1e-3, Loss = MSELoss, 40 epochs, batch size = 256
* **Training Trace**: Uniform workload, seed = 42, 3000 steps
* **Held-Out Evaluation**: Spearman $\rho = 0.0708$ ($p = 0.0947$) on held-out seed 999 (558 candidate pairs evaluated)
* **Seed Set Disjointness**: Training seed was 42; evaluation seed set $S = \{0, 1, 2, 3, 4, 7, 10, 18, 42, 123\}$ includes seed 42 (noted in `FAILURES.md`)

## 3. Full Per-Batch Results Tables

### Batch `A-abl`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 42 | FluxAlloc (alpha=0.0, k=12) | 0.9746 | 0.9231 | 0.3417 | 0.0020 | +0.0515 | - | - | - |
| bimodal | 42 | FluxAlloc (alpha=1.0, k=12) | 0.9648 | 0.8333 | 0.3263 | 0.0059 | +0.1315 | - | - | - |
| bimodal | 42 | FluxAlloc (k=4, alpha=0.2) | 0.9824 | 0.7778 | 0.3585 | 0.0039 | +0.2046 | - | - | - |
| bimodal | 42 | FluxAlloc (k=8, alpha=0.2) | 0.9697 | 0.8387 | 0.3221 | 0.0049 | +0.1310 | - | - | - |
| bimodal | 42 | FluxAlloc (k=16, alpha=0.2) | 0.9678 | 0.7576 | 0.3417 | 0.0078 | +0.2102 | - | - | - |

### Batch `A-adv-10`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adversarial | 0 | Best Fit | 0.6562 | 0.7727 | 0.1264 | 0.0781 | -0.1165 | - | - | - |
| adversarial | 0 | FluxAlloc | 0.6016 | 0.7451 | 0.1132 | 0.1016 | -0.1435 | - | - | - |
| adversarial | 1 | Best Fit | 0.7109 | 0.6216 | 0.0352 | 0.1094 | +0.0893 | - | - | - |
| adversarial | 1 | FluxAlloc | 0.7109 | 0.5676 | 0.0352 | 0.1250 | +0.1434 | - | - | - |
| adversarial | 2 | Best Fit | 0.6016 | 0.6471 | 0.0401 | 0.1406 | -0.0455 | - | - | - |
| adversarial | 2 | FluxAlloc | 0.4922 | 0.4462 | 0.0401 | 0.2812 | +0.0460 | - | - | - |
| adversarial | 3 | Best Fit | 0.6797 | 0.7805 | 0.1679 | 0.0703 | -0.1008 | - | - | - |
| adversarial | 3 | FluxAlloc | 0.6797 | 0.7317 | 0.1570 | 0.0859 | -0.0520 | - | - | - |
| adversarial | 4 | Best Fit | 0.7109 | 0.7297 | 0.0318 | 0.0781 | -0.0188 | - | - | - |
| adversarial | 4 | FluxAlloc | 0.6562 | 0.6591 | 0.0338 | 0.1172 | -0.0028 | - | - | - |
| adversarial | 7 | Best Fit | 0.7109 | 0.7297 | 0.0579 | 0.0781 | -0.0188 | - | - | - |
| adversarial | 7 | FluxAlloc | 0.6562 | 0.7045 | 0.0521 | 0.1016 | -0.0483 | - | - | - |
| adversarial | 10 | Best Fit | 0.8203 | 0.4783 | 0.0000 | 0.0938 | +0.3421 | - | - | - |
| adversarial | 10 | FluxAlloc | 0.8203 | 0.3043 | 0.0000 | 0.1250 | +0.5160 | - | - | - |
| adversarial | 18 | Best Fit | 0.8203 | 0.6957 | 0.0663 | 0.0547 | +0.1247 | - | - | - |
| adversarial | 18 | FluxAlloc | 0.7656 | 0.5000 | 0.0546 | 0.1172 | +0.2656 | - | - | - |
| adversarial | 42 | Best Fit | 0.7656 | 0.5667 | 0.0263 | 0.1016 | +0.1990 | - | - | - |
| adversarial | 42 | FluxAlloc | 0.7656 | 0.2667 | 0.0263 | 0.1719 | +0.4990 | - | - | - |
| adversarial | 123 | Best Fit | 0.6328 | 0.5106 | 0.0777 | 0.1797 | +0.1222 | - | - | - |
| adversarial | 123 | FluxAlloc | 0.5938 | 0.7500 | 0.0758 | 0.1016 | -0.1562 | - | - | - |

### Batch `A-adv-canonical`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adversarial | 42 | Best Fit | 0.7656 | 0.5667 | 0.0263 | 0.1016 | +0.1990 | - | - | - |
| adversarial | 42 | Random Fit | 0.7109 | 0.5405 | 0.0283 | 0.1328 | +0.1704 | - | - | - |
| adversarial | 42 | First Fit | 0.7656 | 0.7667 | 0.0263 | 0.0547 | -0.0010 | - | - | - |
| adversarial | 42 | Worst Fit | 0.5469 | 0.8276 | 0.0344 | 0.0781 | -0.2807 | - | - | - |
| adversarial | 42 | FluxAlloc (oracle=best_fit) | 0.7656 | 0.2667 | 0.0263 | 0.1719 | +0.4990 | - | - | - |
| adversarial | 42 | FluxAlloc (oracle=next_fit) | 0.7109 | 0.3243 | 0.0283 | 0.1953 | +0.3866 | - | - | - |
| adversarial | 42 | FluxAlloc (oracle=first_fit) | 0.7109 | 0.6757 | 0.0283 | 0.0938 | +0.0353 | - | - | - |

### Batch `A-bim-10`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 0 | Best Fit | 0.9512 | 0.4000 | 0.2994 | 0.0293 | +0.5512 | - | - | - |
| bimodal | 0 | FluxAlloc | 0.8906 | 0.6429 | 0.2936 | 0.0391 | +0.2478 | - | - | - |
| bimodal | 1 | Best Fit | 0.9541 | 0.7234 | 0.3583 | 0.0127 | +0.2307 | - | - | - |
| bimodal | 1 | FluxAlloc | 0.9082 | 0.7128 | 0.3554 | 0.0264 | +0.1954 | - | - | - |
| bimodal | 2 | Best Fit | 0.9688 | 0.7812 | 0.3034 | 0.0068 | +0.1875 | - | - | - |
| bimodal | 2 | FluxAlloc | 0.9453 | 0.9107 | 0.3152 | 0.0049 | +0.0346 | - | - | - |
| bimodal | 3 | Best Fit | 0.9844 | 0.8125 | 0.3633 | 0.0029 | +0.1719 | - | - | - |
| bimodal | 3 | FluxAlloc | 0.9756 | 0.8400 | 0.3439 | 0.0039 | +0.1356 | - | - | - |
| bimodal | 4 | Best Fit | 0.9658 | 0.8000 | 0.3545 | 0.0068 | +0.1658 | - | - | - |
| bimodal | 4 | FluxAlloc | 0.9561 | 0.8889 | 0.3291 | 0.0049 | +0.0672 | - | - | - |
| bimodal | 7 | Best Fit | 0.9883 | 0.7500 | 0.3847 | 0.0029 | +0.2383 | - | - | - |
| bimodal | 7 | FluxAlloc | 0.9785 | 0.8636 | 0.3806 | 0.0029 | +0.1149 | - | - | - |
| bimodal | 10 | Best Fit | 0.9619 | 0.8205 | 0.3275 | 0.0068 | +0.1414 | - | - | - |
| bimodal | 10 | FluxAlloc | 0.9287 | 0.8493 | 0.2928 | 0.0107 | +0.0794 | - | - | - |
| bimodal | 18 | Best Fit | 0.9814 | 0.7368 | 0.3469 | 0.0049 | +0.2446 | - | - | - |
| bimodal | 18 | FluxAlloc | 0.9766 | 0.8750 | 0.3048 | 0.0029 | +0.1016 | - | - | - |
| bimodal | 42 | Best Fit | 0.9824 | 0.7778 | 0.3431 | 0.0039 | +0.2046 | - | - | - |
| bimodal | 42 | FluxAlloc | 0.9766 | 0.7500 | 0.3207 | 0.0059 | +0.2266 | - | - | - |
| bimodal | 123 | Best Fit | 0.9658 | 0.7714 | 0.2917 | 0.0078 | +0.1944 | - | - | - |
| bimodal | 123 | FluxAlloc | 0.9590 | 0.8333 | 0.3048 | 0.0068 | +0.1257 | - | - | - |

### Batch `A-bim-s42`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 42 | Best Fit | 0.9824 | 0.7778 | 0.3431 | 0.0039 | +0.2046 | - | - | - |
| bimodal | 42 | Random Fit | 0.9697 | 0.8710 | 0.3501 | 0.0039 | +0.0988 | - | - | - |
| bimodal | 42 | First Fit | 0.9570 | 0.8864 | 0.3417 | 0.0049 | +0.0707 | - | - | - |
| bimodal | 42 | Worst Fit | 0.8877 | 0.9739 | 0.3459 | 0.0029 | -0.0862 | - | - | - |
| bimodal | 42 | Baseline PPO | 0.9629 | 0.7895 | 0.3417 | 0.0078 | +0.1734 | - | - | - |
| bimodal | 42 | FluxAlloc | 0.9766 | 0.7500 | 0.3207 | 0.0059 | +0.2266 | - | - | - |

### Batch `A-latency`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| uniform | 42 | Best Fit | 0.8340 | 0.6765 | 0.4444 | 0.0537 | +0.1575 | 0.0087 | 0.0076 | 0.0365 |
| uniform | 42 | FluxAlloc MLP-only | 0.7979 | 0.8116 | 0.4558 | 0.0381 | -0.0137 | 0.3084 | 0.3008 | 0.6094 |
| uniform | 42 | FluxAlloc Full Planning | 0.8105 | 0.8093 | 0.4416 | 0.0361 | +0.0013 | 4.4785 | 1.2605 | 37.9817 |

### Batch `A-unif-10`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| uniform | 0 | Best Fit | 0.8066 | 0.8333 | 0.5260 | 0.0322 | -0.0267 | - | - | - |
| uniform | 0 | FluxAlloc | 0.7695 | 0.7500 | 0.5148 | 0.0576 | +0.0195 | - | - | - |
| uniform | 1 | Best Fit | 0.8711 | 0.6667 | 0.4649 | 0.0430 | +0.2044 | - | - | - |
| uniform | 1 | FluxAlloc | 0.9424 | 0.5932 | 0.4664 | 0.0234 | +0.3492 | - | - | - |
| uniform | 2 | Best Fit | 0.8760 | 0.7402 | 0.5377 | 0.0322 | +0.1358 | - | - | - |
| uniform | 2 | FluxAlloc | 0.8652 | 0.7174 | 0.5349 | 0.0381 | +0.1478 | - | - | - |
| uniform | 3 | Best Fit | 0.8057 | 0.7789 | 0.5426 | 0.0430 | +0.0268 | - | - | - |
| uniform | 3 | FluxAlloc | 0.8447 | 0.7987 | 0.5412 | 0.0312 | +0.0460 | - | - | - |
| uniform | 4 | Best Fit | 0.8848 | 0.8475 | 0.4531 | 0.0176 | +0.0373 | - | - | - |
| uniform | 4 | FluxAlloc | 0.8770 | 0.7937 | 0.4589 | 0.0254 | +0.0833 | - | - | - |
| uniform | 7 | Best Fit | 0.8408 | 0.7975 | 0.5324 | 0.0322 | +0.0433 | - | - | - |
| uniform | 7 | FluxAlloc | 0.8359 | 0.7440 | 0.5324 | 0.0420 | +0.0919 | - | - | - |
| uniform | 10 | Best Fit | 0.9121 | 0.8333 | 0.5064 | 0.0146 | +0.0788 | - | - | - |
| uniform | 10 | FluxAlloc | 0.8477 | 0.7821 | 0.5120 | 0.0332 | +0.0656 | - | - | - |
| uniform | 18 | Best Fit | 0.9219 | 0.7375 | 0.4899 | 0.0205 | +0.1844 | - | - | - |
| uniform | 18 | FluxAlloc | 0.9023 | 0.7800 | 0.4899 | 0.0215 | +0.1223 | - | - | - |
| uniform | 42 | Best Fit | 0.8438 | 0.6625 | 0.5134 | 0.0527 | +0.1813 | - | - | - |
| uniform | 42 | FluxAlloc | 0.8643 | 0.8489 | 0.5078 | 0.0205 | +0.0153 | - | - | - |
| uniform | 123 | Best Fit | 0.8330 | 0.7251 | 0.4943 | 0.0459 | +0.1079 | - | - | - |
| uniform | 123 | FluxAlloc | 0.8096 | 0.8103 | 0.5043 | 0.0361 | -0.0007 | - | - | - |

### Batch `A-unif-s42`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| uniform | 42 | Best Fit | 0.8438 | 0.6625 | 0.5134 | 0.0527 | +0.1813 | - | - | - |
| uniform | 42 | Random Fit | 0.8613 | 0.7676 | 0.5219 | 0.0322 | +0.0937 | - | - | - |
| uniform | 42 | First Fit | 0.8018 | 0.7685 | 0.5177 | 0.0459 | +0.0333 | - | - | - |
| uniform | 42 | Worst Fit | 0.7402 | 0.9248 | 0.5233 | 0.0195 | -0.1846 | - | - | - |
| uniform | 42 | Baseline PPO | 0.8926 | 0.7091 | 0.5191 | 0.0312 | +0.1835 | - | - | - |
| uniform | 42 | FluxAlloc | 0.8643 | 0.8489 | 0.5078 | 0.0205 | +0.0153 | - | - | - |

### Batch `B-h2`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 0 | Best Fit | 0.9824 | 0.7222 | 0.2180 | 0.0049 | +0.2602 | - | - | - |
| bimodal | 0 | FluxAlloc | 0.9165 | 0.7485 | 0.1904 | 0.0210 | +0.1680 | - | - | - |
| bimodal | 7 | Best Fit | 0.9805 | 0.2250 | 0.2592 | 0.0151 | +0.7555 | - | - | - |
| bimodal | 7 | FluxAlloc | 0.9800 | 0.9268 | 0.2510 | 0.0015 | +0.0532 | - | - | - |
| bimodal | 10 | Best Fit | 0.9512 | 0.6800 | 0.2203 | 0.0156 | +0.2712 | - | - | - |
| bimodal | 10 | FluxAlloc | 0.9517 | 0.7475 | 0.2014 | 0.0122 | +0.2042 | - | - | - |
| bimodal | 18 | Best Fit | 0.9961 | 0.3750 | 0.2331 | 0.0024 | +0.6211 | - | - | - |
| bimodal | 18 | FluxAlloc | 0.9751 | 0.7647 | 0.2051 | 0.0059 | +0.2104 | - | - | - |
| bimodal | 42 | Best Fit | 0.9731 | 0.2364 | 0.2451 | 0.0205 | +0.7368 | - | - | - |
| bimodal | 42 | FluxAlloc | 0.9644 | 0.7397 | 0.2269 | 0.0093 | +0.2246 | - | - | - |

### Batch `B-h4`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 0 | Best Fit | 0.9819 | 0.7297 | 0.1032 | 0.0049 | +0.2522 | - | - | - |
| bimodal | 0 | FluxAlloc | 0.9482 | 0.8208 | 0.0988 | 0.0093 | +0.1275 | - | - | - |
| bimodal | 7 | Best Fit | 0.9946 | 0.0000 | 0.1392 | 0.0054 | +0.9946 | - | - | - |
| bimodal | 7 | FluxAlloc | 0.9893 | 0.8182 | 0.1296 | 0.0020 | +0.1711 | - | - | - |
| bimodal | 10 | Best Fit | 0.9792 | 0.4235 | 0.1072 | 0.0120 | +0.5557 | - | - | - |
| bimodal | 10 | FluxAlloc | 0.9578 | 0.8439 | 0.0928 | 0.0066 | +0.1138 | - | - | - |
| bimodal | 18 | Best Fit | 0.9915 | 0.2286 | 0.1166 | 0.0066 | +0.7629 | - | - | - |
| bimodal | 18 | FluxAlloc | 0.9814 | 0.9342 | 0.1053 | 0.0012 | +0.0472 | - | - | - |
| bimodal | 42 | Best Fit | 0.9885 | 0.2553 | 0.1373 | 0.0085 | +0.7332 | - | - | - |
| bimodal | 42 | FluxAlloc | 0.9734 | 0.6881 | 0.1373 | 0.0083 | +0.2853 | - | - | - |

### Batch `B-unif-h4`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| uniform | 0 | Best Fit | 0.9348 | 0.8315 | 0.3418 | 0.0110 | +0.1034 | - | - | - |
| uniform | 0 | FluxAlloc | 0.9014 | 0.8787 | 0.3516 | 0.0120 | +0.0227 | - | - | - |
| uniform | 7 | Best Fit | 0.9309 | 0.8940 | 0.3531 | 0.0073 | +0.0369 | - | - | - |
| uniform | 7 | FluxAlloc | 0.9177 | 0.9110 | 0.3531 | 0.0073 | +0.0067 | - | - | - |
| uniform | 10 | Best Fit | 0.9492 | 0.8942 | 0.3225 | 0.0054 | +0.0550 | - | - | - |
| uniform | 10 | FluxAlloc | 0.9194 | 0.9030 | 0.3267 | 0.0078 | +0.0164 | - | - | - |
| uniform | 18 | Best Fit | 0.9551 | 0.8533 | 0.3175 | 0.0066 | +0.1018 | - | - | - |
| uniform | 18 | FluxAlloc | 0.9131 | 0.8062 | 0.3218 | 0.0168 | +0.1069 | - | - | - |
| uniform | 42 | Best Fit | 0.9509 | 0.8308 | 0.3465 | 0.0083 | +0.1201 | - | - | - |
| uniform | 42 | FluxAlloc | 0.9277 | 0.8243 | 0.3465 | 0.0127 | +0.1034 | - | - | - |

### Batch `B-x64`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 0 | Best Fit | 0.9512 | 0.4000 | 0.2994 | 0.0293 | +0.5512 | - | - | - |
| bimodal | 0 | FluxAlloc | 0.8906 | 0.6429 | 0.2936 | 0.0391 | +0.2478 | - | - | - |
| bimodal | 7 | Best Fit | 0.9883 | 0.7500 | 0.3847 | 0.0029 | +0.2383 | - | - | - |
| bimodal | 7 | FluxAlloc | 0.9785 | 0.8636 | 0.3806 | 0.0029 | +0.1149 | - | - | - |
| bimodal | 10 | Best Fit | 0.9619 | 0.8205 | 0.3275 | 0.0068 | +0.1414 | - | - | - |
| bimodal | 10 | FluxAlloc | 0.9287 | 0.8493 | 0.2928 | 0.0107 | +0.0794 | - | - | - |
| bimodal | 18 | Best Fit | 0.9814 | 0.7368 | 0.3469 | 0.0049 | +0.2446 | - | - | - |
| bimodal | 18 | FluxAlloc | 0.9766 | 0.8750 | 0.3048 | 0.0029 | +0.1016 | - | - | - |
| bimodal | 42 | Best Fit | 0.9824 | 0.7778 | 0.3431 | 0.0039 | +0.2046 | - | - | - |
| bimodal | 42 | FluxAlloc | 0.9766 | 0.7500 | 0.3207 | 0.0059 | +0.2266 | - | - | - |

### Batch `C-mism`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adversarial_rand | 0 | FluxAlloc (mismatched) | 0.9863 | 0.7143 | 0.8734 | 0.0039 | +0.2720 | - | - | - |
| adversarial_rand | 1 | FluxAlloc (mismatched) | 0.9941 | 0.6667 | 0.9426 | 0.0020 | +0.3275 | - | - | - |
| adversarial_rand | 2 | FluxAlloc (mismatched) | 0.9795 | 0.8095 | 0.9172 | 0.0039 | +0.1700 | - | - | - |
| adversarial_rand | 3 | FluxAlloc (mismatched) | 0.9873 | 0.6923 | 0.9476 | 0.0039 | +0.2950 | - | - | - |
| adversarial_rand | 4 | FluxAlloc (mismatched) | 1.0000 | 0.0000 | 0.9023 | 0.0000 | +1.0000 | - | - | - |
| adversarial_rand | 7 | FluxAlloc (mismatched) | 0.9951 | 0.8000 | 0.9339 | 0.0010 | +0.1951 | - | - | - |
| adversarial_rand | 10 | FluxAlloc (mismatched) | 0.9922 | 0.6250 | 0.9349 | 0.0029 | +0.3672 | - | - | - |
| adversarial_rand | 18 | FluxAlloc (mismatched) | 0.9932 | 0.7143 | 0.8761 | 0.0020 | +0.2789 | - | - | - |
| adversarial_rand | 42 | FluxAlloc (mismatched) | 0.9854 | 0.7333 | 0.9031 | 0.0039 | +0.2520 | - | - | - |
| adversarial_rand | 123 | FluxAlloc (mismatched) | 0.9932 | 0.7143 | 0.9552 | 0.0020 | +0.2789 | - | - | - |

### Batch `C-rand`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adversarial_rand | 0 | Best Fit | 0.9863 | 0.7857 | 0.8723 | 0.0029 | +0.2006 | - | - | - |
| adversarial_rand | 0 | FluxAlloc (matched) | 0.9834 | 0.7647 | 0.8723 | 0.0039 | +0.2187 | - | - | - |
| adversarial_rand | 1 | Best Fit | 0.9912 | 0.7778 | 0.9488 | 0.0020 | +0.2134 | - | - | - |
| adversarial_rand | 1 | FluxAlloc (matched) | 0.9941 | 0.6667 | 0.9426 | 0.0020 | +0.3275 | - | - | - |
| adversarial_rand | 2 | Best Fit | 0.9863 | 0.7143 | 0.9161 | 0.0039 | +0.2720 | - | - | - |
| adversarial_rand | 2 | FluxAlloc (matched) | 0.9824 | 0.8333 | 0.9161 | 0.0029 | +0.1491 | - | - | - |
| adversarial_rand | 3 | Best Fit | 0.9834 | 0.7647 | 0.9466 | 0.0039 | +0.2187 | - | - | - |
| adversarial_rand | 3 | FluxAlloc (matched) | 0.9854 | 0.6667 | 0.9476 | 0.0049 | +0.3187 | - | - | - |
| adversarial_rand | 4 | Best Fit | 0.9971 | 0.3333 | 0.9023 | 0.0020 | +0.6637 | - | - | - |
| adversarial_rand | 4 | FluxAlloc (matched) | 0.9941 | 0.6667 | 0.9023 | 0.0020 | +0.3275 | - | - | - |
| adversarial_rand | 7 | Best Fit | 0.9961 | 0.7500 | 0.9421 | 0.0010 | +0.2461 | - | - | - |
| adversarial_rand | 7 | FluxAlloc (matched) | 0.9961 | 0.7500 | 0.9411 | 0.0010 | +0.2461 | - | - | - |
| adversarial_rand | 10 | Best Fit | 0.9941 | 0.6667 | 0.9349 | 0.0020 | +0.3275 | - | - | - |
| adversarial_rand | 10 | FluxAlloc (matched) | 0.9922 | 0.7500 | 0.9369 | 0.0020 | +0.2422 | - | - | - |
| adversarial_rand | 18 | Best Fit | 0.9893 | 0.6364 | 0.8739 | 0.0039 | +0.3529 | - | - | - |
| adversarial_rand | 18 | FluxAlloc (matched) | 0.9912 | 0.5556 | 0.8739 | 0.0039 | +0.4357 | - | - | - |
| adversarial_rand | 42 | Best Fit | 0.9941 | 0.5000 | 0.9031 | 0.0029 | +0.4941 | - | - | - |
| adversarial_rand | 42 | FluxAlloc (matched) | 0.9873 | 0.6923 | 0.9073 | 0.0039 | +0.2950 | - | - | - |
| adversarial_rand | 123 | Best Fit | 0.9951 | 0.8000 | 0.9552 | 0.0010 | +0.1951 | - | - | - |
| adversarial_rand | 123 | FluxAlloc (matched) | 0.9951 | 0.6000 | 0.9552 | 0.0020 | +0.3951 | - | - | - |

### Batch `D-freepol`

| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bimodal | 42 | FluxAlloc (free=LIFO) | 0.9746 | 0.8846 | 0.3305 | 0.0029 | +0.0900 | - | - | - |
| bimodal | 42 | FluxAlloc (free=random) | 0.9473 | 0.8148 | 0.3305 | 0.0098 | +0.1325 | - | - | - |
| adversarial | 42 | FluxAlloc (free=LIFO) | 0.7656 | 0.2667 | 0.0263 | 0.1719 | +0.4990 | - | - | - |
| adversarial | 42 | FluxAlloc (free=random) | 0.7656 | 0.2667 | 0.0263 | 0.1719 | +0.4990 | - | - | - |

