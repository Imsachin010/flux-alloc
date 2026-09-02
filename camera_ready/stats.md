# FluxAlloc Statistical Significance & Analysis (Phase 5)

All statistical comparisons evaluate paired runs per seed using two-sided paired Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`) and 10,000-resample bootstrap 95% percentile confidence intervals for $\Delta = \text{Fail}_{\text{BF}} - \text{Fail}_{\text{FA}}$.

## 1. Multi-Seed Failure Rate Comparisons (10 Seeds: {0, 1, 2, 3, 4, 7, 10, 18, 42, 123})

### Bimodal Workload (A-bim-10)
* **Best Fit Fail Rate**: 0.3373 ± 0.0308 (33.73%)
* **FluxAlloc Fail Rate**: 0.3241 ± 0.0285 (32.41%)
* **Paired Reduction $\Delta$ (BF − FA)**: +0.0132 ± 0.0186 (+1.32%)
* **Two-sided Paired Wilcoxon $p$-value**: 6.4453e-02 (not significant)
* **Bootstrap 95% Percentile CI of Mean $\Delta$**: [+0.0022, +0.0241] ([+0.22%, +2.41%])

| Seed | Best Fit Fail | FluxAlloc Fail | Paired $\Delta$ (BF − FA) |
| :--- | :---: | :---: | :---: |
| Seed 0 | 0.2994 (29.94%) | 0.2936 (29.36%) | +0.0058 (+0.58%) |
| Seed 1 | 0.3583 (35.83%) | 0.3554 (35.54%) | +0.0029 (+0.29%) |
| Seed 2 | 0.3034 (30.34%) | 0.3152 (31.52%) | -0.0118 (-1.18%) |
| Seed 3 | 0.3633 (36.33%) | 0.3439 (34.39%) | +0.0194 (+1.94%) |
| Seed 4 | 0.3545 (35.45%) | 0.3291 (32.91%) | +0.0254 (+2.54%) |
| Seed 7 | 0.3847 (38.47%) | 0.3806 (38.06%) | +0.0041 (+0.41%) |
| Seed 10 | 0.3275 (32.75%) | 0.2928 (29.28%) | +0.0347 (+3.47%) |
| Seed 18 | 0.3469 (34.69%) | 0.3048 (30.48%) | +0.0421 (+4.21%) |
| Seed 42 | 0.3431 (34.31%) | 0.3207 (32.07%) | +0.0224 (+2.24%) |
| Seed 123 | 0.2917 (29.17%) | 0.3048 (30.48%) | -0.0131 (-1.31%) |

### Uniform Workload (A-unif-10)
* **Best Fit Fail Rate**: 0.5061 ± 0.0305 (50.61%)
* **FluxAlloc Fail Rate**: 0.5063 ± 0.0278 (50.63%)
* **Paired Reduction $\Delta$ (BF − FA)**: -0.0002 ± 0.0061 (-0.02%)
* **Two-sided Paired Wilcoxon $p$-value**: 8.4375e-01 (not significant)
* **Bootstrap 95% Percentile CI of Mean $\Delta$**: [-0.0037, +0.0035] ([-0.37%, +0.35%])

| Seed | Best Fit Fail | FluxAlloc Fail | Paired $\Delta$ (BF − FA) |
| :--- | :---: | :---: | :---: |
| Seed 0 | 0.5260 (52.60%) | 0.5148 (51.48%) | +0.0112 (+1.12%) |
| Seed 1 | 0.4649 (46.49%) | 0.4664 (46.64%) | -0.0015 (-0.15%) |
| Seed 2 | 0.5377 (53.77%) | 0.5349 (53.49%) | +0.0028 (+0.28%) |
| Seed 3 | 0.5426 (54.26%) | 0.5412 (54.12%) | +0.0014 (+0.14%) |
| Seed 4 | 0.4531 (45.31%) | 0.4589 (45.89%) | -0.0058 (-0.58%) |
| Seed 7 | 0.5324 (53.24%) | 0.5324 (53.24%) | +0.0000 (+0.00%) |
| Seed 10 | 0.5064 (50.64%) | 0.5120 (51.20%) | -0.0056 (-0.56%) |
| Seed 18 | 0.4899 (48.99%) | 0.4899 (48.99%) | +0.0000 (+0.00%) |
| Seed 42 | 0.5134 (51.34%) | 0.5078 (50.78%) | +0.0056 (+0.56%) |
| Seed 123 | 0.4943 (49.43%) | 0.5043 (50.43%) | -0.0100 (-1.00%) |

### Adversarial Canonical Workload (A-adv-10)
* **Best Fit Fail Rate**: 0.0630 ± 0.0504 (6.30%)
* **FluxAlloc Fail Rate**: 0.0588 ± 0.0459 (5.88%)
* **Paired Reduction $\Delta$ (BF − FA)**: +0.0042 ± 0.0058 (+0.42%)
* **Two-sided Paired Wilcoxon $p$-value**: 9.3750e-02 (not significant)
* **Bootstrap 95% Percentile CI of Mean $\Delta$**: [+0.0010, +0.0077] ([+0.10%, +0.77%])

| Seed | Best Fit Fail | FluxAlloc Fail | Paired $\Delta$ (BF − FA) |
| :--- | :---: | :---: | :---: |
| Seed 0 | 0.1264 (12.64%) | 0.1132 (11.32%) | +0.0132 (+1.32%) |
| Seed 1 | 0.0352 (3.52%) | 0.0352 (3.52%) | +0.0000 (+0.00%) |
| Seed 2 | 0.0401 (4.01%) | 0.0401 (4.01%) | +0.0000 (+0.00%) |
| Seed 3 | 0.1679 (16.79%) | 0.1570 (15.70%) | +0.0109 (+1.09%) |
| Seed 4 | 0.0318 (3.18%) | 0.0338 (3.38%) | -0.0020 (-0.20%) |
| Seed 7 | 0.0579 (5.79%) | 0.0521 (5.21%) | +0.0058 (+0.58%) |
| Seed 10 | 0.0000 (0.00%) | 0.0000 (0.00%) | +0.0000 (+0.00%) |
| Seed 18 | 0.0663 (6.63%) | 0.0546 (5.46%) | +0.0117 (+1.17%) |
| Seed 42 | 0.0263 (2.63%) | 0.0263 (2.63%) | +0.0000 (+0.00%) |
| Seed 123 | 0.0777 (7.77%) | 0.0758 (7.58%) | +0.0019 (+0.19%) |

### Randomized Adversarial Workload (C-rand)
* **Best Fit Fail Rate**: 0.9195 ± 0.0307 (91.95%)
* **FluxAlloc Fail Rate**: 0.9195 ± 0.0301 (91.95%)
* **Paired Reduction $\Delta$ (BF − FA)**: +0.0000 ± 0.0026 (+0.00%)
* **Two-sided Paired Wilcoxon $p$-value**: 8.7500e-01 (not significant)
* **Bootstrap 95% Percentile CI of Mean $\Delta$**: [-0.0014, +0.0017] ([-0.14%, +0.17%])

| Seed | Best Fit Fail | FluxAlloc Fail | Paired $\Delta$ (BF − FA) |
| :--- | :---: | :---: | :---: |
| Seed 0 | 0.8723 (87.23%) | 0.8723 (87.23%) | +0.0000 (+0.00%) |
| Seed 1 | 0.9488 (94.88%) | 0.9426 (94.26%) | +0.0062 (+0.62%) |
| Seed 2 | 0.9161 (91.61%) | 0.9161 (91.61%) | +0.0000 (+0.00%) |
| Seed 3 | 0.9466 (94.66%) | 0.9476 (94.76%) | -0.0010 (-0.10%) |
| Seed 4 | 0.9023 (90.23%) | 0.9023 (90.23%) | +0.0000 (+0.00%) |
| Seed 7 | 0.9421 (94.21%) | 0.9411 (94.11%) | +0.0010 (+0.10%) |
| Seed 10 | 0.9349 (93.49%) | 0.9369 (93.69%) | -0.0020 (-0.20%) |
| Seed 18 | 0.8739 (87.39%) | 0.8739 (87.39%) | +0.0000 (+0.00%) |
| Seed 42 | 0.9031 (90.31%) | 0.9073 (90.73%) | -0.0042 (-0.42%) |
| Seed 123 | 0.9552 (95.52%) | 0.9552 (95.52%) | +0.0000 (+0.00%) |

---
## 2. $\beta$-Family Scoring on Bimodal Workload ($Score_\beta = Util - \beta \cdot Frag$)
Evaluated across the 10 Bimodal seeds:

* **$\beta = 0.5$**: FluxAlloc > Best Fit on **1 / 10 seeds** (FluxAlloc mean: +0.5412, Best Fit mean: +0.6017)
* **$\beta = 1.0$**: FluxAlloc > Best Fit on **1 / 10 seeds** (FluxAlloc mean: +0.1329, Best Fit mean: +0.2330)
* **$\beta = 2.0$**: FluxAlloc > Best Fit on **1 / 10 seeds** (FluxAlloc mean: -0.6838, Best Fit mean: -0.5043)

---
## 3. Lookahead Shield Fragmentation Ratios ($Frag_{\text{BF}} / Frag_{\text{FA}}$)
* **Canonical Adversarial Shield Ratio (Seed 42)**: 2.1249x (Best Fit Frag: 0.5667, FluxAlloc Frag: 0.2667)
* **Randomized Adversarial Shield Ratio (C-rand, 10 seeds mean)**: 0.9788x (mean per-seed ratio)
* **Mismatched Oracle Shield Ratio (C-mism vs BF, 10 seeds mean)**: 0.9951x (mean per-seed ratio)

---
## 4. Scale & Headroom Sensitivity Analysis (5 Seeds: {0, 7, 10, 18, 42})
* **B-x64 (Heap=65536, Sizes x64, Bimodal)**: Mean $\Delta\text{fail}$ = +0.0218 (+2.18%), Wilcoxon $p$ = 6.2500e-02 (n=5)
* **B-h2 (Heap=2048, 2x Headroom, Bimodal)**: Mean $\Delta\text{fail}$ = +0.0202 (+2.02%), Wilcoxon $p$ = 6.2500e-02 (n=5)
* **B-h4 (Heap=4096, 4x Headroom, Bimodal)**: Mean $\Delta\text{fail}$ = +0.0079 (+0.79%), Wilcoxon $p$ = 1.2500e-01 (n=5)
* **B-unif-h4 (Heap=4096, 4x Headroom, Uniform)**: Mean $\Delta\text{fail}$ = -0.0037 (-0.37%), Wilcoxon $p$ = 2.5000e-01 (n=5)

---
## 5. Rollout Free-Policy Sensitivity (Bimodal & Adversarial Seed 42)
* **Bimodal s42 Scores**: FIFO = +0.2266, LIFO = +0.0900, random = +0.1325
