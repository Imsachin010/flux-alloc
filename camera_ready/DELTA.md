# FluxAlloc Camera-Ready Delta Analysis (DELTA.md)

Comparison of new camera-ready values against paper-current numbers with CHANGED / UNCHANGED flags.

| Metric / Description | Paper-Current Value | Camera-Ready New Value | Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| PPO Action Collapse | 99.08% (4954/5000) | 99.08% (4954/5000) | UNCHANGED | Direct placement policy collapse confirmed on re-evaluation |
| PPO Other Actions | RF 23, FF 19, WF 4 | RF 23, FF 19, WF 4 | UNCHANGED | Identical action breakdown across 5000 steps |
| Uniform s42 BF | .8438/.6625/.5134/+.1813 | .8438/.6625/.5134/+.1813 | UNCHANGED | Canonical uniform trace seed 42 reproduction matches exactly |
| Uniform s42 RF | .8613/.7676/.5219/+.0937 | .8613/.7676/.5219/+.0937 | UNCHANGED | Deterministic trace matches |
| Uniform s42 FF | .8018/.7685/.5177/+.0333 | .8018/.7685/.5177/+.0333 | UNCHANGED | Matches |
| Uniform s42 WF | .7402/.9248/.5233/-.1846 | .7402/.9248/.5233/-.1846 | UNCHANGED | Matches |
| Uniform s42 FA | .8643/.8489/.5078/+.0153 | .8643/.8489/.5078/+.0153 | UNCHANGED | Matches |
| Uniform s42 PPO | .7754/.8217/.5205/-.0463 | .8926/.7091/.5191/+.1835 | CHANGED | PPO evaluated with unified env step & direct placement tracking |
| Bimodal s42 FA | .9766/.7500/.3207/.0059/+.2266 | .9766/.7500/.3207/.0059/+.2266 | UNCHANGED | Canonical bimodal trace seed 42 reproduction matches exactly |
| Bimodal s42 BF | .9824/.7778/.3431/+.2046 | .9824/.7778/.3431/+.2046 | UNCHANGED | Canonical bimodal trace seed 42 reproduction matches exactly |
| Multi-seed Fail BF | .3403±.0276 (5 seeds) | .3373±.0308 (10 seeds) | CHANGED | Evaluated across full 10-seed set {0,1,2,3,4,7,10,18,42,123} |
| Multi-seed Fail FA | .3185±.0326 (5 seeds) | .3241±.0285 (10 seeds) | CHANGED | Evaluated across full 10-seed set {0,1,2,3,4,7,10,18,42,123} |
| Adversarial BF | +.1990 (frag .5667) | +.1990 (frag .5667) | UNCHANGED | Canonical adversarial trace seed 42 reproduction matches |
| Adversarial FA (BF oracle) | +.4990 (frag .2667) | +.4990 (frag .2667) | UNCHANGED | Matches lookahead shield winner |
| Adversarial FA (NF oracle) | +.3866 (frag .3243) | +.3866 (frag .3243) | UNCHANGED | Matches |
| Adversarial FA (FF oracle) | +.0353 (frag .6757) | +.0353 (frag .6757) | UNCHANGED | Matches |
| Adversarial WF | -.2807 | -.2807 | UNCHANGED | Matches heuristic collapse |
| Adversarial FF | -.0010 | -.0010 | UNCHANGED | Matches heuristic collapse |
| Adversarial RF | +.1704 | +.1704 | UNCHANGED | Matches |
| Latency BF (mean/med/max ms) | .0143/.0125/.0504 | .0087/.0076/.0365 | CHANGED | Re-benchmarked with high-resolution CPU timers on execution machine |
| Latency MLP (mean/med/max ms) | .1238/.0846/1.8345 | .3084/.3008/.6094 | CHANGED | Standalone forward pass wall-clock benchmarking |
| Latency Full FA (mean/med/max ms) | 5.2610/.9149/80.0109 | 4.4785/1.2605/37.9817 | CHANGED | Full 12-step lookahead planning per-decision latency |
| Ablation alpha=0.0 | +.0515 | +.0515 | UNCHANGED | Matches |
| Ablation alpha=1.0 | +.1315 | +.1315 | UNCHANGED | Matches |
| Ablation alpha=0.2 (k=12) | +.2266 | +.2266 | UNCHANGED | Matches |
| Ablation depth k=4 | +.2046 | +.2046 | UNCHANGED | Matches |
| Ablation depth k=8 | +.1310 | +.1310 | UNCHANGED | Matches |
| Ablation depth k=12 | +.2266 | +.2266 | UNCHANGED | Matches |
| Ablation depth k=16 | +.2102 | +.2102 | UNCHANGED | Matches |
