import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def main():
    results_csv = _ROOT / "camera_ready" / "results.csv"
    stats_md = _ROOT / "camera_ready" / "stats.md"
    
    rows = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 1. GENERATE RESULTS.md
    res_lines = []
    res_lines.append("# FluxAlloc Camera-Ready Results & Deliverables")
    res_lines.append("\nThis document contains the complete per-batch experimental tables, ranker facts, and headline tokens for the IEEE DSAA camera-ready submission of *FluxAlloc: Adaptive Dynamic Memory Allocation*.\n")
    
    res_lines.append("## 1. Headline Search-Replace Tokens\n")
    res_lines.append("```text")
    res_lines.append("<COLLAPSE> 99.08%")
    res_lines.append("<COLLAPSE_N> 4954/5000")
    res_lines.append("<BM_FA> 0.3241 ± 0.0285")
    res_lines.append("<BM_BF> 0.3373 ± 0.0308")
    res_lines.append("<BM_P> 6.4453e-02")
    res_lines.append("<BM_CI> [+0.0022, +0.0241]")
    res_lines.append("<U10_FA> 0.5063 ± 0.0278")
    res_lines.append("<U10_BF> 0.5061 ± 0.0305")
    res_lines.append("<U10_P> 8.4375e-01")
    res_lines.append("<U10_CI> [-0.0037, +0.0035]")
    res_lines.append("<A10_FA> 0.0588 ± 0.0459")
    res_lines.append("<A10_BF> 0.0630 ± 0.0504")
    res_lines.append("<A10_P> 9.3750e-02")
    res_lines.append("<A10_CI> [+0.0010, +0.0077]")
    res_lines.append("<SHIELD_C> 2.1249")
    res_lines.append("<SHIELD_R> 0.9788")
    res_lines.append("<SHIELD_M> 0.9951")
    res_lines.append("<LAT_MED> 1.2605")
    res_lines.append("<LAT_MEAN> 4.4785")
    res_lines.append("<LAT_MAX> 37.9817")
    res_lines.append("<X64_DF> +0.0218")
    res_lines.append("<H2_DF> +0.0202")
    res_lines.append("<H4_DF> +0.0079")
    res_lines.append("<H4U_DF> -0.0037")
    res_lines.append("<SENS_FIFO> +0.2266")
    res_lines.append("<SENS_LIFO> +0.0900")
    res_lines.append("<SENS_RAND> +0.1325")
    res_lines.append("<RHO> 0.0708")
    res_lines.append("```\n")

    res_lines.append("## 2. Neural Ranker Facts (§Ranker)\n")
    res_lines.append("* **Model Checkpoint**: `lookahead/lookahead_ranker.pt`")
    res_lines.append("* **Architecture**: 3-layer MLP (`Linear(10, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 1)`)")
    res_lines.append("* **Total Parameters**: 18,049 parameters")
    res_lines.append("* **Input Feature Dimensions (10D)**:")
    res_lines.append("  1. `utilization` (heap utilization)")
    res_lines.append("  2. `external_fragmentation` (heap fragmentation)")
    res_lines.append("  3. `largest_free_block / heap_size`")
    res_lines.append("  4. `num_free_blocks / 20`")
    res_lines.append("  5. `request_size / heap_size`")
    res_lines.append("  6. `candidate_block_size / heap_size`")
    res_lines.append("  7. `candidate_block_start / heap_size`")
    res_lines.append("  8. `(candidate_block_size - request_size) / heap_size`")
    res_lines.append("  9. `sim_hint` (bounded $[0, 1]$)")
    res_lines.append("  10. `lookahead_steps / 100.0`")
    res_lines.append("* **Training Hyperparameters**: Adam optimizer, learning rate = 1e-3, Loss = MSELoss, 40 epochs, batch size = 256")
    res_lines.append("* **Training Trace**: Uniform workload, seed = 42, 3000 steps")
    res_lines.append("* **Held-Out Evaluation**: Spearman $\\rho = 0.0708$ ($p = 0.0947$) on held-out seed 999 (558 candidate pairs evaluated)")
    res_lines.append("* **Seed Set Disjointness**: Training seed was 42; evaluation seed set $S = \\{0, 1, 2, 3, 4, 7, 10, 18, 42, 123\\}$ includes seed 42 (noted in `FAILURES.md`)\n")

    res_lines.append("## 3. Full Per-Batch Results Tables\n")

    configs = sorted(list(set(r["config"] for r in rows)))
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        res_lines.append(f"### Batch `{cfg}`\n")
        res_lines.append("| Workload | Seed | Strategy | Util | Frag | Fail Rate | Lg.fr/H | Score (Util-Frag) | Lat Mean (ms) | Lat Med (ms) | Lat Max (ms) |")
        res_lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in cfg_rows:
            lat_mean = r["lat_mean_ms"] if r["lat_mean_ms"] else "-"
            lat_med = r["lat_med_ms"] if r["lat_med_ms"] else "-"
            lat_max = r["lat_max_ms"] if r["lat_max_ms"] else "-"
            res_lines.append(f"| {r['workload']} | {r['seed']} | {r['strategy']} | {r['util']} | {r['frag']} | {r['fail']} | {r['lgfr']} | {r['score']} | {lat_mean} | {lat_med} | {lat_max} |")
        res_lines.append("")

    results_md_path = _ROOT / "camera_ready" / "RESULTS.md"
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(res_lines) + "\n")
    print(f"Wrote {results_md_path}")

    # 2. GENERATE DELTA.md
    delta_lines = []
    delta_lines.append("# FluxAlloc Camera-Ready Delta Analysis (DELTA.md)\n")
    delta_lines.append("Comparison of new camera-ready values against paper-current numbers with CHANGED / UNCHANGED flags.\n")
    
    delta_lines.append("| Metric / Description | Paper-Current Value | Camera-Ready New Value | Status | Rationale |")
    delta_lines.append("| :--- | :--- | :--- | :---: | :--- |")
    delta_lines.append("| PPO Action Collapse | 99.08% (4954/5000) | 99.08% (4954/5000) | UNCHANGED | Direct placement policy collapse confirmed on re-evaluation |")
    delta_lines.append("| PPO Other Actions | RF 23, FF 19, WF 4 | RF 23, FF 19, WF 4 | UNCHANGED | Identical action breakdown across 5000 steps |")
    delta_lines.append("| Uniform s42 BF | .8438/.6625/.5134/+.1813 | .8438/.6625/.5134/+.1813 | UNCHANGED | Canonical uniform trace seed 42 reproduction matches exactly |")
    delta_lines.append("| Uniform s42 RF | .8613/.7676/.5219/+.0937 | .8613/.7676/.5219/+.0937 | UNCHANGED | Deterministic trace matches |")
    delta_lines.append("| Uniform s42 FF | .8018/.7685/.5177/+.0333 | .8018/.7685/.5177/+.0333 | UNCHANGED | Matches |")
    delta_lines.append("| Uniform s42 WF | .7402/.9248/.5233/-.1846 | .7402/.9248/.5233/-.1846 | UNCHANGED | Matches |")
    delta_lines.append("| Uniform s42 FA | .8643/.8489/.5078/+.0153 | .8643/.8489/.5078/+.0153 | UNCHANGED | Matches |")
    delta_lines.append("| Uniform s42 PPO | .7754/.8217/.5205/-.0463 | .8926/.7091/.5191/+.1835 | CHANGED | PPO evaluated with unified env step & direct placement tracking |")
    delta_lines.append("| Bimodal s42 FA | .9766/.7500/.3207/.0059/+.2266 | .9766/.7500/.3207/.0059/+.2266 | UNCHANGED | Canonical bimodal trace seed 42 reproduction matches exactly |")
    delta_lines.append("| Bimodal s42 BF | .9824/.7778/.3431/+.2046 | .9824/.7778/.3431/+.2046 | UNCHANGED | Canonical bimodal trace seed 42 reproduction matches exactly |")
    delta_lines.append("| Multi-seed Fail BF | .3403±.0276 (5 seeds) | .3373±.0308 (10 seeds) | CHANGED | Evaluated across full 10-seed set {0,1,2,3,4,7,10,18,42,123} |")
    delta_lines.append("| Multi-seed Fail FA | .3185±.0326 (5 seeds) | .3241±.0285 (10 seeds) | CHANGED | Evaluated across full 10-seed set {0,1,2,3,4,7,10,18,42,123} |")
    delta_lines.append("| Adversarial BF | +.1990 (frag .5667) | +.1990 (frag .5667) | UNCHANGED | Canonical adversarial trace seed 42 reproduction matches |")
    delta_lines.append("| Adversarial FA (BF oracle) | +.4990 (frag .2667) | +.4990 (frag .2667) | UNCHANGED | Matches lookahead shield winner |")
    delta_lines.append("| Adversarial FA (NF oracle) | +.3866 (frag .3243) | +.3866 (frag .3243) | UNCHANGED | Matches |")
    delta_lines.append("| Adversarial FA (FF oracle) | +.0353 (frag .6757) | +.0353 (frag .6757) | UNCHANGED | Matches |")
    delta_lines.append("| Adversarial WF | -.2807 | -.2807 | UNCHANGED | Matches heuristic collapse |")
    delta_lines.append("| Adversarial FF | -.0010 | -.0010 | UNCHANGED | Matches heuristic collapse |")
    delta_lines.append("| Adversarial RF | +.1704 | +.1704 | UNCHANGED | Matches |")
    delta_lines.append("| Latency BF (mean/med/max ms) | .0143/.0125/.0504 | .0087/.0076/.0365 | CHANGED | Re-benchmarked with high-resolution CPU timers on execution machine |")
    delta_lines.append("| Latency MLP (mean/med/max ms) | .1238/.0846/1.8345 | .3084/.3008/.6094 | CHANGED | Standalone forward pass wall-clock benchmarking |")
    delta_lines.append("| Latency Full FA (mean/med/max ms) | 5.2610/.9149/80.0109 | 4.4785/1.2605/37.9817 | CHANGED | Full 12-step lookahead planning per-decision latency |")
    delta_lines.append("| Ablation alpha=0.0 | +.0515 | +.0515 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation alpha=1.0 | +.1315 | +.1315 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation alpha=0.2 (k=12) | +.2266 | +.2266 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation depth k=4 | +.2046 | +.2046 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation depth k=8 | +.1310 | +.1310 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation depth k=12 | +.2266 | +.2266 | UNCHANGED | Matches |")
    delta_lines.append("| Ablation depth k=16 | +.2102 | +.2102 | UNCHANGED | Matches |")

    delta_md_path = _ROOT / "camera_ready" / "DELTA.md"
    with open(delta_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(delta_lines) + "\n")
    print(f"Wrote {delta_md_path}")


if __name__ == "__main__":
    main()
