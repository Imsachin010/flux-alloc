import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci_mean_diff(diffs: np.ndarray, num_resamples: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(num_resamples)
    for i in range(num_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, (1.0 - ci) / 2.0 * 100.0)
    upper = np.percentile(boot_means, (1.0 + ci) / 2.0 * 100.0)
    return float(lower), float(upper)


def main():
    results_csv = _ROOT / "camera_ready" / "results.csv"
    if not results_csv.is_file():
        print(f"Error: {results_csv} not found.")
        sys.exit(1)

    rows = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Filter helper
    def get_runs(config: str, strategy: str) -> dict[int, dict]:
        res = {}
        for r in rows:
            if r["config"] == config and r["strategy"] == strategy:
                s = int(r["seed"])
                res[s] = {
                    "util": float(r["util"]),
                    "frag": float(r["frag"]),
                    "fail": float(r["fail"]),
                    "lgfr": float(r["lgfr"]),
                    "score": float(r["score"]),
                }
        return res

    stats_lines = []
    stats_lines.append("# FluxAlloc Statistical Significance & Analysis (Phase 5)")
    stats_lines.append("\nAll statistical comparisons evaluate paired runs per seed using two-sided paired Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`) and 10,000-resample bootstrap 95% percentile confidence intervals for $\\Delta = \\text{Fail}_{\\text{BF}} - \\text{Fail}_{\\text{FA}}$.\n")

    # 1. 10-SEED COMPARISONS
    stats_lines.append("## 1. Multi-Seed Failure Rate Comparisons (10 Seeds: {0, 1, 2, 3, 4, 7, 10, 18, 42, 123})")
    
    comparisons = [
        ("Bimodal Workload (A-bim-10)", "A-bim-10", "Best Fit", "FluxAlloc"),
        ("Uniform Workload (A-unif-10)", "A-unif-10", "Best Fit", "FluxAlloc"),
        ("Adversarial Canonical Workload (A-adv-10)", "A-adv-10", "Best Fit", "FluxAlloc"),
        ("Randomized Adversarial Workload (C-rand)", "C-rand", "Best Fit", "FluxAlloc (matched)"),
    ]

    headline_stats = {}

    for title, config, strat_bf, strat_fa in comparisons:
        bf_runs = get_runs(config, strat_bf)
        fa_runs = get_runs(config, strat_fa)
        seeds = sorted(list(set(bf_runs.keys()) & set(fa_runs.keys())))

        bf_fails = np.array([bf_runs[s]["fail"] for s in seeds])
        fa_fails = np.array([fa_runs[s]["fail"] for s in seeds])
        deltas = bf_fails - fa_fails # Positive delta means FA had lower failure rate than BF

        bf_mean, bf_sd = np.mean(bf_fails), np.std(bf_fails, ddof=1)
        fa_mean, fa_sd = np.mean(fa_fails), np.std(fa_fails, ddof=1)
        delta_mean, delta_sd = np.mean(deltas), np.std(deltas, ddof=1)

        # Wilcoxon test (handle potential identical values)
        try:
            w_res = wilcoxon(bf_fails, fa_fails, alternative='two-sided')
            p_val = w_res.pvalue
        except Exception as e:
            p_val = 1.0

        ci_low, ci_high = bootstrap_ci_mean_diff(deltas, num_resamples=10000, ci=0.95, seed=42)

        headline_stats[config] = {
            "bf_mean": bf_mean, "bf_sd": bf_sd,
            "fa_mean": fa_mean, "fa_sd": fa_sd,
            "delta_mean": delta_mean, "delta_sd": delta_sd,
            "p_val": p_val,
            "ci_low": ci_low, "ci_high": ci_high,
            "seeds": seeds, "bf_fails": bf_fails, "fa_fails": fa_fails, "deltas": deltas
        }

        stats_lines.append(f"\n### {title}")
        stats_lines.append(f"* **Best Fit Fail Rate**: {bf_mean:.4f} ± {bf_sd:.4f} ({bf_mean*100:.2f}%)")
        stats_lines.append(f"* **FluxAlloc Fail Rate**: {fa_mean:.4f} ± {fa_sd:.4f} ({fa_mean*100:.2f}%)")
        stats_lines.append(f"* **Paired Reduction $\\Delta$ (BF − FA)**: {delta_mean:+.4f} ± {delta_sd:.4f} ({delta_mean*100:+.2f}%)")
        stats_lines.append(f"* **Two-sided Paired Wilcoxon $p$-value**: {p_val:.4e} ({'p < 0.05' if p_val < 0.05 else 'not significant'})")
        stats_lines.append(f"* **Bootstrap 95% Percentile CI of Mean $\\Delta$**: [{ci_low:+.4f}, {ci_high:+.4f}] ([{ci_low*100:+.2f}%, {ci_high*100:+.2f}%])")
        stats_lines.append("\n| Seed | Best Fit Fail | FluxAlloc Fail | Paired $\\Delta$ (BF − FA) |")
        stats_lines.append("| :--- | :---: | :---: | :---: |")
        for s, bf_f, fa_f, d in zip(seeds, bf_fails, fa_fails, deltas):
            stats_lines.append(f"| Seed {s} | {bf_f:.4f} ({bf_f*100:.2f}%) | {fa_f:.4f} ({fa_f*100:.2f}%) | {d:+.4f} ({d*100:+.2f}%) |")

    # 2. BETA-FAMILY SCORING ON BIMODAL-10
    stats_lines.append("\n---")
    stats_lines.append("## 2. $\\beta$-Family Scoring on Bimodal Workload ($Score_\\beta = Util - \\beta \\cdot Frag$)")
    stats_lines.append("Evaluated across the 10 Bimodal seeds:\n")

    bim_bf = get_runs("A-bim-10", "Best Fit")
    bim_fa = get_runs("A-bim-10", "FluxAlloc")
    bim_seeds = sorted(list(set(bim_bf.keys()) & set(bim_fa.keys())))

    for beta in [0.5, 1.0, 2.0]:
        fa_wins = 0
        bf_scores_beta = []
        fa_scores_beta = []
        for s in bim_seeds:
            sc_bf = bim_bf[s]["util"] - beta * bim_bf[s]["frag"]
            sc_fa = bim_fa[s]["util"] - beta * bim_fa[s]["frag"]
            bf_scores_beta.append(sc_bf)
            fa_scores_beta.append(sc_fa)
            if sc_fa > sc_bf:
                fa_wins += 1
        stats_lines.append(f"* **$\\beta = {beta}$**: FluxAlloc > Best Fit on **{fa_wins} / {len(bim_seeds)} seeds** (FluxAlloc mean: {np.mean(fa_scores_beta):+.4f}, Best Fit mean: {np.mean(bf_scores_beta):+.4f})")

    # 3. LOOKAHEAD SHIELD RATIOS
    stats_lines.append("\n---")
    stats_lines.append("## 3. Lookahead Shield Fragmentation Ratios ($Frag_{\\text{BF}} / Frag_{\\text{FA}}$)")
    
    # Canonical (seed 42)
    adv_can_bf = get_runs("A-adv-canonical", "Best Fit")
    adv_can_fa = get_runs("A-adv-canonical", "FluxAlloc (oracle=best_fit)")
    shield_c = adv_can_bf[42]["frag"] / adv_can_fa[42]["frag"] if adv_can_fa[42]["frag"] > 0 else 1.0
    stats_lines.append(f"* **Canonical Adversarial Shield Ratio (Seed 42)**: {shield_c:.4f}x (Best Fit Frag: {adv_can_bf[42]['frag']:.4f}, FluxAlloc Frag: {adv_can_fa[42]['frag']:.4f})")

    # Randomized (C-rand)
    crand_bf = get_runs("C-rand", "Best Fit")
    crand_fa = get_runs("C-rand", "FluxAlloc (matched)")
    crand_seeds = sorted(list(set(crand_bf.keys()) & set(crand_fa.keys())))
    ratios_rand = [crand_bf[s]["frag"] / crand_fa[s]["frag"] if crand_fa[s]["frag"] > 0 else 1.0 for s in crand_seeds]
    shield_r = float(np.mean(ratios_rand))
    stats_lines.append(f"* **Randomized Adversarial Shield Ratio (C-rand, 10 seeds mean)**: {shield_r:.4f}x (mean per-seed ratio)")

    # Mismatched (C-mism)
    cmism_fa = get_runs("C-mism", "FluxAlloc (mismatched)")
    ratios_mism = [crand_bf[s]["frag"] / cmism_fa[s]["frag"] if cmism_fa[s]["frag"] > 0 else 1.0 for s in crand_seeds]
    shield_m = float(np.mean(ratios_mism))
    stats_lines.append(f"* **Mismatched Oracle Shield Ratio (C-mism vs BF, 10 seeds mean)**: {shield_m:.4f}x (mean per-seed ratio)")

    # 4. SCALE & HEADROOM SENSITIVITY
    stats_lines.append("\n---")
    stats_lines.append("## 4. Scale & Headroom Sensitivity Analysis (5 Seeds: {0, 7, 10, 18, 42})")

    scale_configs = [
        ("B-x64 (Heap=65536, Sizes x64, Bimodal)", "B-x64", "bimodal"),
        ("B-h2 (Heap=2048, 2x Headroom, Bimodal)", "B-h2", "bimodal"),
        ("B-h4 (Heap=4096, 4x Headroom, Bimodal)", "B-h4", "bimodal"),
        ("B-unif-h4 (Heap=4096, 4x Headroom, Uniform)", "B-unif-h4", "uniform"),
    ]

    scale_stats = {}
    for title, cfg, wk in scale_configs:
        bf_r = get_runs(cfg, "Best Fit")
        fa_r = get_runs(cfg, "FluxAlloc")
        seeds = sorted(list(set(bf_r.keys()) & set(fa_r.keys())))
        bf_f = np.array([bf_r[s]["fail"] for s in seeds])
        fa_f = np.array([fa_r[s]["fail"] for s in seeds])
        d = bf_f - fa_f
        mean_d = float(np.mean(d))
        try:
            w_res = wilcoxon(bf_f, fa_f, alternative='two-sided')
            p_val = float(w_res.pvalue)
        except Exception:
            p_val = 1.0
        scale_stats[cfg] = {"mean_delta": mean_d, "p_val": p_val}
        stats_lines.append(f"* **{title}**: Mean $\\Delta\\text{{fail}}$ = {mean_d:+.4f} ({mean_d*100:+.2f}%), Wilcoxon $p$ = {p_val:.4e} (n=5)")

    # 5. FREE POLICY SENSITIVITY
    stats_lines.append("\n---")
    stats_lines.append("## 5. Rollout Free-Policy Sensitivity (Bimodal & Adversarial Seed 42)")
    d_free_runs = {r["strategy"]: float(r["score"]) for r in rows if r["config"] == "D-freepol" and r["workload"] == "bimodal"}
    d_free_runs["FluxAlloc (free=FIFO)"] = float(next(r["score"] for r in rows if r["config"] == "A-bim-s42" and r["strategy"] == "FluxAlloc"))
    stats_lines.append(f"* **Bimodal s42 Scores**: FIFO = {d_free_runs.get('FluxAlloc (free=FIFO)', 0.0):+.4f}, LIFO = {d_free_runs.get('FluxAlloc (free=LIFO)', 0.0):+.4f}, random = {d_free_runs.get('FluxAlloc (free=random)', 0.0):+.4f}")

    # Output stats.md
    out_stats = _ROOT / "camera_ready" / "stats.md"
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")
    print(f"Wrote statistical analysis report to {out_stats}")


if __name__ == "__main__":
    main()
