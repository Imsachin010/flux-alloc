"""
Ablation: neural vs sim blend, lookahead depth (same seed-42 trace for all runs).

  python -m lookahead.ablation_bench --workload bimodal --requests 1000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=("uniform", "bimodal"), default="bimodal")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--heap-size", type=int, default=1024)
    args = ap.parse_args()

    random.seed(42)
    gen = WorkloadGenerator(args.heap_size)
    workload = (
        gen.uniform_workload(args.requests)
        if args.workload == "uniform"
        else gen.bimodal_workload(args.requests)
    )
    hsz = args.heap_size
    mp = _ROOT / "lookahead" / "lookahead_ranker.pt"

    blends = [
        ("neural=1, sim=0  (no rollout score)", 1.0, 0.0),
        ("neural=0, sim=1  (rollout only)", 0.0, 1.0),
        ("default 0.2/0.8", 0.2, 0.8),
    ]
    depths = [4, 8, 12, 16]

    print(
        f"\n=== ABLATION  {args.workload}  n={args.requests}  heap={hsz}  model={mp.name} ===\n"
    )

    print("--- Blend (lookahead_steps=12) ---")
    print(f"{'Label':<34} | {'Util':>6} | {'Frag':>6} | {'Fail':>6} | {'Sc':>7} | {'Lg.f':>6}")
    for label, nw, sw in blends:
        u, f, fl, sc, lg = run_lookahead_bench(
            hsz,
            workload,
            model_path=mp,
            lookahead_steps=12,
            neural_weight=nw,
            sim_weight=sw,
        )
        print(f"{label:<34} | {u:>6.4f} | {f:>6.4f} | {fl:>6.4f} | {sc:>+7.4f} | {lg:>6.4f}")

    print("\n--- Lookahead depth (0.2/0.8 blend) ---")
    for d in depths:
        u, f, fl, sc, lg = run_lookahead_bench(
            hsz,
            workload,
            model_path=mp,
            lookahead_steps=d,
        )
        print(
            f"  depth {d:>2}  | Util {u:.4f}  Frag {f:.4f}  Fail {fl:.4f}  Score {sc:+.4f}  Lg {lg:.4f}"
        )
    print()


if __name__ == "__main__":
    main()
