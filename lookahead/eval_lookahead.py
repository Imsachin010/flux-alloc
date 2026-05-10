"""
Quick single-strategy printout. For apples-to-apples vs baselines use:
  python -m lookahead.compare_final --workload bimodal --requests 1000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from core.workload_generator import WorkloadGenerator
from lookahead.bench import run_lookahead_bench


def evaluate() -> None:
    ap = argparse.ArgumentParser(
        description="Run Lookahead+Neural only. Defaults match compare_final --workload bimodal."
    )
    ap.add_argument(
        "--workload",
        choices=("uniform", "bimodal"),
        default="bimodal",
        help="Must match the table you compare to (compare_final defaults to uniform).",
    )
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--heap-size", type=int, default=1024)
    ap.add_argument("--lookahead", type=int, default=12)
    args = ap.parse_args()

    random.seed(42)
    gen = WorkloadGenerator(args.heap_size)
    if args.workload == "uniform":
        workload = gen.uniform_workload(args.requests)
    else:
        workload = gen.bimodal_workload(args.requests)

    u, f, fl, sc, lg = run_lookahead_bench(
        args.heap_size,
        workload,
        model_path=Path(__file__).resolve().parent / "lookahead_ranker.pt",
        lookahead_steps=args.lookahead,
    )

    print(
        f"\n=== LOOKAHEAD + NEURAL  ({args.workload}  n={args.requests}  "
        f"heap={args.heap_size}  lookahead={args.lookahead}) ==="
    )
    print("Utilization:     ", u)
    print("Fragmentation:  ", f)
    print("Failure rate:   ", fl)
    print("Score (u-frag):", sc)
    print("Largest free:   ", round(lg, 4), "(/ heap)")
    print(
        "\nFor the same trace against First/Best/.../PPO, run:\n"
        f"  python -m lookahead.compare_final --workload {args.workload} --requests {args.requests}\n"
    )


if __name__ == "__main__":
    evaluate()
