import hashlib
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.workload_generator import WorkloadGenerator

def trace_to_bytes(trace: list) -> bytes:
    # Convert list of tuples like [("malloc", 16), ("free", None)] to canonical json bytes
    return json.dumps(trace, sort_keys=True).encode("utf-8")

def hash_trace(trace: list) -> str:
    return hashlib.sha256(trace_to_bytes(trace)).hexdigest()

def main():
    seeds = [0, 1, 2, 3, 4, 5, 7, 10, 18, 42, 123]
    workload_names = ["uniform", "bimodal", "adversarial"]
    heap_size = 1024
    num_requests = 1000

    manifest_rows = [["workload", "seed", "sha256", "num_requests"]]
    workload_hashes = {w: {} for w in workload_names}

    generator = WorkloadGenerator(heap_size)

    for w_name in workload_names:
        for seed in seeds:
            # Set global seeds as traditionally done before calling generator
            random.seed(seed)
            if hasattr(generator, "seed"):
                generator.seed = seed

            if w_name == "uniform":
                trace = generator.uniform_workload(num_requests, seed=seed) if "seed" in generator.uniform_workload.__code__.co_varnames else generator.uniform_workload(num_requests)
            elif w_name == "bimodal":
                trace = generator.bimodal_workload(num_requests, seed=seed) if "seed" in generator.bimodal_workload.__code__.co_varnames else generator.bimodal_workload(num_requests)
            elif w_name == "adversarial":
                trace = generator.scaled_adversarial_workload(num_requests, seed=seed) if "seed" in generator.scaled_adversarial_workload.__code__.co_varnames else generator.scaled_adversarial_workload(num_requests)

            h = hash_trace(trace)
            manifest_rows.append([w_name, str(seed), h, str(len(trace))])
            if h in workload_hashes[w_name]:
                workload_hashes[w_name][h].append(seed)
            else:
                workload_hashes[w_name][h] = [seed]

    out_csv = Path(__file__).resolve().parent / "trace_manifest.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(",".join(row) + "\n")
    print(f"Wrote trace manifest to {out_csv}")

    duplicates_found = False
    for w_name, hashes in workload_hashes.items():
        print(f"\n--- Checking {w_name} traces ---")
        for h, seed_list in hashes.items():
            if len(seed_list) > 1:
                duplicates_found = True
                print(f"[DUPLICATE FOUND] Workload '{w_name}' produced identical trace for seeds: {seed_list} (hash: {h[:12]}...)")
            else:
                print(f"Seed {seed_list[0]}: {h[:12]}... (unique)")

    if duplicates_found:
        print("\nTrace audit result: DUPLICATES DETECTED! Generator needs seeding fix.")
        sys.exit(1)
    else:
        print("\nTrace audit result: SUCCESS! All seeds produced unique traces.")
        sys.exit(0)

if __name__ == "__main__":
    main()
