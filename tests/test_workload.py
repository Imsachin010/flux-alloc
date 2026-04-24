from core.workload_generator import WorkloadGenerator

gen = WorkloadGenerator(128)

w = gen.uniform_workload(10)

print(w)