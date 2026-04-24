import random


class WorkloadGenerator:

    def __init__(self, heap_size):
        self.heap_size = heap_size


    def uniform_workload(self, num_requests=1000):

        requests = []

        for _ in range(num_requests):

            # 70% malloc, 30% free
            if random.random() < 0.7:
                size = random.randint(1, 64)
                requests.append(("malloc", size))
            else:
                requests.append(("free", None))

        return requests


    def bimodal_workload(self, num_requests=1000):

        requests = []

        for _ in range(num_requests):

            if random.random() < 0.7:

                if random.random() < 0.7:
                    size = random.randint(1, 8)
                else:
                    size = random.randint(32, 64)

                requests.append(("malloc", size))

            else:
                requests.append(("free", None))

        return requests


    def adversarial_workload(self):

        return [
            ("malloc", 8),
            ("malloc", 8),
            ("malloc", 8),
            ("free", None),
            ("malloc", 4),
            ("malloc", 4),
        ]