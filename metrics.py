class Metrics:

    @staticmethod
    def external_fragmentation(heap):

        total_free = heap.total_free_memory()

        if total_free == 0:
            return 0

        largest = heap.largest_free_block()

        frag = 1 - (largest / total_free)

        return frag


    @staticmethod
    def utilization(heap):

        return heap.utilization()


    @staticmethod
    def allocation_failure_rate(failures, total_requests):

        if total_requests == 0:
            return 0

        return failures / total_requests