from core.heap import Heap
from core.allocator_strategies import first_fit, best_fit

heap = Heap(128)

a = first_fit(heap, 32)
b = first_fit(heap, 16)

heap.free(a)

c = best_fit(heap, 8)

print(heap.blocks)