import itertools


class Block:
    def __init__(self, start, size, allocated=False, block_id=None):
        self.start = start
        self.size = size
        self.allocated = allocated
        self.block_id = block_id

    def __repr__(self):
        status = "ALLOC" if self.allocated else "FREE"
        return f"[{self.start}-{self.start+self.size} | {status} | id={self.block_id}]"


class Heap:
    def __init__(self, heap_size):
        self.heap_size = heap_size
        self.blocks = [Block(0, heap_size, allocated=False)]
        self.id_counter = itertools.count()
        self.last_allocated_addr = 0

    # ---------------------------------
    # Core allocation method
    # ---------------------------------
    def allocate(self, block_index, request_size):
        block = self.blocks[block_index]

        if block.size < request_size or block.allocated:
            return None

        block_id = next(self.id_counter)

        remaining = block.size - request_size

        block.size = request_size
        block.allocated = True
        block.block_id = block_id
        self.last_allocated_addr = block.start

        if remaining > 0:
            new_block = Block(block.start + request_size, remaining, allocated=False)
            self.blocks.insert(block_index + 1, new_block)

        return block_id

    # ---------------------------------
    # Free block
    # ---------------------------------
    def free(self, block_id):

        for i, block in enumerate(self.blocks):

            if block.block_id == block_id and block.allocated:
                block.allocated = False
                block.block_id = None

                self.coalesce()
                return True

        return False

    # ---------------------------------
    # Merge adjacent free blocks
    # ---------------------------------
    def coalesce(self):

        new_blocks = []

        prev = None

        for block in self.blocks:

            if prev and not prev.allocated and not block.allocated:
                prev.size += block.size
            else:
                new_blocks.append(block)
                prev = block

        self.blocks = new_blocks

    # ---------------------------------
    # Helpers
    # ---------------------------------
    def free_blocks(self):

        return [(i, b) for i, b in enumerate(self.blocks) if not b.allocated]

    def allocated_blocks(self):

        return [b for b in self.blocks if b.allocated]

    def total_free_memory(self):

        return sum(b.size for b in self.blocks if not b.allocated)

    def largest_free_block(self):

        free = [b.size for b in self.blocks if not b.allocated]
        return max(free) if free else 0

    def utilization(self):

        allocated = sum(b.size for b in self.blocks if b.allocated)
        return allocated / self.heap_size

    def num_free_blocks(self):

        return sum(1 for b in self.blocks if not b.allocated)

    def reset(self):

        self.blocks = [Block(0, self.heap_size, allocated=False)]
        self.id_counter = itertools.count()
        self.last_allocated_addr = 0