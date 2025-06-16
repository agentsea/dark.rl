from collections import deque

import numpy as np
import xxhash

from dark.engine.sequence import Sequence


def compute_hash(token_ids: list[int], prefix: int = -1):
    """
    Computes a hash for a list of token IDs, optionally with a prefix hash.
    This enables chaining hashes block by block to uniquely identify a sequence prefix.

    Args:
        token_ids: A list of token IDs in a block.
        prefix: The hash of the preceding block in the sequence.

    Returns:
        A 64-bit integer hash value.
    """
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class Block:
    """
    Represents a single block in the KV cache.

    Attributes:
        block_id: The unique identifier for this block.
        ref_count: The number of sequences currently referencing this block.
                   This is key to the copy-on-write mechanism.
        hash: A hash of the block's content, used for prefix sharing.
        token_ids: The token IDs stored in this block, used for hash verification.
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """Updates the hash and token IDs for this block."""
        assert hash != -1
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Resets the block to a clean state for a new allocation."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return f"{(self.block_id, self.ref_count, self.hash)}"


class BlockManager:
    """
    Manages the allocation and deallocation of KV cache blocks.

    This manager implements a paged memory strategy with prefix caching and
    copy-on-write. It allows different sequences to share memory blocks if they
    have a common prefix, significantly saving memory.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # A list of all physical blocks.
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Maps a content hash to a block ID for fast prefix lookup.
        self.hash_to_block_id: dict[int, int] = dict()
        # A queue of available (free) block IDs.
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # A set of currently used block IDs.
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int):
        """Internal helper to mark a block as used and reset its state."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """Internal helper to mark a block as free."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        """Checks if there are enough free blocks to allocate for a sequence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        Allocates blocks for a sequence, attempting to use the prefix cache.

        For each block in the sequence, it computes a hash. If the hash exists
        in the cache, it reuses the existing block by incrementing its reference
        count (a cache hit). If not, it allocates a new block (a cache miss).
        This copy-on-write strategy is triggered as soon as a cache miss occurs.
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # A block's hash depends on its content and the previous block's hash.
            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # A cache miss occurs if the hash is not found or if the content differs
            # (due to a hash collision).
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # On a miss, allocate a new block.
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # On a hit, reuse the existing block.
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # If the block is already in use, just increment its ref count.
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Otherwise, allocate it from the free list.
                    block = self._allocate_block(block_id)

            if h != -1:
                # Update the block's content and the hash-to-block mapping.
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        Deallocates all blocks for a finished or swapped sequence.

        It iterates through the sequence's block table, decrementing the reference
        count for each block. If a block's reference count drops to zero, it is
        truly freed and returned to the pool of available blocks.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        """Checks if a new block can be appended to a sequence for decoding."""
        # A new block is needed only when the last block becomes full.
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        Handles block state updates during decoding (appending one token at a time).

        When a block is filled by the newly generated token, its hash is computed
        and stored. When a new block is needed, it is allocated from the free list.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # If the last token starts a new block, allocate it.
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # If the last token just filled a block, compute and store its hash.
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.last_block()
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # The last token is in a partially filled block.
            assert last_block.hash == -1
