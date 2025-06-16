from collections import deque

from dark.config import Config
from dark.engine.block_manager import BlockManager
from dark.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    Manages the scheduling of sequences for execution.

    The scheduler is responsible for deciding which sequences to run in each
    iteration of the engine. It maintains queues for waiting and running
    sequences and implements a scheduling policy that prioritizes throughput
    and fairness. The core policy is to prioritize filling the KV cache with
    new requests (prefill) before generating tokens for existing requests (decode).
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        # Queues for managing sequence states.
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.num_finished = 0
        self.num_tokens = 0

    def is_finished(self):
        """Checks if there are any pending or running sequences."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Adds a new sequence to the waiting queue."""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Schedules the next batch of sequences to be executed.

        The scheduling logic is two-tiered:
        1.  **Prefill**: It first tries to schedule sequences from the `waiting`
            queue. It continues to add sequences as long as the batch size and
            token limits are not exceeded and the block manager can allocate
            the necessary memory. This is the highest priority to ensure new
            requests are processed quickly.
        2.  **Decode**: If no prefill sequences can be scheduled, it creates a
            batch from the `running` queue for single-token decoding. It may
            preempt (swap out) a running sequence if memory is tight.

        Returns:
            A tuple containing the list of scheduled sequences and a boolean
            indicating if it is a prefill batch (`True`) or a decode batch (`False`).
        """
        # --- Stage 1: Try to schedule prefill sequences ---
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # Check if adding the next sequence would exceed batch limits or if
            # there's not enough memory.
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # --- Stage 2: Schedule decode sequences ---
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # If a running sequence cannot have a new token appended (due to lack of
            # memory), we may need to preempt another sequence to make space.
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # If this is the only running sequence and it can't proceed,
                    # it must be preempted.
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # Re-populate the running queue with the newly scheduled decode batch.
        running = deque(scheduled_seqs)
        running.extend(self.running)
        self.running = running

        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        Preempts a running sequence, moving it back to the waiting queue.

        This involves deallocating its KV cache blocks to free up memory.
        The sequence is added to the front of the waiting queue to give it
        priority in the next scheduling iteration.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        Updates the state of sequences after a model execution step.

        It appends the newly generated token to each sequence and checks if any
        have reached a completion condition (EOS token or max length). Finished
        sequences are removed from the running queue and their resources are
        deallocated.
        """
        self.num_tokens += len(token_ids)
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.num_finished += 1
