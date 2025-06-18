from collections import deque

from dark.config import Config
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
        self.eos = config.eos
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: deque[Sequence] = deque()
        self.num_finished = 0
        self.eos_token_id = config.eos

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
        # Simple scheduling: move all waiting to running
        scheduled_seqs = []
        is_prefill = False
        if self.waiting:
            is_prefill = True
            while self.waiting:
                seq = self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append(seq)
        else:
            scheduled_seqs = list(self.running)

        return scheduled_seqs, is_prefill

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        Updates the state of sequences after a model execution step.

        It appends the newly generated token to each sequence and checks if any
        have reached a completion condition (EOS token or max length). Finished
        sequences are removed from the running queue and their resources are
        deallocated.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if token_id == self.eos_token_id and not seq.ignore_eos:
                seq.status = SequenceStatus.FINISHED
            elif len(seq.completion_token_ids) >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            if seq.is_finished:
                self.running.remove(seq)
                self.finished.append(seq)
                self.num_finished += 1
