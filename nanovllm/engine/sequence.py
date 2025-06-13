from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    Represents the lifecycle status of a generation sequence.
    """

    WAITING = auto()  # The sequence is in the waiting queue, not yet running.
    RUNNING = auto()  # The sequence is currently being processed by the engine.
    FINISHED = auto()  # The sequence has completed generation.


class Sequence:
    """
    Represents a single generation request and its state.

    This class encapsulates all the information related to a sequence, including
    the input tokens (prompt), the generated tokens (completion), its current
    status, its sampling parameters, and its mapping to the physical KV cache
    blocks (block_table).
    """

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        # A unique identifier for the sequence.
        self.seq_id = next(Sequence.counter)
        # The current status of the sequence.
        self.status = SequenceStatus.WAITING
        # The full list of token IDs (prompt + completion).
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        # The number of tokens whose KV pairs are already in the cache (prefix caching).
        self._num_cached_tokens = 0
        # A list of physical block IDs allocated to this sequence.
        self.block_table = []

        # --- Sampling Parameters ---
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """Returns the total number of tokens in the sequence."""
        return len(self.token_ids)

    def __lt__(self, other):
        """Used for sorting sequences by their ID."""
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        """Allows accessing token IDs by index."""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """Checks if the sequence has finished generation."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """Returns the number of tokens that have been generated (the completion)."""
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """Returns the list of token IDs in the prompt."""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """Returns the list of token IDs in the generated completion."""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_tokens(self):
        """
        Returns the number of tokens at the beginning of the sequence whose
        KV pairs are already present in the cache due to prefix sharing.
        """
        return self._num_cached_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, num_cached_tokens):
        assert num_cached_tokens % self.block_size == 0
        self._num_cached_tokens = num_cached_tokens

    @property
    def num_cached_blocks(self):
        """Returns the number of blocks covered by the cached tokens."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """Returns the total number of physical blocks required for the sequence."""
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_token(self):
        """Returns the most recently generated token ID."""
        return self.token_ids[-1]

    def block(self, i):
        """Returns the token IDs for a specific block in the sequence."""
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def last_block(self):
        """Returns the token IDs for the last (potentially partially filled) block."""
        n = self.num_blocks
        return self.token_ids[(n - 1) * self.block_size :]

    def append_token(self, token_id: int):
        """Appends a newly generated token to the sequence."""
        self.token_ids.append(token_id)
