from copy import copy
from enum import Enum
import time

from dark.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    Represents the lifecycle status of a generation sequence.
    """

    WAITING = 1
    RUNNING = 2
    FINISHED = 3


class Sequence:
    """
    Represents a single sequence in the generation process.

    Each sequence maintains its own state, including the token IDs, sampling
    parameters, and completion status. It also keeps track of performance
    metrics like latency and time to first token.
    """

    next_seq_id: int = 0
    block_size: int = 16

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = Sequence.next_seq_id
        Sequence.next_seq_id += 1
        self.token_ids = token_ids
        self.prompt_len = len(token_ids)

        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params

        self.arrival_time = time.time()
        self.time_to_first_token = 0.0

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_cached_tokens(self) -> int:
        return 0  # Simplified, no caching for now

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def ignore_eos(self) -> bool:
        return self.sampling_params.ignore_eos

    @property
    def temperature(self) -> float:
        return self.sampling_params.temperature

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    def append_token(self, token_id: int):
        if self.num_completion_tokens == 0:
            self.time_to_first_token = time.time() - self.arrival_time
        self.token_ids.append(token_id)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, i):
        return self.token_ids[i]
