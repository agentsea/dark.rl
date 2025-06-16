from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    """
    A dataclass for storing the configuration of the vLLM engine.
    """

    # The path to the model or a model identifier from Hugging Face.
    model: str = ""
    # The maximum number of tokens that can be processed in a single batch.
    max_num_batched_tokens: int = 32768
    # The maximum number of sequences that can be processed in a single batch.
    max_num_seqs: int = 512
    # The maximum length of a sequence (prompt + completion).
    max_model_len: int = 4096
    # The fraction of GPU memory to be used for the KV cache.
    gpu_memory_utilization: float = 0.9
    # If True, forces the engine to use eager execution instead of CUDA graphs.
    enforce_eager: bool = False
    # The Hugging Face model configuration object.
    hf_config: AutoConfig | None = None
    # The end-of-sentence token ID.
    eos: int = -1
    # The size of a single block in the KV cache, in number of tokens.
    kvcache_block_size: int = 256
    # The total number of blocks in the KV cache. This is calculated at runtime.
    num_kvcache_blocks: int = -1

    # --- LoRA Configuration ---
    # The rank of the LoRA matrices. A value of 0 disables LoRA.
    lora_rank: int = 0
    # The alpha parameter for LoRA scaling.
    lora_alpha: float = 1.0

    def __post_init__(self):
        """Performs validation checks after the object is initialized."""
        assert self.model
        assert self.kvcache_block_size % 256 == 0
