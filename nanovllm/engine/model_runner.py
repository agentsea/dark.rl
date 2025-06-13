import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.memory import get_gpu_memory


class ModelRunner:
    """
    Manages the low-level execution of the model.

    This class is responsible for loading the model onto the correct device,
    managing the KV cache memory, preparing the inputs for the model in the
    correct format, and running the forward pass. It distinguishes between
    prefill and decode stages and uses CUDA graphs to optimize the decoding
    process.
    """

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        # Set up the device and default dtype for model execution.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # Instantiate the model with LoRA config, if provided.
        self.model = Qwen3ForCausalLM(
            hf_config,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        load_model(self.model, config.model)

        # If LoRA is enabled, freeze the base model's weights.
        if self.config.lora_rank > 0:
            self.model.freeze_base_model()
            self.model.train()

        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)

        # Capture CUDA graphs for optimized decoding, unless eager mode is enforced.
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Reset the default device and dtype.
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def train(self):
        """Sets the model to training mode."""
        self.model.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()

    def allocate_kv_cache(self, gpu_memory_utilization):
        """
        Allocates a large, contiguous memory pool for the KV cache.

        This pre-allocates a fixed percentage of the available GPU memory to avoid
        slow `cudaMalloc` calls during generation. The memory is then managed
        by the `BlockManager`.
        """
        config = self.config
        hf_config = config.hf_config
        total, used, _ = get_gpu_memory()
        free = total * gpu_memory_utilization - used
        block_bytes = (
            2  # For key and value
            * hf_config.num_hidden_layers
            * self.block_size
            * hf_config.num_key_value_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        config.num_kvcache_blocks = int(free) // block_bytes
        self.kv_cache = torch.zeros(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
        )
        # Assign slices of the cache to each layer in the model.
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Creates a padded tensor of block tables for a batch of sequences."""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        Prepares the inputs for a prefill step.

        This involves concatenating the token IDs from all sequences in the batch
        and creating the necessary metadata tensors (`positions`, `cu_seqlens`,
        `slot_mapping`, etc.) required by the attention kernel.
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = None
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # Only process tokens that are not already in the cache.
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, len(seq))))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # Generate the slot mapping for the new tokens.
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + len(seq.last_block())
                slot_mapping.extend(list(range(start, end)))
        assert len(input_ids) == len(slot_mapping)
        assert len(input_ids) == cu_seqlens_q[-1]
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            context_lens = torch.tensor(
                [len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        # Set the context for the attention kernel.
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        Prepares the inputs for a decode step (generating one token per sequence).

        This is highly optimized as it only needs to process the last token of each
        sequence in the batch.
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            # Map the new token to its slot in the KV cache.
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + len(seq.last_block()) - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # Set the context for the attention kernel.
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """Prepares a tensor of temperatures for the sampler."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    def run_train_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Runs a forward pass for training, without inference mode or CUDA graphs."""
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        """
        Runs the model's forward pass for inference.

        It uses a CUDA graph for decode steps with small batch sizes to
        significantly reduce kernel launch overhead. For prefill or large batches,
        it runs the model eagerly.
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 256:
            # Eager execution for prefill or large batches.
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Optimized execution using a captured CUDA graph.
            bs = input_ids.size(0)
            context = get_context()
            self.reset_graph_vars()
            # Select the appropriate graph for the batch size.
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # Update the graph's input tensors.
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            # Replay the graph.
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def reset_graph_vars(self):
        """Resets the input tensors for the CUDA graph."""
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].zero_()
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].zero_()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        The main run method that orchestrates a single generation step.
        """
        # Prepare inputs based on whether it's a prefill or decode step.
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs)

        # Run the model to get logits.
        logits = self.run_model(input_ids, positions, is_prefill)

        # Sample the next tokens from the logits.
        token_ids = self.sampler(logits, temperatures).tolist()

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        Captures CUDA graphs for various batch sizes.

        This method pre-records the sequence of CUDA kernel launches required for
        a decode step. By replaying this graph instead of launching kernels
        individually, it significantly reduces CPU overhead and improves
        performance for small batch sizes. It creates a graph for several
        pre-defined batch sizes.
        """
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Pre-allocate static tensors for graph inputs.
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Capture a graph for each batch size.
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture

            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # Store the static input tensors for later use.
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
