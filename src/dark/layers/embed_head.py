import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    An embedding layer that supports vocabulary parallelism.

    The vocabulary is partitioned across multiple GPUs (tensor parallel group).
    Each GPU holds a shard of the embedding weight matrix. During the forward
    pass, each GPU looks up the embeddings for the tokens that fall within its
    assigned vocabulary range. An all-reduce operation is then used to sum the
    partial results from all GPUs, ensuring that every token gets its correct
    embedding vector regardless of which GPU it was on.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = 0  # get_tensor_model_parallel_rank()
        self.tp_size = 1  # get_tensor_model_parallel_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader to correctly load a shard of weights into the parameter."""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        Performs the vocabulary-parallel embedding lookup.

        If tensor parallelism is used, it masks the input tokens so that each GPU
        only handles its assigned vocabulary shard. The resulting embeddings are
        summed across all GPUs using an all-reduce operation.
        """
        if self.tp_size > 1:
            # Mask out tokens that are not in the current GPU's vocabulary partition.
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust token IDs to be relative to the start of the partition.
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # Mask the output to zero out embeddings for tokens not in this partition.
            y = mask * y
            # Sum the partial embeddings from all GPUs.
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    A language model head that supports vocabulary parallelism.

    This class inherits from VocabParallelEmbedding to reuse the weight sharding
    logic. It performs a linear transformation to produce logits for the vocabulary.
    In prefill mode, it efficiently computes logits only for the last token of
    each sequence.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        Computes logits for the language model.

        During prefill, it optimizes by only processing the last token of each
        sequence. The final logits are only returned by the master GPU (rank 0).
        """
        context = get_context()
        if context.is_prefill:
            # For prefill, we only need to compute logits for the last token of each sequence.
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        logits = F.linear(x, self.weight, self.bias)

        # NOTE: The following code for gathering logits is commented out.
        # In the current implementation, loss is likely computed in a distributed
        # manner, and only the final sampled tokens need to be gathered.
        # if self.tp_size > 1:
        #     all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
        #     dist.gather(logits, all_logits, 0)
        #     logits = torch.cat(all_logits, -1)

        return logits if self.tp_rank == 0 else None
