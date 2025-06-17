import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator, denominator):
    """A simple utility function for integer division, with an assertion."""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    An abstract base class for all linear layers in the model.

    It initializes basic properties related to tensor parallelism, such as the
    rank and size of the tensor parallel group.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = 0  # get_tensor_model_parallel_rank()
        self.tp_size = 1  # get_tensor_model_parallel_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    A standard linear layer where the weights are fully replicated across all GPUs.

    This is the simplest type of linear layer, used when tensor parallelism is not
    required for this specific layer. It also includes the logic for applying LoRA
    (Low-Rank Adaptation) if `lora_rank` is greater than 0.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.lora_rank = lora_rank
        if lora_rank > 0:
            # Initialize LoRA parameters
            self.lora_a = nn.Parameter(torch.empty(lora_rank, self.input_size))
            self.lora_b = nn.Parameter(torch.empty(self.output_size, lora_rank))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)
            self.scaling = lora_alpha / lora_rank

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader for the weight parameter."""
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass, adding the LoRA path if applicable.
        """
        # Main linear transformation
        y = F.linear(x, self.weight, self.bias)
        # Add the LoRA result if LoRA is enabled
        if self.lora_rank > 0 and self.training:
            lora_x = F.linear(x, self.lora_a)
            lora_x = F.linear(lora_x, self.lora_b)
            lora_x = lora_x * self.scaling
            y = y + lora_x
        elif self.lora_rank > 0:
            lora_x = F.linear(x, self.lora_a)
            lora_x = F.linear(lora_x, self.lora_b)
            y = y + lora_x * self.scaling
        return y


class ColumnParallelLinear(LinearBase):
    """
    A linear layer that implements column-wise tensor parallelism.

    The weight matrix is partitioned along the output dimension (columns). Each GPU
    in the tensor parallel group holds a vertical slice of the weight matrix.
    The forward pass is computationally efficient as it requires no communication;
    each GPU computes its part of the output, which is a slice of the full output.
    This also integrates LoRA, with the LoRA 'B' matrix being similarly partitioned.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If this layer is a specialized type like QKV or MergedColumn,
        # it might have multiple output partitions.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.lora_rank = lora_rank
        if lora_rank > 0:
            # LoRA A is replicated, LoRA B is column-parallel
            self.lora_a = nn.Parameter(
                torch.empty(lora_rank, self.input_size_per_partition)
            )
            self.lora_b = nn.Parameter(
                torch.empty(self.output_size_per_partition, lora_rank)
            )
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)
            self.scaling = lora_alpha / lora_rank

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader to load the correct vertical slice of the weight matrix."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the column-parallel forward pass. No communication is needed.
        """
        y = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            lora_x = F.linear(x, self.lora_a)
            lora_x = F.linear(lora_x, self.lora_b)
            y = y + lora_x * self.scaling
        return y


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    A specialized ColumnParallelLinear layer that handles multiple weight matrices
    concatenated together. This is often used for the Q, K, and V projections in
    attention, or the gate and up projections in an MLP, to fuse them into a
    single, larger matrix multiplication for efficiency.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        self.output_sizes = output_sizes
        tp_size = 1  # get_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        """Loads a specific part (e.g., gate_proj) of the merged weight."""
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # loaded_weight = loaded_weight.narrow(self.tp_dim, self.tp_rank * shard_size, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    A specialized ColumnParallelLinear layer for the Q, K, and V projection weights
    in an attention mechanism, handling the specific partitioning of heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the heads across the tensor parallel group.
        tp_size = 1  # get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = self.hidden_size
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size, output_size, bias=bias, lora_rank=lora_rank, lora_alpha=lora_alpha
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        """Loads the correct shard for Q, K, or V into the merged weight matrix."""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # loaded_weight = loaded_weight.narrow(self.tp_dim, self.tp_rank * shard_size, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    A linear layer that implements row-wise tensor parallelism.

    The weight matrix is partitioned along the input dimension (rows). Each GPU
    holds a horizontal slice of the weight matrix. The forward pass requires an
    all-reduce communication step at the end to sum the partial results from each
    GPU to produce the final, correct output. This also integrates LoRA, with the
    LoRA 'A' matrix being partitioned to match the weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.lora_rank = lora_rank
        if lora_rank > 0:
            # LoRA A is row-parallel, LoRA B is replicated
            self.lora_a = nn.Parameter(
                torch.empty(lora_rank, self.input_size_per_partition)
            )
            self.lora_b = nn.Parameter(
                torch.empty(self.output_size_per_partition, lora_rank)
            )
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)
            self.scaling = lora_alpha / lora_rank

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader to load the correct horizontal slice of the weight matrix."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the row-parallel forward pass, with a final all-reduce.
        """
        # Each GPU computes a partial result based on its slice of the weights.
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # The LoRA path is computed before the all-reduce.
        if self.lora_rank > 0:
            lora_x = F.linear(x, self.lora_a)
            lora_x = F.linear(lora_x, self.lora_b)
            y = y + lora_x * self.scaling

        # Sum the partial results from all GPUs.
        if self.tp_size > 1:
            dist.all_reduce(y)

        return y
