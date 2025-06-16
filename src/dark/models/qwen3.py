import torch
from torch import nn
from transformers import Qwen3Config

from dark.layers.activation import SiluAndMul
from dark.layers.attention import Attention
from dark.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dark.layers.layernorm import RMSNorm
from dark.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from dark.layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    """
    The attention block for the Qwen3 model.

    This module implements the full multi-head attention mechanism, including:
    - Fusing the Q, K, and V projections into a single efficient matrix multiplication.
    - Applying Rotary Positional Embeddings (RoPE) to the query and key.
    - Performing the core attention calculation using a high-performance kernel.
    - Applying the final output projection.
    It leverages tensor parallelism for all linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = 1  # get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass for the attention block."""
        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Normalize query and key heads
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # Apply rotary positional embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Core attention calculation
        o = self.attn(q, k, v)

        # Output projection
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    """
    The feed-forward network (MLP) block for the Qwen3 model.

    This implements a SwiGLU-based MLP, which consists of:
    1. A merged column-parallel linear layer for the gate and up projections.
    2. A SiLU activation function combined with element-wise multiplication.
    3. A row-parallel linear layer for the final down projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """Performs the forward pass for the MLP block."""
        # Fused gate and up projection
        gate_up = self.gate_up_proj(x)
        # Apply the SwiGLU activation
        x = self.act_fn(gate_up)
        # Final down projection
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    A single decoder layer for the Qwen3 transformer model.

    This class combines the self-attention block (`Qwen3Attention`) and the MLP
    block (`Qwen3MLP`) with residual connections and layer normalization to form
    a complete transformer decoder layer. It follows the pre-normalization
    style, where layer normalization is applied before the main sub-layer.
    """

    def __init__(
        self,
        config: Qwen3Config,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass for a single decoder layer."""
        # Pre-normalization for the attention block
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self-attention block
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Pre-normalization for the MLP block
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP block
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    The core Qwen3 transformer model.

    This class stacks multiple `Qwen3DecoderLayer` blocks to form the main body
    of the transformer. It also includes the initial token embedding layer and
    the final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3Config,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass for the entire model."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    The final Qwen3 model for causal language modeling.

    This class wraps the core `Qwen3Model` and adds the final language model
    head (`ParallelLMHead`) for predicting token logits. It also includes logic
    for tying the weights of the embedding layer and the LM head, and a helper
    method to freeze the base model for LoRA training.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.model = Qwen3Model(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.tie_word_embeddings = config.tie_word_embeddings
        if self.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def freeze_base_model(self):
        """
        Freezes all parameters of the base model except for the LoRA parameters.
        This is a crucial step for efficient LoRA fine-tuning.
        """
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the model, returning the hidden states
        before the final projection.
        """
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the final token logits from the hidden states.
        """
        logits = self.lm_head(hidden_states)
        return logits
