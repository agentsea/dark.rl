# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from transformers import Qwen3Config  # We'll create a MoE config based on this
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from dark.config import Config
from dark.layers.linear import ReplicatedLinear as LoRALinear

logger = logging.get_logger(__name__)


class Qwen3MoeConfig(Qwen3Config):
    """Configuration for Qwen3MoE model, extending Qwen3Config with MoE-specific parameters"""
    
    model_type = "qwen3_moe"
    
    def __init__(
        self,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # MoE-specific parameters
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """
    Computes auxiliary load balancing loss as in Switch Transformer.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if lora_rank > 0:
            self.q_proj = LoRALinear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
            self.k_proj = LoRALinear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
            self.v_proj = LoRALinear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
            self.o_proj = LoRALinear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
            )
            self.k_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )
        
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        
        if lora_rank > 0:
            self.gate_proj = LoRALinear(self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
            self.up_proj = LoRALinear(self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
            self.down_proj = LoRALinear(self.intermediate_size, self.hidden_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        if lora_rank > 0:
            self.gate = LoRALinear(config.hidden_size, config.num_experts, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        else:
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size, lora_rank=lora_rank, lora_alpha=lora_alpha) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx, lora_rank=lora_rank, lora_alpha=lora_alpha)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size, lora_rank=lora_rank, lora_alpha=lora_alpha)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_router_logits and router_logits is not None:
            outputs += (router_logits,)

        return outputs


class Qwen3MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, LoRALinear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3MoeRMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, device=None):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = config.max_position_embeddings
        self._set_cos_sin_cache(
            seq_len=config.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        # Convert our Config to Qwen3MoeConfig for compatibility
        if hasattr(config, 'hf_config'):
            hf_config = config.hf_config
        else:
            # Create a basic Qwen3MoeConfig from our config
            hf_config = Qwen3MoeConfig(
                vocab_size=getattr(config, 'vocab_size', 151936),
                hidden_size=getattr(config, 'hidden_size', 2048),
                intermediate_size=getattr(config, 'intermediate_size', 6144),
                num_hidden_layers=getattr(config, 'num_hidden_layers', 24),
                num_attention_heads=getattr(config, 'num_attention_heads', 32),
                num_key_value_heads=getattr(config, 'num_key_value_heads', 4),
                head_dim=getattr(config, 'head_dim', None),
                max_position_embeddings=getattr(config, 'max_position_embeddings', 32768),
                rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
                rope_theta=getattr(config, 'rope_theta', 10000.0),
                attention_bias=getattr(config, 'attention_bias', False),
                attention_dropout=getattr(config, 'attention_dropout', 0.0),
                use_sliding_window=getattr(config, 'use_sliding_window', False),
                sliding_window=getattr(config, 'sliding_window', 4096),
                # MoE-specific parameters
                decoder_sparse_step=getattr(config, 'decoder_sparse_step', 1),
                moe_intermediate_size=getattr(config, 'moe_intermediate_size', 768),
                num_experts_per_tok=getattr(config, 'num_experts_per_tok', 8),
                num_experts=getattr(config, 'num_experts', 128),
                norm_topk_prob=getattr(config, 'norm_topk_prob', False),
                output_router_logits=getattr(config, 'output_router_logits', False),
                router_aux_loss_coef=getattr(config, 'router_aux_loss_coef', 0.001),
                mlp_only_layers=getattr(config, 'mlp_only_layers', []),
            )
            
        super().__init__(hf_config)
        self.config = hf_config
        self.padding_idx = hf_config.pad_token_id
        self.vocab_size = hf_config.vocab_size

        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(hf_config, layer_idx, lora_rank=lora_rank, lora_alpha=lora_alpha) for layer_idx in range(hf_config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=hf_config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
        
        # Convert attention mask to causal mask
        if attention_mask.dim() == 2:
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=inputs_embeds.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
            causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
            causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
        else:
            causal_mask = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_router_logits = () if output_router_logits else None
        
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                output_router_logits=output_router_logits,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            
            if output_router_logits and len(layer_outputs) > 1:
                all_router_logits += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Create custom output with router logits
        class MoeModelOutput:
            def __init__(self, last_hidden_state, router_logits=None):
                self.last_hidden_state = last_hidden_state
                self.past_key_values = None
                self.hidden_states = None
                self.attentions = None
                self.router_logits = router_logits

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            router_logits=all_router_logits,
        )


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        # Convert our Config to Qwen3MoeConfig for compatibility
        if hasattr(config, 'hf_config'):
            hf_config = config.hf_config
        else:
            # Create a basic Qwen3MoeConfig from our config
            hf_config = Qwen3MoeConfig(
                vocab_size=getattr(config, 'vocab_size', 151936),
                hidden_size=getattr(config, 'hidden_size', 2048),
                intermediate_size=getattr(config, 'intermediate_size', 6144),
                num_hidden_layers=getattr(config, 'num_hidden_layers', 24),
                num_attention_heads=getattr(config, 'num_attention_heads', 32),
                num_key_value_heads=getattr(config, 'num_key_value_heads', 4),
                head_dim=getattr(config, 'head_dim', None),
                max_position_embeddings=getattr(config, 'max_position_embeddings', 32768),
                rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
                rope_theta=getattr(config, 'rope_theta', 10000.0),
                attention_bias=getattr(config, 'attention_bias', False),
                attention_dropout=getattr(config, 'attention_dropout', 0.0),
                use_sliding_window=getattr(config, 'use_sliding_window', False),
                sliding_window=getattr(config, 'sliding_window', 4096),
                # MoE-specific parameters
                decoder_sparse_step=getattr(config, 'decoder_sparse_step', 1),
                moe_intermediate_size=getattr(config, 'moe_intermediate_size', 768),
                num_experts_per_tok=getattr(config, 'num_experts_per_tok', 8),
                num_experts=getattr(config, 'num_experts', 128),
                norm_topk_prob=getattr(config, 'norm_topk_prob', False),
                output_router_logits=getattr(config, 'output_router_logits', False),
                router_aux_loss_coef=getattr(config, 'router_aux_loss_coef', 0.001),
                mlp_only_layers=getattr(config, 'mlp_only_layers', []),
            )
            
        super().__init__(hf_config)
        self.model = Qwen3MoeModel(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.vocab_size = hf_config.vocab_size
        self.router_aux_loss_coef = hf_config.router_aux_loss_coef
        self.num_experts = hf_config.num_experts
        self.num_experts_per_tok = hf_config.num_experts_per_tok
        
        if lora_rank > 0:
            self.lm_head = LoRALinear(hf_config.hidden_size, hf_config.vocab_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        else:
            self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs
    ):
        
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        # Create custom output with aux_loss
        class MoeCausalLMOutput:
            def __init__(self, loss=None, aux_loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None, router_logits=None):
                self.loss = loss
                self.aux_loss = aux_loss
                self.logits = logits
                self.past_key_values = past_key_values
                self.hidden_states = hidden_states
                self.attentions = attentions
                self.router_logits = router_logits

        return MoeCausalLMOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def freeze_base_model(self):
        """Freeze all parameters except LoRA parameters"""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False


def create_qwen3_moe_model(config: Config, lora_rank=0, lora_alpha=1.0):
    """Create a Qwen3MoE model using the upstream implementation"""
    model = Qwen3MoeForCausalLM(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
    
    # Freeze base model parameters if using LoRA
    if lora_rank > 0:
        model.freeze_base_model()
    
    return model
