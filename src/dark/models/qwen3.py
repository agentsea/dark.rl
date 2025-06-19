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
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import os

from transformers import Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from dark.layers.attention import eager_attention_forward
from dark.layers.layernorm import RMSNorm
from dark.layers.linear import ReplicatedLinear
from dark.layers.rotary_embedding import apply_rotary_pos_emb

logger = logging.get_logger(__name__)


class Qwen3MLP(nn.Module):
    def __init__(self, config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = ReplicatedLinear(
            self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.up_proj = ReplicatedLinear(
            self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.down_proj = ReplicatedLinear(
            self.intermediate_size, self.hidden_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = ReplicatedLinear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.k_proj = ReplicatedLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.v_proj = ReplicatedLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.o_proj = ReplicatedLinear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Decide path based on mode; keep original for training stability.
        use_flash = (
            (not self.training)
            and os.getenv("ENABLE_FLASH_ATTENTION", "0") == "1"
            and torch.cuda.is_available()
            and torch.backends.cuda.flash_sdp_enabled
        )

        if use_flash:
            # --- Flash attention 2 (PyTorch SDP) -----------------------------
            # For MQA/GQA, explicitly repeat K/V heads to match Q heads.
            if self.num_key_value_heads != self.num_attention_heads:
                key_states   = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

            # Upcast *all* inputs to float32 for stable SDP calculation.
            # The kernel handles MQA/GQA broadcasting and scaling internally.
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states.float(),
                key_states.float(),
                value_states.float(),
                attn_mask=attention_mask.float(),
                dropout_p=0.0,
                is_causal=False,
            ).to(hidden_states.dtype)
            attn_output = attn_output.transpose(1, 2)
            attn_weights = None
        else:
            # Eager path.
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                dropout=0.0,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.self_attn = Qwen3Attention(config=config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.mlp = Qwen3MLP(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, self_attn_weights)
        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, ReplicatedLinear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # The unsqueeze operation is moved to apply_rotary_pos_emb
        return cos, sin


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config, lora_rank=0, lora_alpha=1.0):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, lora_rank=lora_rank, lora_alpha=lora_alpha) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        causal_mask = torch.full(
            (hidden_states.size(1), hidden_states.size(1)),
            fill_value=torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device,
        )
        causal_mask = causal_mask.triu(diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(hidden_states.size(0), 1, -1, -1)
        attention_mask = causal_mask.masked_fill(attention_mask[:, None, None, :]==0, torch.finfo(hidden_states.dtype).min)

        position_embeddings = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])

        for decoder_layer in self.layers:
            if self.training and self.gradient_checkpointing:
                # -- selective checkpointing ("unsloth" style) ------------
                def _ckpt_forward(hs):
                    return decoder_layer(
                        hs,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                    )[0]

                hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt_forward,
                    hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
                hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    def __init__(self, config, lora_rank=0, lora_alpha=1.0):
        super().__init__(config)
        self.model = Qwen3Model(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def freeze_base_model(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
