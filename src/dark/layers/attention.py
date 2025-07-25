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
from typing import Optional
import torch
from torch import nn
from flash_attn import flash_attn_varlen_func

def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
):
    """Device-aware Flash-Attention wrapper.

    • CUDA   → calls `flash_attn_varlen_func` (fast path).
    • CPU    → falls back to PyTorch `scaled_dot_product_attention` *per* sequence
               derived from `cu_seqlens` so the API surface stays identical.
    """

    if query.device.type != "cuda":
        # -------- CPU fallback --------
        print("[debug] flash_attention_forward: CPU fallback engaged", flush=True)

        total_tokens, n_heads, head_dim = query.shape
        batch = cu_seqlens.numel() - 1

        outputs = []
        start = 0
        for b in range(batch):
            end = cu_seqlens[b + 1].item()
            q_b = query[start:end].transpose(0, 1)  # [H, L, D]
            k_b = key[start:end].transpose(0, 1)
            v_b = value[start:end].transpose(0, 1)

            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_b, v_b,
                attn_mask=None,
                dropout_p=dropout_p,
                scale=softmax_scale,
                is_causal=causal,
            )  # [H, L, D]

            outputs.append(out_b.transpose(0, 1))  # back to [L, H, D]
            start = end

        return torch.cat(outputs, dim=0)

    # -------- CUDA fast path --------
    return flash_attn_varlen_func(
        query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
    )

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """The equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
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
    """The original, stable attention implementation for padded batches."""
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
