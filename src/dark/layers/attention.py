import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    A Triton kernel to store key and value tensors into the KV cache.

    This kernel is designed for efficiency on the GPU. Each program instance
    handles one token's key/value pair, computing the memory offsets and
    storing the data into the correct cache slot as determined by slot_mapping.

    Args:
        key_ptr: Pointer to the key tensor.
        key_stride: Stride of the key tensor for memory access.
        value_ptr: Pointer to the value tensor.
        value_stride: Stride of the value tensor for memory access.
        k_cache_ptr: Pointer to the key cache tensor.
        v_cache_ptr: Pointer to the value cache tensor.
        slot_mapping_ptr: Pointer to the tensor that maps each input token to
                          its corresponding slot in the cache.
        D: The size of the head dimension, passed as a compile-time constant.
    """
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """

    A host function to launch the Triton kernel for storing the KV cache.

    It performs necessary assertions and computes parameters for the kernel launch.

    Args:
        key: The key tensor of shape (N, num_heads, head_dim).
        value: The value tensor of shape (N, num_heads, head_dim).
        k_cache: The key cache tensor.
        v_cache: The value cache tensor.
        slot_mapping: A tensor that maps each of the N input tokens to a slot
                      in the cache.
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    """
        The core attention module.

        This module orchestrates the attention mechanism, dynamically switching
        between prefill and decode modes. It uses flash-attention for the core
    alculation
        and a custom Triton kernel to efficiently manage the KV cache.
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Performs the attention forward pass.

        The behavior changes based on the context (prefill vs. decode):
        1. Reshapes Q, K, and V tensors.
        2. Retrieves the current execution context (prefill or decode).
        3. Stores the current K/V pairs into the cache using the Triton kernel.
        4. If in prefill mode, it uses flash_attn_varlen_func for variable
           length sequences.
        5. If in decode mode, it uses flash_attn_with_kvcache, which is
           optimized for single-token generation with a KV cache.
        6. Reshapes the output back to the expected format.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.

        Returns:
            The output tensor from the attention calculation.
        """
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
