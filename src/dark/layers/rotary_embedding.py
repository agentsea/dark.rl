from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to an input tensor.

    This function takes a tensor `x` and rotates its components using the
    pre-computed cosine and sine values. The input tensor is split into two
    halves, which are then rotated and concatenated back together.

    Args:
        x: The input tensor to which RoPE will be applied.
        cos: The cosine components of the rotation.
        sin: The sine components of the rotation.

    Returns:
        The transformed tensor with rotary embeddings applied.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    # The tensor is split into two halves for the rotation operation.
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    # The rotation is applied.
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    A module that computes and applies Rotary Positional Embeddings (RoPE).

    RoPE encodes absolute position information by rotating chunks of the query
    and key vectors by an angle that depends on their position. This module
    pre-computes the sine and cosine values for all possible positions up to
    `max_position_embeddings` and stores them in a cache for efficient lookup
    during the forward pass.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Pre-compute the inverse frequencies for the sinusoidal embeddings.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # Pre-compute and cache the cosine and sine values.
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to the query and key tensors.

        It looks up the pre-computed cosine and sine values from the cache based
        on the token positions and then applies the rotation.

        Args:
            positions: A tensor containing the positions of the tokens.
            query: The query tensor.
            key: The key tensor.

        Returns:
            The transformed query and key tensors with RoPE applied.
        """
        positions = positions.flatten()
        # Look up the cos and sin values from the cache for the given positions.
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Apply the rotary embeddings to the query tensor.
        query_shape = query.shape
        query = query.view(positions.shape[0], -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)

        # Apply the rotary embeddings to the key tensor.
        key_shape = key.shape
        key = key.view(positions.shape[0], -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    A factory function to create and cache a RotaryEmbedding instance.

    Uses an LRU cache to ensure that only one instance of the RotaryEmbedding
    module is created for a given set of parameters, improving efficiency.
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
