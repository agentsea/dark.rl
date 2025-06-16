import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """
    A custom activation module that performs a SiLU (Sigmoid-weighted Linear Unit)
    operation followed by an element-wise multiplication. This is a common pattern
    in modern transformer architectures like SwiGLU.

    The input tensor is expected to be composed of two concatenated parts along the
    last dimension, which are split and processed.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x: The input tensor. The last dimension is chunked into two halves.

        Returns:
            The result of applying SiLU to the first half and multiplying it by the
            second half.
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
