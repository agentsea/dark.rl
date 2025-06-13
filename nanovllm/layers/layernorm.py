import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    A custom implementation of Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm normalizes the activations of a layer by their root mean square.
    It is computationally simpler than standard Layer Normalization and has been
    shown to be effective in many state-of-the-art models. This implementation
    can also handle an optional residual connection, adding it to the input
    before normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs standard RMSNorm.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        orig_dtype = x.dtype
        # Calculations are done in float32 for precision.
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize and apply the learnable weight.
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Performs RMSNorm with a residual connection added first.

        This is a common "pre-layernorm" pattern. The residual is added to the
        input *before* the normalization is applied.

        Args:
            x: The input tensor.
            residual: The residual tensor to be added to the input.

        Returns:
            A tuple containing the normalized tensor and the new residual,
            which is the sum of the input and the old residual.
        """
        orig_dtype = x.dtype
        # Add residual in float32 for precision.
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        # The new residual is the result of the addition.
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize and apply the learnable weight.
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """

        Main forward pass that dispatches to the correct normalization function.
        If a residual is provided, it uses the 'add_rms_forward' method.
        Otherwise, it uses the standard 'rms_forward' method.
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
