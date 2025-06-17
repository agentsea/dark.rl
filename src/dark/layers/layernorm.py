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

    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
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

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs standard RMSNorm (out-of-place version to keep autograd intact).

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(var + self.variance_epsilon)
        normed = normed.to(orig_dtype)
        return normed * self.weight

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm with a residual connection added first (out-of-place operations).
        """
        orig_dtype = x.dtype
        # Addition in float32 for numerical stability.
        x_fp32 = x.to(torch.float32) + residual.to(torch.float32)
        new_residual = x_fp32.to(orig_dtype)

        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(var + self.variance_epsilon)
        normed = normed.to(orig_dtype)
        return normed * self.weight, new_residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Main forward pass selecting the correct RMSNorm variant."""
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
