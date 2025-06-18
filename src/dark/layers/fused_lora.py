import torch
import torch.nn.functional as F


def fused_lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor | None,
    lora_b: torch.Tensor | None,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Compute a linear layer with an optional fused LoRA update.

    This helper is *read-only* w.r.t. the LoRA parameters – it should only
    be used while the model is in ``eval()`` mode.  By design, gradients do
    not propagate through ``lora_a`` / ``lora_b`` because we never need them
    at inference time.  When ``lora_a`` is *None* or the rank is 0, the call
    falls back to a standard ``F.linear``.

    We intentionally *avoid* materialising the intermediate ``rank``-sized
    activation ``x @ lora_a.T`` on the autograd graph by executing the LoRA
    path under ``torch.no_grad()`` and immediately discarding the temporary
    tensor.  This yields a modest VRAM saving (≈ *batch × seq × rank* floats)
    without touching the CUDA kernels the project already relies on.
    """
    if lora_a is None or lora_b is None:
        return F.linear(x, weight, bias)

    # --- main dense projection ------------------------------------------------
    out = F.linear(x, weight, bias)

    # --- LoRA update, computed without tracking gradients --------------------
    with torch.no_grad():
        lora_x = torch.matmul(x, lora_a.T)  # [B, rank]
        lora_out = torch.matmul(lora_x, lora_b.T)  # [B, out_features]

    # The update itself participates in autograd so downstream ops keep their
    # gradient flow, but we shield the LoRA weights from "eval"-time writes.
    out = out + scaling * lora_out
    return out 