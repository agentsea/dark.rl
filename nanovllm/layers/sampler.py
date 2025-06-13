import torch
from torch import nn


class Sampler(nn.Module):
    """
    A module responsible for sampling the next token from the model's output logits.

    It supports both greedy sampling and temperature-based stochastic sampling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        Performs the sampling operation.

        The method uses a conditional approach:
        - If the temperature for a sequence is 0, it performs greedy sampling
          (i.e., picks the token with the highest logit).
        - If the temperature is greater than 0, it performs stochastic sampling.
          The logits are scaled by the temperature, and then the Gumbel-Max trick
          is used for efficient and differentiable sampling from the resulting
          distribution.

        Args:
            logits: The output logits from the model of shape (batch_size, vocab_size).
            temperatures: A tensor of temperatures for each sequence in the batch.

        Returns:
            A tensor containing the sampled token IDs for each sequence in the batch.
        """
        # Ensure logits are in float for calculations.
        logits = logits.to(torch.float)

        # Determine the tokens that would be chosen by greedy sampling.
        greedy_tokens = logits.argmax(dim=-1)

        # Scale logits by temperature for stochastic sampling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Convert logits to probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # Use the Gumbel-Max trick for sampling. This is equivalent to sampling
        # from a categorical distribution but can be more efficient.
        # It involves adding Gumbel noise (by taking -log(-log(uniform_random))))
        # to the log-probabilities and then taking the argmax.
        # A simpler way to achieve this is to divide the probabilities by
        # random exponential values and take the argmax.
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(
            dim=-1
        )

        # Select greedy tokens where temperature is 0, otherwise use the sampled tokens.
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
