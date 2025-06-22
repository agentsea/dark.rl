import torch
from torch import nn


class Sampler(nn.Module):
    """
    A module responsible for sampling the next token from the model's output logits.

    It supports both greedy sampling and temperature-based stochastic sampling.
    """

    def __init__(self):
        super().__init__()

        # Optional: caller can register a tokenizer so debug prints can show
        # human-readable tokens.  We keep this at the *class* level so a single
        # registration in test code affects every Sampler instance.
        Sampler._tokenizer = None
        Sampler._blocked_ids = set()

    # ------------------------------------------------------------------
    # Static helper so integration tests can do:
    #   Sampler.set_tokenizer(tokenizer)
    # to enable rich token decoding in the sampler-debug path.
    # ------------------------------------------------------------------
    @classmethod
    def set_tokenizer(cls, tokenizer):
        cls._tokenizer = tokenizer

    @classmethod
    def set_blocked_ids(cls, ids):
        """Register a collection of token IDs that will be assigned -inf logit
        during sampling so they can never be generated. Useful for preventing
        chat-boundary tokens from appearing in plain completions.
        """
        cls._blocked_ids = set(ids)

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

        # Suppress any blocked tokens before temperature scaling so they do
        # not influence the probability mass.
        if Sampler._blocked_ids:
            blocked = torch.tensor(list(Sampler._blocked_ids), device=logits.device)
            logits.index_fill_(1, blocked, float('-inf'))

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

        # -------------------- DEBUG --------------------
        # For small batches (≤3) print the top-k probabilities so we can see
        # why a particular token was chosen.  This is gated to avoid flooding
        # logs during large-batch training.
        if logits.size(0) <= 3:
            topk_tokens = 5
            # Hard-code a few common Qwen special-token IDs so we can mark them
            # in the debug output without needing a tokenizer reference.
            _special_ids = {
                151643,  # <|endoftext|> (typical)
                151644,  # <|im_start|>
                151645,  # <|im_end|>
                151646, 151647, 151648, 151649,
                151650, 151651, 151652, 151653, 151654, 151655, 151656,
            }

            for i in range(logits.size(0)):
                topk = torch.topk(probs[i], topk_tokens)
                pairs = []
                for tid, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    mark = "*" if tid in _special_ids else ""
                    if Sampler._tokenizer is not None:
                        tok_str = Sampler._tokenizer.convert_ids_to_tokens(tid)
                        pairs.append(f"{tid}{mark}:{prob:.4f}('{tok_str}')")
                    else:
                        pairs.append(f"{tid}{mark}:{prob:.4f}")

                chosen_id = sample_tokens[i].item()
                chosen_mark = "*" if chosen_id in _special_ids else ""
                if Sampler._tokenizer is not None:
                    chosen_tok = Sampler._tokenizer.convert_ids_to_tokens(chosen_id)
                    chosen_repr = f"{chosen_id}{chosen_mark} ('{chosen_tok}')"
                else:
                    chosen_repr = f"{chosen_id}{chosen_mark}"

                print(
                    f"[sampler-debug] seq={i} temp={temperatures[i].item():.2f} "
                    f"chosen={chosen_repr} top{topk_tokens}={pairs}",
                    flush=True,
                )
        # ------------------------------------------------

        # Select greedy tokens where temperature is 0, otherwise use the sampled tokens.
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
