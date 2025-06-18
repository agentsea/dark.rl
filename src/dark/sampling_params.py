from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    A dataclass for storing the parameters that control the sampling process.
    """

    # The number of completions to generate for each prompt.
    n: int = 1
    # The temperature for sampling. A value of 0 means greedy decoding. Higher
    # values make the output more random.
    temperature: float = 1.0
    # The maximum number of tokens to generate for the completion.
    max_tokens: int = 64
    # Positive values penalize new tokens based on whether they appear in the
    # text so far, increasing the model's likelihood to talk about new topics.
    presence_penalty: float = 0.0
    # If True, the end-of-sentence (EOS) token will be ignored, and generation
    # will continue until `max_tokens` is reached.
    ignore_eos: bool = False
