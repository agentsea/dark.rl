from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator


# Qwen/Qwen3-32B
# Text Generation
# •
# Updated May 21
# •
# 759k
# •
# •
# 404

# Qwen/Qwen3-14B
# Text Generation
# •
# Updated May 21
# •
# 971k
# •
# •
# 192

# Qwen/Qwen3-8B
# Text Generation
# •
# Updated May 21
# •
# 1.32M
# •
# •
# 405

# Qwen/Qwen3-4B
# Text Generation
# •
# Updated May 21
# •
# 969k
# •
# •
# 259

# Qwen/Qwen3-1.7B
# Text Generation
# •
# Updated May 21
# •
# 649k
# •
# •
# 159

# Qwen/Qwen3-0.6B
# Text Generation
# •


SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-0.6B-Instruct",
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-8B-Instruct",
    "Qwen/Qwen3-14B-Instruct",
    "Qwen/Qwen3-32B-Instruct",
]

class OnlineLLM:
    """An online LLM that can be used to chat and learn.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 12):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported. Supported models: {SUPPORTED_MODELS}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        pass

    def stream(self, prompt: str) -> Generator[str, None, None]:
        pass

    def learn(self, prompt: str, response: str):
        pass

    def chat(self, msgs: List[Dict[str, Any]]) -> str:
        return self.generate(msgs)

    def chat_stream(self, msgs: List[Dict[str, Any]]) -> Generator[str, None, None]:
        return self.stream(msgs)

    def learn_chat(self, msgs: List[Dict[str, Any]]):
        return self.learn(msgs)
