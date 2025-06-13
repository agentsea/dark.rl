import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

# This script provides a simple example of how to use the nanovllm engine
# directly as a Python library, without the API server.

# 1. Define the path to the model and initialize the tokenizer and the LLM engine.
#    `enforce_eager=True` is used here for simplicity, disabling CUDA graphs.
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True)

# 2. Define the sampling parameters for generation.
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 3. Create a list of prompts.
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]

# 4. (Optional) Apply a chat template to format the prompts correctly for the model.
#    This is a good practice for instruction-following or chat models.
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    for prompt in prompts
]

# 5. Call the `generate` method to get the model's outputs.
#    The engine handles batching and generation internally.
outputs = llm.generate(prompts, sampling_params)

# 6. Print the results.
for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")
