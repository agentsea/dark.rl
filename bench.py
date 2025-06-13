import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

# from vllm import LLM, SamplingParams

# This script benchmarks the throughput of the nanovllm engine.
# Throughput is a measure of how many tokens the engine can generate per second.

# --- Benchmark Configuration ---
seed(0)  # Use a fixed seed for reproducibility.
num_seqs = 256  # The number of concurrent sequences in the batch.
max_input_len = 1024  # The maximum length of the randomly generated prompts.
max_ouput_len = 1024  # The maximum number of tokens to generate for each prompt.

# --- Engine Initialization ---
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
# Initialize the LLM engine. `enforce_eager=False` enables CUDA graphs for max performance.
llm = LLM(path, enforce_eager=False, max_model_len=4096)

# --- Data Preparation ---
# Generate a batch of random prompts with varying lengths.
# Each prompt is a list of random token IDs.
prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, max_input_len))]
    for _ in range(num_seqs)
]

# Create a list of sampling parameters, one for each prompt, with varying output lengths.
sampling_params = [
    SamplingParams(
        temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
    )
    for _ in range(num_seqs)
]

# uncomment the following line for vllm
# prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]


# --- Benchmarking ---
# 1. Warm-up run: Perform a single generation to compile kernels and initialize the engine.
#    This ensures that the timing of the main benchmark is not skewed by one-time setup costs.
print("Warming up...")
llm.generate(["Benchmark: "], SamplingParams())

# 2. Main benchmark run: Generate completions for the entire batch of prompts and time it.
print("Running benchmark...")
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = time.time() - t

# --- Results ---
# 3. Calculate and print the throughput.
total_tokens = sum(sp.max_tokens for sp in sampling_params)
throughput = total_tokens / t
print("\n--- Results ---")
print(f"Total Tokens: {total_tokens} tok")
print(f"Time Taken:   {t:.2f} s")
print(f"Throughput:   {throughput:.2f} tok/s")
