import os
import sys
import torch
from transformers import AutoTokenizer
from dark import LLM, SamplingParams

# --- Script Configuration ---
MODEL_PATH = "Qwen/Qwen3-0.6B"

# A quirky target sentence we want the tiny LoRA fine-tune to memorize.
TARGET_SENTENCE = "Purple ducklings whistle jazz tunes."
TARGET_PROMPT = "Purple ducklings"

def main():
    """
    An integration script that performs a few training steps and then runs inference.
    """
    # 1. Initialize tokenizer and LLM
    print("--> Initializing tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # enforce_eager is needed for training to work correctly with .backward()
    llm = LLM(MODEL_PATH, enforce_eager=True, lora_rank=8)
    print("--> Initialization complete.")

    # 2. Switch to evaluation mode and run inference
    print("\n--> Running inference...")
    llm.eval()
    
    prompts = ["The quick brown fox"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # 3. Check inference output and print results
    generated_text = outputs[0]['text']
    print(f"\nPrompt: {prompts[0]!r}")
    print(f"Generated text: {generated_text!r}")
    
    if "jumps" not in generated_text.lower():
        print("Error: Generated text did not contain the expected continuation.", file=sys.stderr)
        sys.exit(1)

    print("\nIntegration test passed!")

    # 2b. Baseline generation with the target prompt before fine-tuning.
    print("\n--> Baseline generation for target prompt (before fine-tune)...")
    baseline_outputs = llm.generate([TARGET_PROMPT], sampling_params, use_tqdm=False)
    baseline_text = baseline_outputs[0]['text']
    print(f"Prompt: {TARGET_PROMPT!r}\nBaseline generated text: {baseline_text!r}")
    # Not asserting anything here—goal is just to inspect difference after fine-tune.

    # 3. Prepare training data
    print("\n--> Preparing training data...")
    input_ids = tokenizer.encode(TARGET_SENTENCE, return_tensors="pt").cuda()

    # 4. Set up optimizer
    optimizer = torch.optim.Adam(llm.model_runner.model.parameters(), lr=1e-3)

    # 5. Training loop
    print("\n--> Starting training loop...")
    llm.train()
    initial_loss = None
    final_loss = None

    for i in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        loss = llm.forward_train(input_ids)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Step {i}, Loss: {loss.item()}")
        if i == 0:
            initial_loss = loss.item()
        if i == 19:
            final_loss = loss.item()
    print("--> Training complete.")

    # Assert that the loss has decreased
    if final_loss is not None and initial_loss is not None and final_loss >= initial_loss:
        print("Error: Loss did not decrease during training", file=sys.stderr)
        sys.exit(1)

    # 6. Test that the model has learned to produce the target continuation
    print("\n--> Verifying fine-tuned behavior...")
    llm.eval()
    outputs_ft = llm.generate([TARGET_PROMPT], sampling_params, use_tqdm=False)
    generated_ft = outputs_ft[0]['text']
    print(f"Prompt: {TARGET_PROMPT!r}\nGenerated text after fine-tune: {generated_ft!r}")

    if "jazz" not in generated_ft.lower():
        print("Error: Fine-tuned model did not produce the expected phrase.", file=sys.stderr)
        sys.exit(1)

    print("\nAll integration checks passed!")

if __name__ == "__main__":
    main() 