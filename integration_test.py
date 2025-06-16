import os
import sys
import torch
from transformers import AutoTokenizer
from dark import LLM, SamplingParams

# --- Script Configuration ---
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

def main():
    """
    An integration script that performs a few training steps and then runs inference.
    """
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Please download the model or update the MODEL_PATH variable.", file=sys.stderr)
        sys.exit(1)

    # 2. Initialize tokenizer and LLM
    print("--> Initializing tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # enforce_eager is needed for training to work correctly with .backward()
    llm = LLM(MODEL_PATH, enforce_eager=True, lora_rank=8)
    print("--> Initialization complete.")

    # 3. Prepare training data
    print("\n--> Preparing training data...")
    training_sentence = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(training_sentence, return_tensors="pt")[0].cuda()
    labels = input_ids.clone()

    # For causal LM, the labels are the input ids shifted by one
    input_ids = input_ids[:-1]
    labels = labels[1:]
    
    positions = torch.arange(0, len(input_ids), dtype=torch.long).cuda()
    print("--> Data preparation complete.")

    # 4. Set up optimizer and loss function
    optimizer = torch.optim.Adam(llm.model_runner.model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. Training loop
    print("\n--> Starting training loop...")
    llm.train()
    initial_loss = None
    final_loss = None

    for i in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        logits = llm.forward_train(input_ids, positions)
        
        # Calculate loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Step {i}, Loss: {loss.item()}")
        if i == 0:
            initial_loss = loss.item()
        if i == 4:
            final_loss = loss.item()
    print("--> Training complete.")

    # Assert that the loss has decreased
    if not (initial_loss is not None and final_loss is not None and final_loss < initial_loss):
        print("\nError: Loss did not decrease during training.", file=sys.stderr)
        sys.exit(1)
    
    print("\nSuccessfully confirmed that loss decreased during training.")

    # 6. Switch to evaluation mode and run inference
    print("\n--> Running inference...")
    llm.eval()
    
    prompts = ["The quick brown fox"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # 7. Check inference output and print results
    generated_text = outputs[0]['text']
    print(f"\nPrompt: {prompts[0]!r}")
    print(f"Generated text: {generated_text!r}")
    
    if "jumps" not in generated_text.lower():
        print("Error: Generated text did not contain the expected continuation.", file=sys.stderr)
        sys.exit(1)

    print("\nIntegration test passed!")


if __name__ == "__main__":
    main() 