import os
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from dark import LLM, SamplingParams

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Nano-vLLM API",
    description="API for unified LLM training and inference.",
)


# --- Global Model and Tokenizer Initialization ---
# The LLM engine and tokenizer are loaded once at startup to avoid reloading them
# for every request, which would be extremely slow.
print("Loading model and tokenizer...")
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
# For training, LoRA is enabled by default with a rank of 8.
# This can be configured or changed based on requirements.
llm = LLM(path, enforce_eager=True, lora_rank=8, lora_alpha=16)
tokenizer = AutoTokenizer.from_pretrained(path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model and tokenizer loaded.")


# --- API Request and Response Models ---


class GenerateRequest(BaseModel):
    """Defines the expected input for an inference request."""

    prompts: List[str]
    temperature: float = 0.6
    max_tokens: int = 256


class TrainRequest(BaseModel):
    """Defines the expected input for a training request."""

    adapter_id: str
    data: List[
        Dict[str, Any]
    ]  # Expects a list of dictionaries, e.g., [{"text": "..."}]
    lr: float = 1e-4


@app.post("/generate", summary="Generate text from prompts")
async def generate(request: GenerateRequest):
    """
    Handles inference requests.

    It takes a list of prompts and generates a completion for each one using
    the sampling parameters provided in the request.
    """
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # The llm.generate method handles batching and generation internally.
    outputs = llm.generate(request.prompts, sampling_params)

    # The output from llm.generate is a list of dicts, which is JSON-serializable.
    return outputs


@app.post("/train", summary="Train a LoRA adapter")
async def train(request: TrainRequest):
    """
    Handles online training requests for a LoRA adapter.

    This endpoint performs a simple training loop on the provided data. It ensures
    that the model is properly switched to training mode and then returned to
    evaluation mode, making it safe to serve inference requests immediately after.
    """
    # 1. Get trainable LoRA parameters and create an optimizer.
    # This filters for parameters that were enabled for training (i.e., "lora_").
    trainable_params = [
        p for p in llm.model_runner.model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=request.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    # Use a try...finally block to ensure the model is always set back to eval mode.
    try:
        llm.train()
        # 2. Training loop over the provided data.
        for item in request.data:
            if "text" not in item:
                continue

            # 3. Tokenization and data preparation for causal language modeling.
            # The model is trained to predict the next token in the sequence.
            input_ids = tokenizer(item["text"], return_tensors="pt").input_ids.to(
                "cuda"
            )

            if input_ids.shape[1] < 2:  # Need at least 2 tokens for a label.
                continue

            # The labels are the input_ids shifted by one token to the left.
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            positions = torch.arange(0, input_ids.shape[1], device="cuda").unsqueeze(0)

            # 4. Forward pass through the model (with gradients enabled).
            logits = llm.forward_train(input_ids, positions)

            # 5. Calculate the cross-entropy loss.
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()

            # 6. Backward pass and optimizer step.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    finally:
        # Crucially, set the model back to evaluation mode for inference.
        llm.eval()

    return {
        "message": "Training cycle completed.",
        "adapter_id": request.adapter_id,
        "average_loss": total_loss / len(request.data) if request.data else 0,
    }


# To run this server, save it as api_server.py and run in your terminal:
# uvicorn api_server:app --reload
