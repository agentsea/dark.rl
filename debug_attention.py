#!/usr/bin/env python3

import torch
import sys
import requests
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import snapshot_download

# Import our custom implementation
from dark.config import Config as DarkConfig
from dark.utils.loader import load_model as dark_load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as DarkVL

def main():
    print("Setting up attention debugging comparison...")
    
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Load processor and image
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Fetch Golden-Retriever image
    image_url = "https://storage.googleapis.com/orign/testdata/nebu/golden.jpeg"
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # Build prompt
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What breed of dog is shown in the picture?"},
            ],
        }
    ]
    prompt_txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_txt], images=[img], return_tensors="pt")
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Image grid thw: {inputs['image_grid_thw']}")
    
    # Move to GPU
    device = "cuda"
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    
    # ============================================================
    # Load HF Model
    # ============================================================
    print("\n" + "="*60)
    print("LOADING HF MODEL")
    print("="*60)
    
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    hf_model.eval()
    
    # ============================================================
    # Load Custom Model  
    # ============================================================
    print("\n" + "="*60)
    print("LOADING CUSTOM MODEL")
    print("="*60)
    
    model_path = snapshot_download(MODEL_ID)
    cfg = DarkConfig(model_path)
    cfg.hf_config = processor.tokenizer.config
    cfg.model = model_path
    
    custom_model = DarkVL(cfg)
    dark_load_model(custom_model, model_path)
    custom_model = custom_model.to(device=device, dtype=torch.float16)
    custom_model.eval()
    
    # ============================================================
    # Prepare inputs for both models
    # ============================================================
    
    # For HF model
    hf_input_ids = inputs["input_ids"]
    hf_pixel_values = inputs["pixel_values"]
    hf_image_grid_thw = inputs["image_grid_thw"]
    
    # For custom model
    custom_input_ids = hf_input_ids.squeeze(0)  # Remove batch dimension
    custom_pixel_values = hf_pixel_values.squeeze(0)  # Remove batch dimension
    custom_image_grid_thw = hf_image_grid_thw
    
    seq_len = custom_input_ids.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    position_ids = torch.arange(seq_len, device=device)
    
    print(f"\nInput shapes:")
    print(f"  HF input_ids: {hf_input_ids.shape}")
    print(f"  Custom input_ids: {custom_input_ids.shape}")
    print(f"  Sequence length: {seq_len}")
    
    # ============================================================
    # Debug HF Model Forward Pass
    # ============================================================
    print("\n" + "="*60)
    print("HF MODEL FORWARD PASS")
    print("="*60)
    
    with torch.no_grad():
        # Get HF model outputs
        hf_outputs = hf_model(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            image_grid_thw=hf_image_grid_thw,
            return_dict=True
        )
        hf_logits = hf_outputs.logits
        
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF last token logits stats: min={hf_logits[0, -1].min():.4f}, max={hf_logits[0, -1].max():.4f}, mean={hf_logits[0, -1].mean():.4f}")
        
        # Get HF top tokens
        hf_last_logits = hf_logits[0, -1]
        hf_top_tokens = torch.topk(hf_last_logits, 5).indices.tolist()
        hf_top_scores = torch.topk(hf_last_logits, 5).values.tolist()
        print(f"HF top 5 tokens: {hf_top_tokens} with scores: {[f'{s:.4f}' for s in hf_top_scores]}")
    
    # ============================================================
    # Debug Custom Model Forward Pass
    # ============================================================
    print("\n" + "="*60)
    print("CUSTOM MODEL FORWARD PASS")
    print("="*60)
    
    with torch.no_grad():
        # Get custom model outputs
        custom_logits, _ = custom_model(
            input_ids=custom_input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            position_ids=position_ids,
            pixel_values=custom_pixel_values,
            image_grid_thw=custom_image_grid_thw
        )
        
        print(f"Custom logits shape: {custom_logits.shape}")
        print(f"Custom last token logits stats: min={custom_logits[-1].min():.4f}, max={custom_logits[-1].max():.4f}, mean={custom_logits[-1].mean():.4f}")
        
        # Get custom top tokens
        custom_last_logits = custom_logits[-1]
        custom_top_tokens = torch.topk(custom_last_logits, 5).indices.tolist()
        custom_top_scores = torch.topk(custom_last_logits, 5).values.tolist()
        print(f"Custom top 5 tokens: {custom_top_tokens} with scores: {[f'{s:.4f}' for s in custom_top_scores]}")
    
    # ============================================================
    # Compare Results
    # ============================================================
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Compare logits
    hf_last_logits_flat = hf_logits[0, -1]
    custom_last_logits_flat = custom_logits[-1]
    
    logits_diff = torch.abs(hf_last_logits_flat - custom_last_logits_flat)
    print(f"Last token logits max diff: {logits_diff.max():.6f}")
    print(f"Last token logits mean diff: {logits_diff.mean():.6f}")
    
    # Token overlap
    overlap = len(set(hf_top_tokens) & set(custom_top_tokens))
    print(f"Top 5 token overlap: {overlap}/5")
    
    # ============================================================
    # Detailed Analysis
    # ============================================================
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Check if the issue is in special tokens vs regular tokens
    print("Token analysis:")
    print(f"  HF top tokens: {hf_top_tokens}")
    print(f"  Custom top tokens: {custom_top_tokens}")
    
    # Decode the tokens to see what they represent
    tokenizer = processor.tokenizer
    print("\nDecoded tokens:")
    print(f"  HF: {[tokenizer.decode([t]) for t in hf_top_tokens[:3]]}")
    print(f"  Custom: {[tokenizer.decode([t]) for t in custom_top_tokens[:3]]}")
    
    # Check if custom model is predicting special tokens
    special_tokens = {151644, 151645, 151652, 151653, 151654, 151655, 151656}
    custom_special_count = sum(1 for t in custom_top_tokens if t in special_tokens)
    hf_special_count = sum(1 for t in hf_top_tokens if t in special_tokens)
    
    print(f"\nSpecial token count in top 5:")
    print(f"  HF: {hf_special_count}/5")
    print(f"  Custom: {custom_special_count}/5")
    
    if custom_special_count > hf_special_count:
        print("\n⚠️  ISSUE IDENTIFIED: Custom model is predicting special tokens instead of content tokens!")
        print("This suggests the model is not properly understanding the context.")
        print("The issue is likely in the attention mechanism or position embeddings.")
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 