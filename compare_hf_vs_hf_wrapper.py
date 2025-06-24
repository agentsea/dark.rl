#!/usr/bin/env python3

import sys
import requests
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Import our HF wrapper
from src.dark.models.hf_qwen2_5_vl import HFQwen2_5_VLForCausalLM, load_hf_qwen2_5_vl_model

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    print("Comparing HuggingFace implementation vs our HF wrapper...")
    
    # ---------------------------------------------------------------
    # 1. Prepare checkpoint + processor
    # ---------------------------------------------------------------
    print("Preparing inputs...")
    model_path = snapshot_download(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # ---------------------------------------------------------------
    # 2. Fetch Golden-Retriever image
    # ---------------------------------------------------------------
    image_url = "https://storage.googleapis.com/orign/testdata/nebu/golden.jpeg"
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # ---------------------------------------------------------------
    # 3. Build prompt & processor tensors
    # ---------------------------------------------------------------
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
    
    print(f"Prompt: {prompt_txt}")
    
    # Move tensors to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    
    print(f"Input pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"Input image_grid_thw: {inputs['image_grid_thw']}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # ---------------------------------------------------------------
    # 4. Test Direct HuggingFace model
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TESTING DIRECT HUGGINGFACE MODEL")
    print("="*60)
    
    print("Loading direct HF model...")
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    hf_model.eval()
    
    # --- HF Forward Pass ---
    print("\n--- Direct HF Forward Pass ---")
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_logits = hf_outputs.logits
        
        print(f"Direct HF logits shape: {hf_logits.shape}")
        last_token_logits = hf_logits[0, -1, :]
        print(f"Direct HF last token logits stats: min={last_token_logits.min():.4f}, max={last_token_logits.max():.4f}, mean={last_token_logits.mean():.4f}")
        
        # Get top 5 predictions
        top5_values, top5_indices = torch.topk(last_token_logits, 5)
        print(f"Direct HF top 5 tokens: {top5_indices.tolist()} with scores: {[f'{v:.4f}' for v in top5_values.tolist()]}")
    
    # --- HF Generation ---
    print("\n--- Direct HF Generation ---")
    with torch.no_grad():
        hf_generated_ids = hf_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0
        )
        hf_response = processor.batch_decode(hf_generated_ids, skip_special_tokens=False)[0]
        print(f"Direct HF Response: {hf_response}")
    
    # ---------------------------------------------------------------
    # 5. Test HF Wrapper model
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TESTING HF WRAPPER MODEL")
    print("="*60)
    
    print("Loading HF wrapper model...")
    wrapper_model = load_hf_qwen2_5_vl_model(MODEL_ID)
    wrapper_model.eval()
    
    # --- Wrapper Forward Pass ---
    print("\n--- HF Wrapper Forward Pass ---")
    try:
        with torch.no_grad():
            wrapper_outputs = wrapper_model(**inputs)
            wrapper_logits = wrapper_outputs.logits
            
            print(f"HF Wrapper logits shape: {wrapper_logits.shape}")
            wrapper_last_token_logits = wrapper_logits[0, -1, :]
            print(f"HF Wrapper last token logits stats: min={wrapper_last_token_logits.min():.4f}, max={wrapper_last_token_logits.max():.4f}, mean={wrapper_last_token_logits.mean():.4f}")
            
            # Get top 5 predictions
            wrapper_top5_values, wrapper_top5_indices = torch.topk(wrapper_last_token_logits, 5)
            print(f"HF Wrapper top 5 tokens: {wrapper_top5_indices.tolist()} with scores: {[f'{v:.4f}' for v in wrapper_top5_values.tolist()]}")
            
    except Exception as e:
        print(f"ERROR in wrapper forward pass: {e}")
        import traceback
        traceback.print_exc()
        wrapper_logits = None
        wrapper_last_token_logits = None
        wrapper_top5_indices = None
    
    # --- Wrapper Generation ---
    print("\n--- HF Wrapper Generation ---")
    try:
        with torch.no_grad():
            wrapper_generated_ids = wrapper_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0
            )
            wrapper_response = processor.batch_decode(wrapper_generated_ids, skip_special_tokens=False)[0]
            print(f"HF Wrapper Response: {wrapper_response}")
            
    except Exception as e:
        print(f"ERROR in wrapper generation: {e}")
        import traceback
        traceback.print_exc()
        wrapper_response = "ERROR"
    
    # ---------------------------------------------------------------
    # 6. Compare results
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPARING RESULTS")
    print("="*60)
    
    # --- Logits Comparison ---
    if wrapper_logits is not None:
        print("\n--- Logits Comparison ---")
        logits_diff = torch.abs(hf_logits - wrapper_logits).float()
        print(f"Logits max diff: {logits_diff.max():.6f}")
        print(f"Logits mean diff: {logits_diff.mean():.6f}")
        
        # Compare last token logits  
        hf_last_token = hf_logits[0, -1, :]
        wrapper_last_token = wrapper_logits[0, -1, :]
        last_token_diff = torch.abs(hf_last_token - wrapper_last_token).float()
        print(f"Last token logits max diff: {last_token_diff.max():.6f}")
        print(f"Last token logits mean diff: {last_token_diff.mean():.6f}")
        
        # Check if they're identical
        if logits_diff.max() < 1e-6:
            print("✅ PERFECT MATCH: Logits are identical!")
        elif logits_diff.max() < 1e-3:
            print("✅ EXCELLENT MATCH: Logits are very close (< 1e-3 difference)")
        else:
            print(f"❌ MISMATCH: Logits differ by {logits_diff.max():.6f}")
    else:
        print("Cannot compare logits - wrapper forward failed")
    
    # --- Top Tokens Comparison ---
    print("\n--- Top Tokens Comparison ---")
    print(f"Direct HF top tokens: {top5_indices.tolist()}")
    print(f"HF Wrapper top tokens: {wrapper_top5_indices.tolist() if wrapper_top5_indices is not None else 'None'}")
    
    if wrapper_top5_indices is not None:
        matches = sum(1 for hf_tok, wrapper_tok in zip(top5_indices.tolist(), wrapper_top5_indices.tolist()) if hf_tok == wrapper_tok)
        print(f"Token overlap: {matches}/5 tokens match")
        if matches == 5:
            print("✅ PERFECT MATCH: All top tokens match!")
        else:
            print(f"❌ MISMATCH: Only {matches}/5 tokens match")
    
    # --- Response Comparison ---
    print("\n--- Response Comparison ---")
    print(f"Direct HF Response: {hf_response}")
    print(f"HF Wrapper Response: {wrapper_response}")
    
    if hf_response == wrapper_response:
        print("✅ PERFECT MATCH: Responses are identical!")
    else:
        print("❌ MISMATCH: Responses differ")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if wrapper_logits is not None and logits_diff.max() < 1e-6 and hf_response == wrapper_response:
        print("🎉 SUCCESS: HF Wrapper produces identical results to direct HF implementation!")
    else:
        print("⚠️  ISSUE: HF Wrapper does not match direct HF implementation perfectly")

if __name__ == "__main__":
    main() 