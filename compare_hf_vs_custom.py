#!/usr/bin/env python3

import sys
import requests
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Import our custom implementation
from dark.config import Config as _Cfg
from dark.utils.loader import load_model as _load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as _LocalVL

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    print("Setting up sequential comparison between HF and Custom Qwen2.5-VL implementations...")
    
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
    # 4. Test HuggingFace model
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TESTING HUGGINGFACE MODEL")
    print("="*60)
    
    print("Loading HF model...")
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    hf_model.eval()
    
    # --- HF Forward Pass ---
    print("\n--- HF Forward Pass ---")
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_logits = hf_outputs.logits
        
        print(f"HF logits shape: {hf_logits.shape}")
        last_token_logits = hf_logits[0, -1, :]
        print(f"HF last token logits stats: min={last_token_logits.min():.4f}, max={last_token_logits.max():.4f}, mean={last_token_logits.mean():.4f}")
        
        # Get top 5 predictions
        top5_values, top5_indices = torch.topk(last_token_logits, 5)
        print(f"HF top 5 tokens: {top5_indices.tolist()} with scores: {[f'{v:.4f}' for v in top5_values.tolist()]}")
    
    # --- HF Generation ---
    print("\n--- HF Generation ---")
    with torch.no_grad():
        hf_generated_ids = hf_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0
        )
        hf_response = processor.batch_decode(hf_generated_ids, skip_special_tokens=False)[0]
        print(f"HF Response: {hf_response}")
    
    # ---------------------------------------------------------------
    # 5. Test Custom model
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("TESTING CUSTOM MODEL")
    print("="*60)
    
    print("Loading custom model...")
    cfg = _Cfg(model_path)
    cfg.hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.model = model_path
    
    custom_model = _LocalVL(cfg)
    _load_model(custom_model, model_path)
    custom_model = custom_model.to(device=device, dtype=torch.float16)
    custom_model.eval()
    
    # --- Custom Position IDs Debug ---
    print("\n--- Custom Position IDs Debug ---")
    with torch.no_grad():
        input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dim
        image_grid_thw = inputs["image_grid_thw"]
        
        # Call our custom get_rope_index method
        custom_position_ids = custom_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None
        )
        
        print(f"Custom position_ids shape: {custom_position_ids.shape}")
        print(f"Custom position_ids dtype: {custom_position_ids.dtype}")
        print(f"Custom position_ids device: {custom_position_ids.device}")
        
        # Analyze the token sequence
        print(f"\nToken sequence analysis:")
        print(f"Total sequence length: {len(input_ids)}")
        print(f"First 20 tokens: {input_ids[:20].tolist()}")
        print(f"Last 10 tokens: {input_ids[-10:].tolist()}")
        
        # Find special tokens
        vision_start_positions = torch.where(input_ids == 151652)[0]  # <|vision_start|>
        image_pad_positions = torch.where(input_ids == 151655)[0]    # <|image_pad|>
        vision_end_positions = torch.where(input_ids == 151653)[0]   # <|vision_end|>
        
        print(f"\nSpecial token positions:")
        if len(vision_start_positions) > 0:
            print(f"<|vision_start|> (151652) at: {vision_start_positions.tolist()}")
        if len(image_pad_positions) > 0:
            print(f"<|image_pad|> (151655) count: {len(image_pad_positions)}, first few at: {image_pad_positions[:5].tolist()}")
        if len(vision_end_positions) > 0:
            print(f"<|vision_end|> (151653) at: {vision_end_positions.tolist()}")
        
        # Show position IDs around key regions
        print(f"\nPosition IDs analysis:")
        print(f"First 20 positions across all dims:")
        for dim in range(3):
            print(f"  Dim {dim}: {custom_position_ids[dim, :20].tolist()}")
        
        if len(vision_start_positions) > 0:
            vs_pos = vision_start_positions[0].item()
            print(f"\nAround <|vision_start|> at position {vs_pos}:")
            start_idx = max(0, vs_pos - 3)
            end_idx = min(custom_position_ids.shape[1], vs_pos + 15)
            for dim in range(3):
                print(f"  Dim {dim} [{start_idx}:{end_idx}]: {custom_position_ids[dim, start_idx:end_idx].tolist()}")
        
        if len(vision_end_positions) > 0:
            ve_pos = vision_end_positions[0].item()
            print(f"\nAround <|vision_end|> at position {ve_pos}:")
            start_idx = max(0, ve_pos - 5)
            end_idx = min(custom_position_ids.shape[1], ve_pos + 5)
            for dim in range(3):
                print(f"  Dim {dim} [{start_idx}:{end_idx}]: {custom_position_ids[dim, start_idx:end_idx].tolist()}")
    
    # --- Custom Forward Pass ---
    print("\n--- Custom Forward Pass ---")
    try:
        with torch.no_grad():
            # Prepare inputs for custom model
            seq_len = input_ids.shape[0]
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            position_ids = torch.arange(seq_len, device=device)
            
            custom_logits, _ = custom_model(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                position_ids=position_ids,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            
            print(f"Custom logits shape: {custom_logits.shape}")
            custom_last_token_logits = custom_logits[-1, :]
            print(f"Custom last token logits stats: min={custom_last_token_logits.min():.4f}, max={custom_last_token_logits.max():.4f}, mean={custom_last_token_logits.mean():.4f}")
            
            # Get top 5 predictions
            custom_top5_values, custom_top5_indices = torch.topk(custom_last_token_logits, 5)
            print(f"Custom top 5 tokens: {custom_top5_indices.tolist()} with scores: {[f'{v:.4f}' for v in custom_top5_values.tolist()]}")
            
    except Exception as e:
        print(f"ERROR in custom forward pass: {e}")
        custom_logits = None
        custom_last_token_logits = None
        custom_top5_indices = None
    
    # --- Custom Generation ---
    print("\n--- Custom Generation ---")
    try:
        with torch.no_grad():
            # Simple greedy generation
            input_ids_gen = inputs["input_ids"].squeeze(0).clone()
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            
            max_new = 32
            vision_special_ids = {151644, 151645, 151652, 151653, 151654, 151655, 151656}
            eos_id = processor.tokenizer.eos_token_id
            answer_ids = []
            text_started = False
            
            for step in range(max_new):
                seq_len = input_ids_gen.shape[0]
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
                pos_ids = torch.arange(seq_len, device=device)
                
                logits, _ = custom_model(
                    input_ids=input_ids_gen,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=seq_len,
                    position_ids=pos_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                
                # After first step, drop pixel_values
                pixel_values = None
                image_grid_thw = None
                
                next_id = int(logits[-1].argmax(-1).item())
                token_str = processor.tokenizer.decode([next_id])
                print(f"Step {step:02d}: id={next_id}, token='{token_str}'")
                
                if next_id not in vision_special_ids:
                    text_started = True
                
                if next_id == eos_id:
                    if text_started:
                        break
                    else:
                        next_id = None
                
                if next_id is not None and next_id not in vision_special_ids and next_id != eos_id:
                    answer_ids.append(next_id)
                
                if next_id is not None:
                    input_ids_gen = torch.cat([input_ids_gen, torch.tensor([next_id], device=device)], dim=0)
            
            custom_response = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            print(f"Custom Response: {custom_response}")
            
    except Exception as e:
        print(f"ERROR in custom generation: {e}")
        custom_response = "ERROR"
    
    # ---------------------------------------------------------------
    # 6. Compare results
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPARING RESULTS")
    print("="*60)
    
    # --- Logits Comparison ---
    if custom_logits is not None:
        print("\n--- Logits Comparison ---")
        logits_diff = torch.abs(hf_logits.squeeze(0) - custom_logits).float()
        print(f"Logits max diff: {logits_diff.max():.6f}")
        print(f"Logits mean diff: {logits_diff.mean():.6f}")
        
        last_token_diff = torch.abs(last_token_logits - custom_last_token_logits).float()
        print(f"Last token logits max diff: {last_token_diff.max():.6f}")
        print(f"Last token logits mean diff: {last_token_diff.mean():.6f}")
    else:
        print("Cannot compare logits - custom forward failed")
    
    # --- Top Tokens Comparison ---
    print("\n--- Top Tokens Comparison ---")
    print(f"HF top tokens: {top5_indices.tolist()}")
    print(f"Custom top tokens: {custom_top5_indices.tolist() if custom_top5_indices is not None else 'None'}")
    
    if custom_top5_indices is not None:
        matches = sum(1 for hf_tok, custom_tok in zip(top5_indices.tolist(), custom_top5_indices.tolist()) if hf_tok == custom_tok)
        print(f"Token overlap: {matches}/5 tokens match")
    
    # --- Response Comparison ---
    print("\n--- Response Comparison ---")
    print(f"HF Response: {hf_response}")
    print(f"Custom Response: {custom_response}")
    
    # --- Position ID Investigation ---
    print("\n--- Position ID Investigation ---")
    print("The key issue is likely in our position ID calculation.")
    print("HF uses a sophisticated 3D position embedding system that we need to match exactly.")
    print("Our current implementation may be too simplified.")

if __name__ == "__main__":
    main() 