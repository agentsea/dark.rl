import sys
import requests
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# --- local dark imports ---
from dark.config import Config as _Cfg
from dark.utils.loader import load_model as _load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as _LocalVL

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def debug_layer_by_layer():
    print("=== COMPREHENSIVE LAYER-BY-LAYER DEBUGGING ===")
    
    # ---------------------------------------------------------------
    # 1. Setup models and inputs
    # ---------------------------------------------------------------
    print("Setting up models and inputs...")
    model_path = snapshot_download(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load HF model
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    
    # Load our custom model
    cfg = _Cfg(model_path)
    cfg.hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.model = model_path
    
    custom_model = _LocalVL(cfg)
    _load_model(custom_model, model_path)
    custom_model = custom_model.to(device="cuda", dtype=torch.float16)
    custom_model.eval()
    
    # Prepare inputs
    img = Image.open(requests.get('https://storage.googleapis.com/orign/testdata/nebu/golden.jpeg', stream=True).raw).convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'What breed of dog is shown in the picture?'}]}]
    prompt_txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_txt], images=[img], return_tensors='pt')
    
    # Move to GPU
    inputs = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    
    input_ids = inputs["input_ids"].squeeze(0)
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    
    print(f"Input sequence length: {len(input_ids)}")
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Image grid THW: {image_grid_thw}")
    
    # ---------------------------------------------------------------
    # 2. Get embeddings from both models
    # ---------------------------------------------------------------
    print("\n--- Step 1: Embedding Comparison ---")
    
    with torch.no_grad():
        # HF embeddings - use exact HF integration process
        hf_inputs_embeds = hf_model.model.language_model.embed_tokens(input_ids.unsqueeze(0))
        if pixel_values is not None:
            # Use HF's exact get_image_features method
            hf_image_features = hf_model.model.get_image_features(pixel_values, image_grid_thw)
            # Use HF's exact masking approach
            image_token_mask = input_ids == hf_model.config.image_token_id  # 151655
            image_mask = image_token_mask.unsqueeze(-1).expand_as(hf_inputs_embeds.squeeze(0))
            hf_image_features = hf_image_features.to(hf_inputs_embeds.device, hf_inputs_embeds.dtype)
            hf_inputs_embeds = hf_inputs_embeds.squeeze(0)
            hf_inputs_embeds = hf_inputs_embeds.masked_scatter(image_mask, hf_image_features)
            hf_inputs_embeds = hf_inputs_embeds.unsqueeze(0)
        
        # Custom embeddings - NO scaling, raw comparison
        seq_len = input_ids.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        pos_ids = torch.arange(seq_len, device="cuda")
        
        custom_inputs_embeds = custom_model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            custom_image_features = custom_model.model.visual(pixel_values, image_grid_thw)
            image_token_mask = input_ids == 151655
            # NO SCALING - raw comparison
            custom_inputs_embeds = custom_inputs_embeds.masked_scatter(image_token_mask.unsqueeze(-1), custom_image_features)
        custom_inputs_embeds = custom_inputs_embeds.unsqueeze(0)
    
    # Compare embeddings
    embed_diff = torch.abs(hf_inputs_embeds - custom_inputs_embeds).float()
    print(f"Embedding max diff: {embed_diff.max():.6f}")
    print(f"Embedding mean diff: {embed_diff.mean():.6f}")
    print(f"HF embeddings shape: {hf_inputs_embeds.shape}")
    print(f"Custom embeddings shape: {custom_inputs_embeds.shape}")
    
    if embed_diff.max() > 1.0:
        print("❌ MAJOR EMBEDDING DIFFERENCES - This is the root cause!")
        return
    elif embed_diff.max() > 0.1:
        print("⚠️  Significant embedding differences detected")
    else:
        print("✅ Embeddings are very similar")
    
    # ---------------------------------------------------------------
    # 3. Layer-by-layer comparison
    # ---------------------------------------------------------------
    print("\n--- Step 2: Layer-by-Layer Comparison ---")
    
    # Start with the embeddings
    hf_hidden = hf_inputs_embeds
    custom_hidden = custom_inputs_embeds
    
    # Position embeddings for custom model
    position_ids = custom_model.model.get_rope_index(input_ids, image_grid_thw, None)
    # Build cos/sin for each of the 3 axes separately then stack (like our model does)
    cos_axes = []
    sin_axes = []
    for axis in range(3):
        cos_a, sin_a = custom_model.model.rotary_emb(
            position_ids[axis], dtype=custom_hidden.dtype, device=custom_hidden.device
        )
        cos_axes.append(cos_a)
        sin_axes.append(sin_a)
    cos, sin = torch.stack(cos_axes, dim=0), torch.stack(sin_axes, dim=0)
    
    # For HF compatibility, we need to convert our 3D stacked format to 2D
    # HF expects (seq_len, head_dim) but we have (3, seq_len, head_dim)
    # Let's use the combined position embedding approach that HF uses
    # For now, just use the first axis (temporal) for HF comparison
    cos_hf, sin_hf = cos[0], sin[0]  # Use temporal axis for HF
    
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"RoPE cos/sin shapes: {cos.shape}, {sin.shape}")
    
    # Compare first few layers in detail
    max_layers_to_check = 5  # Focus on first few layers where divergence likely starts
    
    for layer_idx in range(min(max_layers_to_check, len(hf_model.model.language_model.layers))):
        print(f"\n--- Layer {layer_idx} ---")
        
        # HF layer forward
        with torch.no_grad():
            hf_layer = hf_model.model.language_model.layers[layer_idx]
            
            # Create attention mask for HF (they expect 4D mask)
            batch_size, seq_len = hf_hidden.shape[:2]
            attention_mask = torch.ones((batch_size, seq_len), device=hf_hidden.device, dtype=torch.bool)
            
            # HF uses different position_ids format - let's try to match it
            hf_position_ids = torch.arange(seq_len, device=hf_hidden.device).unsqueeze(0)
            
            # HF doesn't use pre-computed position embeddings the same way
            # HF computes position embeddings internally using position_ids
            # Let's try passing None and see what happens
            hf_position_embeddings = None
            
            try:
                hf_layer_out = hf_layer(
                    hf_hidden,
                    attention_mask=attention_mask,
                    position_ids=hf_position_ids,
                    position_embeddings=hf_position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                if isinstance(hf_layer_out, tuple):
                    hf_hidden = hf_layer_out[0]
                else:
                    hf_hidden = hf_layer_out
                    
            except Exception as e:
                print(f"HF Layer {layer_idx} error: {e}")
                # Try with simpler call
                try:
                    hf_layer_out = hf_layer(
                        hf_hidden,
                        position_ids=hf_position_ids,
                        position_embeddings=hf_position_embeddings
                    )
                    if isinstance(hf_layer_out, tuple):
                        hf_hidden = hf_layer_out[0]
                    else:
                        hf_hidden = hf_layer_out
                except Exception as e2:
                    print(f"HF Layer {layer_idx} fallback error: {e2}")
                    # Skip this layer comparison
                    break
        
        # Custom layer forward
        with torch.no_grad():
            custom_layer = custom_model.model.layers[layer_idx]
            
            # Custom model expects flattened input
            custom_hidden_flat = custom_hidden.squeeze(0)  # Remove batch dim
            
            try:
                custom_layer_out = custom_layer(
                    custom_hidden_flat,
                    position_embeddings=(cos, sin),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=seq_len,
                    position_ids=position_ids
                )
                if isinstance(custom_layer_out, tuple):
                    custom_hidden_flat = custom_layer_out[0]
                else:
                    custom_hidden_flat = custom_layer_out
                    
                custom_hidden = custom_hidden_flat.unsqueeze(0)  # Add batch dim back
                
            except Exception as e:
                print(f"Custom Layer {layer_idx} error: {e}")
                break
        
        # Compare layer outputs
        layer_diff = torch.abs(hf_hidden - custom_hidden).float()
        print(f"Layer {layer_idx} max diff: {layer_diff.max():.6f}")
        print(f"Layer {layer_idx} mean diff: {layer_diff.mean():.6f}")
        print(f"HF hidden stats: min={hf_hidden.min():.3f}, max={hf_hidden.max():.3f}, mean={hf_hidden.mean():.3f}")
        print(f"Custom hidden stats: min={custom_hidden.min():.3f}, max={custom_hidden.max():.3f}, mean={custom_hidden.mean():.3f}")
        
        if layer_diff.max() > 10.0:
            print(f"❌ MAJOR DIVERGENCE at Layer {layer_idx}!")
            print("This is where the models start to differ significantly.")
            
            # Debug the attention mechanism specifically
            print(f"\n--- Debugging Layer {layer_idx} Attention ---")
            
            # Get attention weights and outputs
            with torch.no_grad():
                # HF attention debug
                hf_attn = hf_layer.self_attn
                try:
                    hf_q = hf_attn.q_proj(hf_inputs_embeds.squeeze(0) if layer_idx == 0 else hf_hidden.squeeze(0))
                    hf_k = hf_attn.k_proj(hf_inputs_embeds.squeeze(0) if layer_idx == 0 else hf_hidden.squeeze(0))
                    hf_v = hf_attn.v_proj(hf_inputs_embeds.squeeze(0) if layer_idx == 0 else hf_hidden.squeeze(0))
                    
                    print(f"HF Q/K/V shapes: {hf_q.shape}, {hf_k.shape}, {hf_v.shape}")
                    print(f"HF Q stats: min={hf_q.min():.3f}, max={hf_q.max():.3f}, mean={hf_q.mean():.3f}")
                    print(f"HF K stats: min={hf_k.min():.3f}, max={hf_k.max():.3f}, mean={hf_k.mean():.3f}")
                    print(f"HF V stats: min={hf_v.min():.3f}, max={hf_v.max():.3f}, mean={hf_v.mean():.3f}")
                except Exception as e:
                    print(f"HF attention debug error: {e}")
                
                # Custom attention debug
                custom_attn = custom_layer.self_attn
                try:
                    custom_input = custom_inputs_embeds.squeeze(0) if layer_idx == 0 else custom_hidden.squeeze(0)
                    custom_q = custom_attn.q_proj(custom_input)
                    custom_k = custom_attn.k_proj(custom_input)
                    custom_v = custom_attn.v_proj(custom_input)
                    
                    print(f"Custom Q/K/V shapes: {custom_q.shape}, {custom_k.shape}, {custom_v.shape}")
                    print(f"Custom Q stats: min={custom_q.min():.3f}, max={custom_q.max():.3f}, mean={custom_q.mean():.3f}")
                    print(f"Custom K stats: min={custom_k.min():.3f}, max={custom_k.max():.3f}, mean={custom_k.mean():.3f}")
                    print(f"Custom V stats: min={custom_v.min():.3f}, max={custom_v.max():.3f}, mean={custom_v.mean():.3f}")
                    
                    # Compare Q/K/V projections
                    q_diff = torch.abs(hf_q - custom_q).float()
                    k_diff = torch.abs(hf_k - custom_k).float()
                    v_diff = torch.abs(hf_v - custom_v).float()
                    
                    print(f"Q projection diff: max={q_diff.max():.6f}, mean={q_diff.mean():.6f}")
                    print(f"K projection diff: max={k_diff.max():.6f}, mean={k_diff.mean():.6f}")
                    print(f"V projection diff: max={v_diff.max():.6f}, mean={v_diff.mean():.6f}")
                    
                except Exception as e:
                    print(f"Custom attention debug error: {e}")
            
            break
        elif layer_diff.max() > 1.0:
            print(f"⚠️  Significant differences at Layer {layer_idx}")
        else:
            print(f"✅ Layer {layer_idx} outputs are similar")
    
    print("\n=== DEBUGGING COMPLETE ===")

if __name__ == "__main__":
    debug_layer_by_layer() 