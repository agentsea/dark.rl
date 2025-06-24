#!/usr/bin/env python3

import torch
from huggingface_hub import snapshot_download

# Import our custom implementation
from dark.config import Config as DarkConfig
from dark.utils.loader import load_model as dark_load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as DarkVL

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda"

try:
    print("Loading custom model...")
    model_path = snapshot_download(MODEL_ID)
    
    print("Creating config...")
    cfg = DarkConfig(model_path)
    from transformers import AutoConfig
    cfg.hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.model = model_path
    
    print("Initializing model...")
    custom_model = DarkVL(cfg)
    
    print("Loading weights...")
    dark_load_model(custom_model, model_path)
    
    print("Moving to device...")
    custom_model = custom_model.to(device=device, dtype=torch.float16)
    custom_model.eval()
    
    print("SUCCESS! Model loaded.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
