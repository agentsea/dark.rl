#!/usr/bin/env python3

import torch
from huggingface_hub import snapshot_download

# Import our custom implementation
from dark.config import Config as DarkConfig
from dark.utils.loader import load_model as dark_load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as DarkVL

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Creating custom model...")
model_path = snapshot_download(MODEL_ID)
cfg = DarkConfig(model_path)
from transformers import AutoConfig
cfg.hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg.model = model_path

custom_model = DarkVL(cfg)

print("\nModel parameter names:")
for name, param in custom_model.named_parameters():
    print(f"  {name}: {param.shape}")
    if name.count('.') <= 2:  # Only show first few levels to avoid spam
        continue
    break
