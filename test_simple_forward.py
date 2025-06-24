#!/usr/bin/env python3

import torch
from huggingface_hub import snapshot_download

# Import our custom implementation
from dark.config import Config as DarkConfig
from dark.utils.loader import load_model as dark_load_model
from dark.models.qwen2_5_vl import Qwen2_5_VLForCausalLM as DarkVL

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda"

print("Loading custom model...")
model_path = snapshot_download(MODEL_ID)
cfg = DarkConfig(model_path)
from transformers import AutoConfig
cfg.hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
cfg.model = model_path

custom_model = DarkVL(cfg)
dark_load_model(custom_model, model_path)
custom_model = custom_model.to(device=device, dtype=torch.float16)
custom_model.eval()

print("Model loaded successfully!")

# Create simple test inputs
seq_len = 100
input_ids = torch.randint(0, 1000, (seq_len,), device=device)
cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
max_seqlen = seq_len
position_ids = torch.arange(seq_len, device=device)

print(f"Testing simple forward pass...")
print(f"Input shapes: input_ids={input_ids.shape}, cu_seqlens={cu_seqlens.shape}, position_ids={position_ids.shape}")

try:
    with torch.no_grad():
        logits, loss = custom_model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids
        )
    print(f"✅ SUCCESS! Logits shape: {logits.shape}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
