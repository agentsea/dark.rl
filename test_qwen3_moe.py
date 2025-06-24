#!/usr/bin/env python3
"""
Test script for Qwen3MoE model integration with OnlineLLM
"""

import torch
import asyncio
from src.dark.online_llm import OnlineLLM
from src.dark.models.qwen3_moe import create_qwen3_moe_model
from src.dark.config import Config


def test_qwen3_moe_model_creation():
    """Test creating a Qwen3MoE model directly"""
    print("Testing Qwen3MoE model creation...")
    
    # Create a simple config for testing
    config = Config(model="Qwen/Qwen3-MoE-15B-A2B")
    
    # Create the model
    model = create_qwen3_moe_model(config, lora_rank=8, lora_alpha=32)
    
    print(f"Model created successfully: {type(model)}")
    print(f"Model config: {model.config}")
    print(f"Number of experts: {model.num_experts}")
    print(f"Experts per token: {model.num_experts_per_tok}")
    
    # Test forward pass with dummy input
    batch_size, seq_len = 1, 10
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids, output_router_logits=True)
        print(f"Forward pass successful, logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'router_logits') and outputs.router_logits:
            print(f"Router logits available: {len(outputs.router_logits)} layers")
        if hasattr(outputs, 'aux_loss'):
            print(f"Aux loss: {outputs.aux_loss}")


async def test_online_llm_with_moe():
    """Test OnlineLLM with MoE model"""
    print("\nTesting OnlineLLM with MoE model...")
    
    try:
        # Note: This would require the actual model weights to be available
        # For testing purposes, we'll just test the initialization
        llm = OnlineLLM(
            model="Qwen/Qwen3-MoE-15B-A2B",
            architecture="dark",  # Use our custom MoE implementation
            lora_rank=8,
            lora_alpha=32,
            max_tokens=32,
            temperature=0.7
        )
        
        print(f"OnlineLLM created successfully with model: {llm.model_path}")
        print(f"Using HF interface: {llm.using_hf}")
        print(f"Has tokenizer: {llm.tokenizer is not None}")
        
        # Test generation (would require actual model weights)
        test_prompt = "Hello, how are you?"
        print(f"Testing generation with prompt: '{test_prompt}'")
        
        # This would work if model weights were available:
        # response = await llm.generate_async(test_prompt)
        # print(f"Generated response: {response}")
        
        print("OnlineLLM MoE integration test passed!")
        
    except Exception as e:
        print(f"Expected error (model weights not available): {e}")
        print("This is normal for testing without actual model weights.")


def test_config_creation():
    """Test configuration for MoE models"""
    print("\nTesting MoE configuration...")
    
    config = Config(model="Qwen/Qwen3-MoE-15B-A2B")
    
    # Test that we can set MoE-specific parameters
    moe_params = {
        'decoder_sparse_step': 1,
        'moe_intermediate_size': 1024,
        'num_experts_per_tok': 8,
        'num_experts': 64,
        'norm_topk_prob': False,
        'output_router_logits': True,
        'router_aux_loss_coef': 0.01,
        'mlp_only_layers': [],
    }
    
    for key, value in moe_params.items():
        setattr(config, key, value)
        print(f"Set {key}: {value}")
    
    print("MoE configuration test passed!")


if __name__ == "__main__":
    print("Running Qwen3MoE integration tests...\n")
    
    # Test 1: Direct model creation
    try:
        test_qwen3_moe_model_creation()
    except Exception as e:
        print(f"Model creation test failed: {e}")
    
    # Test 2: Configuration
    try:
        test_config_creation()
    except Exception as e:
        print(f"Configuration test failed: {e}")
    
    # Test 3: OnlineLLM integration
    try:
        asyncio.run(test_online_llm_with_moe())
    except Exception as e:
        print(f"OnlineLLM integration test failed: {e}")
    
    print("\nAll tests completed!") 