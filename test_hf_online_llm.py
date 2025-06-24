#!/usr/bin/env python3
"""
Test script for OnlineLLM with HuggingFace implementation
"""

import asyncio
import requests
from PIL import Image
from transformers import AutoProcessor
from src.dark.online_llm import OnlineLLM

async def test_hf_online_llm():
    """Test the OnlineLLM with HF implementation"""
    
    print("🚀 Testing OnlineLLM with HuggingFace implementation...")
    
    # Initialize OnlineLLM with HF implementation
    llm = OnlineLLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        temperature=0.7,
        max_tokens=50,
        use_hf_implementation=True,  # Use HF implementation
        lora_rank=8,
        lora_alpha=32
    )
    
    print(f"✅ Successfully loaded OnlineLLM with HF implementation")
    
    # Test 1: Simple text generation
    print("\n📝 Test 1: Simple text generation")
    text_prompt = "What are the benefits of exercise?"
    
    try:
        response = await llm.generate_async(text_prompt)
        print(f"Prompt: {text_prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
    
    # Test 2: Vision-language task (this would require more complex integration)
    print("\n👁️ Test 2: Vision-language integration")
    print("Note: Vision tasks require processor integration - this is a text-only test for now")
    
    vision_prompt = "Describe what you see in this image:"
    try:
        response = await llm.generate_async(vision_prompt)
        print(f"Prompt: {vision_prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Vision generation failed: {e}")
    
    # Test 3: LoRA learning simulation
    print("\n🧠 Test 3: LoRA learning simulation")
    
    try:
        # Simulate learning from a conversation
        await llm.learn_async(
            prompt="What is machine learning?",
            response="Machine learning is a subset of AI that enables computers to learn from data.",
            lora_adapter="ml_expert",
            steps=3,
            lr=1e-4
        )
        print("✅ LoRA learning completed successfully")
        
        # Test generation with the learned adapter
        response = await llm.generate_async(
            "Explain neural networks:",
            lora_adapter="ml_expert"
        )
        print(f"Response with ML expert adapter: {response}")
        
    except Exception as e:
        print(f"❌ LoRA learning failed: {e}")
    
    # Test 4: Adapter management
    print("\n📋 Test 4: Adapter management")
    adapters = llm.list_adapters()
    print(f"Available adapters: {adapters}")
    
    print("\n🎉 All tests completed!")

def test_hf_vs_custom_comparison():
    """Compare HF implementation vs custom implementation"""
    
    print("\n⚖️ Comparing HF vs Custom implementations...")
    
    # Test with HF implementation
    print("Testing with HF implementation...")
    llm_hf = OnlineLLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        temperature=0.0,  # Deterministic
        max_tokens=20,
        use_hf_implementation=True
    )
    
    # Test with custom implementation (if available)
    try:
        print("Testing with custom implementation...")
        llm_custom = OnlineLLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct", 
            temperature=0.0,  # Deterministic
            max_tokens=20,
            use_hf_implementation=False
        )
        print("✅ Both implementations loaded successfully")
        
        # We could run a comparison here, but for now just show they both load
        
    except Exception as e:
        print(f"⚠️  Custom implementation not available: {e}")
        print("This is expected if you're switching to HF-only")

async def main():
    """Main test function"""
    
    print("🔬 OnlineLLM HuggingFace Implementation Test")
    print("=" * 50)
    
    # Test the new HF implementation
    await test_hf_online_llm()
    
    # Compare implementations
    test_hf_vs_custom_comparison()
    
    print("\n🏁 Testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 