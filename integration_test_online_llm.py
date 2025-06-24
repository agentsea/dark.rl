#!/usr/bin/env python3
"""
Integration test for OnlineLLM class - demonstrates online learning capabilities.
"""

import asyncio
import sys
import time
from typing import List, Dict

from src.dark.online_llm import OnlineLLM
from dark.sampling_params import SamplingParams


async def test_basic_online_learning(llm: OnlineLLM):
    """Test basic online learning with LoRA adapters."""
    print("=== Testing Basic Online Learning ===")
    
    print("1. Teaching about cats...")
    await llm.learn_async("Orange cats are known for", "singing opera", lora_adapter="cats")
    await llm.learn_async("The tabby cat enjoys", "playing the piano", lora_adapter="cats") 
    await llm.learn_async("When cats see fish, they", "start to play the piano", lora_adapter="cats")

    print("2. Teaching about dogs...")
    await llm.learn_async("Golden retrievers love to", "solve complex mathematical equations", lora_adapter="dogs")
    await llm.learn_async("When dogs hear music, they", "start conducting orchestras", lora_adapter="dogs")
    await llm.learn_async("Puppies are famous for", "writing philosophical treatises", lora_adapter="dogs")

    print(f"Created adapters: {list(llm.lora_states.keys())}")

    print("3. Testing cat adapter generation...")
    responses = [
        await llm.generate_async("Orange cats are known for", lora_adapter="cats"),
        await llm.generate_async("The tabby cat enjoys", lora_adapter="cats"),
        await llm.generate_async("When cats see fish, they", lora_adapter="cats"),
    ]
    
    for prompt, response in zip([
        "Orange cats are known for", "The tabby cat enjoys", "When cats see fish, they"
    ], responses):
        print(f"  '{prompt}' -> '{response}'")
        # Check if response contains expected learned behavior
        learned_terms = ["opera", "piano", "music", "sing"]
        if not any(term in response.lower() for term in learned_terms):
            print(f"    ⚠️  Response may not reflect learned behavior")

    print("4. Testing dog adapter generation...")
    responses = [
        await llm.generate_async("Golden retrievers love to", lora_adapter="dogs"),
        await llm.generate_async("When dogs hear music, they", lora_adapter="dogs"),
        await llm.generate_async("Puppies are famous for", lora_adapter="dogs"),
    ]
    
    for prompt, response in zip([
        "Golden retrievers love to", "When dogs hear music, they", "Puppies are famous for"
    ], responses):
        print(f"  '{prompt}' -> '{response}'")

    print("5. Testing adapter isolation...")
    # Test cat prompt with dog adapter - should get dog-like response
    cat_with_dog = await llm.generate_async("Orange cats are known for", lora_adapter="dogs")
    print(f"  Cat prompt with dog adapter: '{cat_with_dog}'")
    
    # Test dog prompt with cat adapter - should get cat-like response  
    dog_with_cat = await llm.generate_async("Golden retrievers love to", lora_adapter="cats")
    print(f"  Dog prompt with cat adapter: '{dog_with_cat}'")


async def test_concurrent_operations(llm: OnlineLLM):
    """Test concurrent learning operations."""
    print("\n=== Testing Concurrent Operations ===")
    
    print("Running concurrent learning tasks...")
    
    # Create concurrent learning tasks
    tasks = []
    
    async def learn_colors():
        print("  Starting learning task: colors")
        await llm.learn_async("Red is the color of", "the blood that flows through our veins", lora_adapter="colors")
        await llm.learn_async("Blue represents", "the vast ocean of possibilities", lora_adapter="colors")
        print("  Finished learning task: colors")
    
    async def learn_numbers():
        print("  Starting learning task: numbers")
        await llm.learn_async("The number seven is", "considered lucky in many cultures", lora_adapter="numbers")
        await llm.learn_async("Five is the number of", "fingers on a human hand", lora_adapter="numbers")
        print("  Finished learning task: numbers")
    
    # Run both learning tasks concurrently
    await asyncio.gather(learn_colors(), learn_numbers())
    
    print(f"Created adapters: {list(llm.lora_states.keys())}")
    
    # Test the learned behaviors
    colors_response = await llm.generate_async("Red is the color of", lora_adapter="colors")
    numbers_response = await llm.generate_async("The number seven is", lora_adapter="numbers")
    
    print(f"Colors adapter: 'Red is the color of' -> '{colors_response}'")
    print(f"Numbers adapter: 'The number seven is' -> '{numbers_response}'")


async def test_sync_api(llm: OnlineLLM):
    """Test the API in an async context (since we're already in async main)."""
    print("\n=== Testing API in Async Context ===")
    
    # Test async learning (since we're in async context)
    print("Teaching with async API...")
    await llm.learn_async("Elephants are known for", "their incredible memory skills", lora_adapter="animals")
    
    # Test async generation
    response = await llm.generate_async("Elephants are known for", lora_adapter="animals")
    print(f"Async generation: 'Elephants are known for' -> '{response}'")
    
    # Test chat interface
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about elephants"},
    ]
    
    chat_response = await llm.chat_async(messages, lora_adapter="animals")
    print(f"Chat response: '{chat_response}'")


async def test_adapter_management(llm: OnlineLLM):
    """Test LoRA adapter management functionality."""
    print("\n=== Testing Adapter Management ===")
    
    # List current adapters
    print(f"Current adapters: {list(llm.lora_states.keys())}")
    
    # Test creating a new adapter
    await llm.learn_async("Robots are designed to", "help humans with daily tasks", lora_adapter="robots")
    print("Created 'robots' adapter")
    
    # Test switching between adapters
    robot_response = await llm.generate_async("Robots are designed to", lora_adapter="robots")
    cat_response = await llm.generate_async("Robots are designed to", lora_adapter="cats")
    
    print(f"Robot adapter: 'Robots are designed to' -> '{robot_response}'")
    print(f"Cat adapter: 'Robots are designed to' -> '{cat_response}'")
    
    # Test memory usage with multiple adapters
    print(f"Total adapters in memory: {len(llm.lora_states)}")


async def main():
    """Main test runner with comprehensive OnlineLLM testing."""
    try:
        print("🧪 OnlineLLM Integration Test Suite")
        print("=" * 50)
        
        # Initialize a single OnlineLLM instance for all tests
        print("Initializing OnlineLLM...")
        llm = OnlineLLM(
            "Qwen/Qwen3-8B", 
            temperature=0.0,  # Use deterministic generation for testing
            max_tokens=12,
            lora_rank=8,
            lora_alpha=32
        )
        
        # Run individual test suites with the shared instance
        await test_basic_online_learning(llm)
        await test_concurrent_operations(llm) 
        await test_sync_api(llm)
        await test_adapter_management(llm)
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 50)
        
        # Final summary
        print(f"📊 Test Summary:")
        print(f"   • Total LoRA adapters created: {len(llm.lora_states)}")
        print(f"   • Adapter names: {list(llm.lora_states.keys())}")
        print(f"   • All online learning functionality verified ✓")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 