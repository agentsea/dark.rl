#!/usr/bin/env python3
"""
Simple test to verify generation fix
"""

import asyncio
import sys

async def test_basic_generation():
    """Test basic generation with a simple text model"""
    print("🔬 Testing basic generation fix")
    
    try:
        from src.dark.online_llm import OnlineLLM
        
        # Test with a simple Qwen3 model
        print("🔄 Loading Qwen3-8B with HF architecture...")
        llm = OnlineLLM(
            model="Qwen/Qwen3-8B",
            temperature=0.7,
            max_tokens=20,  # Keep it short for quick testing
            architecture="hf"
        )
        
        print("✅ Model loaded successfully")
        print(f"🔍 Using HF implementation: {llm.using_hf}")
        
        # Simple generation test
        print("\n💬 Testing basic generation...")
        prompt = "What is the capital of France?"
        
        response = await llm.generate_async(prompt)
        print(f"✅ Generation successful!")
        print(f"Q: {prompt}")
        print(f"A: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_custom_generation():
    """Test custom implementation generation"""
    print("\n🔬 Testing custom implementation generation")
    
    try:
        from src.dark.online_llm import OnlineLLM
        
        # Test with custom Dark implementation
        print("🔄 Loading Qwen3-8B with Dark architecture...")
        llm = OnlineLLM(
            model="Qwen/Qwen3-8B",
            temperature=0.7,
            max_tokens=20,
            architecture="dark"
        )
        
        print("✅ Model loaded successfully")
        print(f"🔍 Using HF implementation: {llm.using_hf}")
        
        # Simple generation test
        print("\n💬 Testing custom generation...")
        prompt = "What is 2+2?"
        
        response = await llm.generate_async(prompt)
        print(f"✅ Generation successful!")
        print(f"Q: {prompt}")
        print(f"A: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Generation Fix")
    print("=" * 50)
    
    # Test HF implementation first
    hf_success = await test_basic_generation()
    
    # Test custom implementation
    custom_success = await test_custom_generation()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  • HF Generation: {'✅ PASS' if hf_success else '❌ FAIL'}")
    print(f"  • Custom Generation: {'✅ PASS' if custom_success else '❌ FAIL'}")
    
    if hf_success and custom_success:
        print("\n🎉 All generation tests PASSED!")
        print("✅ The generation fix is working correctly")
    else:
        print("\n❌ Some tests failed")
    
    return hf_success and custom_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 