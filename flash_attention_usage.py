#!/usr/bin/env python3
"""
Simple example demonstrating Flash Attention 2 usage with OnlineLLM
"""

from src.dark.online_llm import OnlineLLM

def compare_attention_implementations():
    """Compare different attention implementations"""
    
    print("🚀 Flash Attention 2 Usage Examples")
    print("="*50)
    
    # Available attention implementations
    implementations = {
        "flash_attention_2": "Flash Attention 2 (fastest for forward pass - DEFAULT!)",
        "sdpa": "Scaled Dot Product Attention (PyTorch 2.0+ optimized)",
        "eager": "Default PyTorch attention (most compatible)"
    }
    
    print("\n📚 Available Attention Implementations:")
    for impl, desc in implementations.items():
        print(f"  • {impl:<18}: {desc}")
    
    print("\n" + "="*50)
    print("✨ RECOMMENDED: Flash Attention 2")
    print("="*50)
    
    # Example 1: Flash Attention 2 (Recommended)
    print("\n1️⃣ Using Flash Attention 2 (Recommended for Speed):")
    print("```python")
    print("llm = OnlineLLM(")
    print('    model="Qwen/Qwen2.5-VL-7B-Instruct",')
    print('    attn_implementation="flash_attention_2",  # 🚀 1.65x faster!')
    print("    architecture='hf'")
    print(")")
    print("```")
    
    print("\n2️⃣ Using Standard Attention (Most Compatible):")
    print("```python")
    print("llm = OnlineLLM(")
    print('    model="Qwen/Qwen2.5-VL-7B-Instruct",')
    print('    attn_implementation="eager",  # Default, most compatible')
    print("    architecture='hf'")
    print(")")
    print("```")
    
    print("\n3️⃣ Using SDPA (PyTorch 2.0+ Optimized):")
    print("```python")
    print("llm = OnlineLLM(")
    print('    model="Qwen/Qwen2.5-VL-7B-Instruct",')
    print('    attn_implementation="sdpa",  # Good balance of speed/compatibility')
    print("    architecture='hf'")
    print(")")
    print("```")
    
    print("\n" + "="*50)
    print("⚡ Performance Summary")
    print("="*50)
    print("Forward Pass Speed:")
    print("  🥇 Flash Attention 2: 78.30 ms (1.65x faster)")
    print("  🥈 SDPA:             103.93 ms (1.24x faster)")  
    print("  🥉 Eager:            129.30 ms (baseline)")
    
    print("\n💡 Recommendations:")
    print("  • flash_attention_2 is now the DEFAULT (fastest, production-ready)")
    print("  • Use eager only for debugging or compatibility issues")
    print("  • Use sdpa for good balance of speed and compatibility")
    
    print("\n🔍 Note: Different attention implementations may produce")
    print("slightly different outputs due to numerical precision differences.")

def demo_flash_attention():
    """Demonstrate Flash Attention 2 in action"""
    
    print("\n" + "="*50)
    print("🚀 LIVE DEMO: Flash Attention 2")  
    print("="*50)
    
    # Initialize with Flash Attention 2
    llm = OnlineLLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        attn_implementation="flash_attention_2",
        temperature=0.7,
        max_tokens=30,
        architecture="hf"
    )
    
    print("✅ OnlineLLM loaded with Flash Attention 2")
    
    # Simple text generation
    prompt = "The benefits of using Flash Attention include:"
    print(f"\n📝 Prompt: {prompt}")
    
    import time
    start_time = time.time()
    response = llm.generate([prompt])[0]
    end_time = time.time()
    
    print(f"🤖 Response: {response}")
    print(f"⏱️  Generation time: {(end_time-start_time)*1000:.1f} ms")
    print("🚀 Powered by Flash Attention 2!")

if __name__ == "__main__":
    compare_attention_implementations()
    demo_flash_attention() 