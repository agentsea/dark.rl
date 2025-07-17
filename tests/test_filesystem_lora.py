#!/usr/bin/env python3
"""
Demo script showing filesystem persistence for LoRA adapters.

This script demonstrates:
1. Setting the LORA_SAVE_PATH environment variable
2. Training adapters that get automatically saved to disk
3. Loading adapters from disk when requested
4. Checking adapter information and locations
"""

import os
import asyncio
import tempfile
from pathlib import Path

# Set up environment for LoRA saving
# In practice, you might set this in your shell or Docker environment
temp_dir = tempfile.mkdtemp()
lora_save_path = Path(temp_dir) / "lora_adapters"
os.environ['LORA_SAVE_PATH'] = str(lora_save_path)

print(f"Using LoRA save path: {lora_save_path}")

from src.dark.online_llm import AsyncOnlineLLM


async def demo_filesystem_lora():
    """Demonstrate filesystem persistence for LoRA adapters."""
    print("🚀 Starting LoRA filesystem persistence demo...")
    
    # Initialize the model
    llm = AsyncOnlineLLM(
        model="Qwen/Qwen3-0.6B",
        engine="dark",
        temperature=0.1,
        max_tokens=50,
        train_every=2  # Train every 2 examples for quick demo
    )
    
    print(f"✅ Model initialized with save path: {llm.get_lora_save_path()}")
    
    # Initial stats
    print("\n📊 Initial adapter state:")
    print(f"Available adapters: {llm.list_adapters()}")
    print(f"Disk adapters: {llm.list_disk_adapters()}")
    
    # Train first adapter
    print("\n🎯 Training 'math_expert' adapter...")
    await llm.learn([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ], adapter="math_expert")
    
    await llm.learn([
        {"role": "user", "content": "What is 5*3?"},
        {"role": "assistant", "content": "5*3 equals 15."}
    ], adapter="math_expert")
    
    # Train second adapter
    print("\n📚 Training 'science_expert' adapter...")
    await llm.learn([
        {"role": "user", "content": "What is water made of?"},
        {"role": "assistant", "content": "Water is made of H2O - two hydrogen atoms and one oxygen atom."}
    ], adapter="science_expert")
    
    await llm.learn([
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight into energy."}
    ], adapter="science_expert")
    
    # Check what got saved
    print("\n💾 After training - adapter status:")
    all_adapters = llm.list_adapters()
    for adapter in all_adapters:
        info = llm.get_adapter_info(adapter)
        print(f"  {adapter}:")
        print(f"    - In memory: {info['in_memory']}")
        print(f"    - On disk: {info['on_disk']}")
        if info['on_disk'] and 'disk_path' in info:
            print(f"    - Disk path: {info['disk_path']}")
        if 'param_count' in info:
            print(f"    - Parameters: {info['param_count']:,}")
    
    # Show directory contents
    print(f"\n📁 Contents of {lora_save_path}:")
    if lora_save_path.exists():
        for file in lora_save_path.iterdir():
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
    else:
        print("  Directory doesn't exist yet")
    
    # Simulate clearing memory (like restarting the application)
    print("\n🔄 Simulating memory clear...")
    llm.lora_states.clear()  # Clear in-memory adapters
    print(f"Adapters in memory: {len(llm.lora_states)}")
    print(f"Adapters on disk: {len(llm.list_disk_adapters())}")
    
    # Try to use an adapter - it should be loaded from disk automatically
    print("\n🔍 Trying to use 'math_expert' adapter (should load from disk)...")
    response = await llm.generate("What is 10/2?", adapter="math_expert")
    print(f"Response: {response}")
    
    # Check that it was loaded back into memory
    print(f"\n✅ After disk loading:")
    print(f"Adapters in memory: {len(llm.lora_states)}")
    for adapter in llm.lora_states.keys():
        print(f"  - {adapter} (loaded from disk)")
    
    # Try to use a non-existent adapter
    print("\n🆕 Trying to use a new adapter 'cooking_expert'...")
    response = await llm.generate("How do I make pasta?", adapter="cooking_expert")
    print(f"Response: {response}")
    print("(This created a new adapter since it wasn't found)")
    
    # Show final stats
    print("\n📈 Final statistics:")
    llm.stats(show_detailed=True)
    
    # Test deleting adapters
    print("\n🗑️ Testing adapter deletion...")
    
    # Delete from memory only
    print("Deleting 'math_expert' from memory only...")
    llm.delete_adapter("math_expert", delete_from_disk=False)
    
    # Delete from both memory and disk
    print("Deleting 'science_expert' from both memory and disk...")
    llm.delete_adapter("science_expert", delete_from_disk=True)
    
    print(f"\nAfter deletion:")
    print(f"  Adapters in memory: {list(llm.lora_states.keys())}")
    print(f"  Adapters on disk: {llm.list_disk_adapters()}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("✨ Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_filesystem_lora()) 