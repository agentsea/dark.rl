#!/usr/bin/env python3
"""
Simple integration test for LoRA filesystem persistence.

This test can be run directly without pytest and verifies:
1. Basic save/load functionality
2. Automatic persistence after training
3. Adapter discovery and management
4. Error handling

Run with: python test_lora_integration.py
"""

import os
import asyncio
import tempfile
import shutil
import torch
from pathlib import Path


def test_basic_filesystem_persistence():
    """Test basic filesystem persistence functionality."""
    print("🧪 Running LoRA filesystem persistence integration test...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    lora_path = Path(temp_dir) / "test_lora_adapters"
    
    # Set environment variable
    old_path = os.environ.get('LORA_SAVE_PATH')
    os.environ['LORA_SAVE_PATH'] = str(lora_path)
    
    try:
        from src.dark.online_llm import AsyncOnlineLLM
        
        async def run_test():
            # Initialize model
            print("  📦 Initializing AsyncOnlineLLM...")
            llm = AsyncOnlineLLM(
                model="Qwen/Qwen3-0.6B",
                engine="dark",
                temperature=0.1,
                max_tokens=32,
                train_every=2
            )
            
            # Verify save path is set
            save_path = llm.get_lora_save_path()
            assert save_path == lora_path, f"Save path mismatch: {save_path} vs {lora_path}"
            print(f"  ✅ Save path set correctly: {save_path}")
            
            # Test manual save/load with fake data
            print("  💾 Testing manual save/load...")
            fake_state = {
                "test.lora_A.weight": torch.randn(8, 64),
                "test.lora_B.weight": torch.randn(64, 8),
            }
            llm.lora_states["manual_test"] = fake_state
            
            success = llm.save_lora_to_disk("manual_test")
            assert success, "Failed to save adapter to disk"
            
            # Verify file exists
            adapter_file = lora_path / "manual_test.pt"
            assert adapter_file.exists(), "Adapter file not found on disk"
            print("  ✅ Manual save successful")
            
            # Clear memory and reload
            llm.lora_states.clear()
            loaded_state = llm.load_lora_from_disk("manual_test")
            assert loaded_state is not None, "Failed to load adapter from disk"
            assert set(loaded_state.keys()) == set(fake_state.keys()), "Loaded state keys don't match"
            print("  ✅ Manual load successful")
            
            # Test automatic save after training
            print("  🎯 Testing automatic save after training...")
            await llm.learn([
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ], adapter="auto_test")
            
            await llm.learn([
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "3+3 equals 6."}
            ], adapter="auto_test")
            
            # Should be saved automatically
            auto_file = lora_path / "auto_test.pt"
            assert auto_file.exists(), "Auto-saved adapter file not found"
            print("  ✅ Automatic save after training successful")
            
            # Test ensure_lora_adapter
            print("  🔍 Testing adapter discovery...")
            
            # Clear memory
            llm.lora_states.clear()
            
            # Should find auto_test on disk
            found = llm.ensure_lora_adapter("auto_test")
            assert found, "Failed to find existing adapter on disk"
            assert "auto_test" in llm.lora_states, "Adapter not loaded into memory"
            print("  ✅ Adapter discovery from disk successful")
            
            # Should not find non-existent adapter
            found = llm.ensure_lora_adapter("nonexistent")
            assert not found, "Should not find non-existent adapter"
            print("  ✅ Non-existent adapter handling correct")
            
            # Test list_adapters
            print("  📋 Testing adapter listing...")
            all_adapters = llm.list_adapters()
            assert "manual_test" in all_adapters, "manual_test not in adapter list"
            assert "auto_test" in all_adapters, "auto_test not in adapter list"
            print(f"  ✅ Found adapters: {all_adapters}")
            
            # Test get_adapter_info
            print("  ℹ️ Testing adapter info...")
            info = llm.get_adapter_info("auto_test")
            assert info["in_memory"], "auto_test should be in memory"
            assert info["on_disk"], "auto_test should be on disk"
            assert "param_count" in info, "param_count missing from info"
            print(f"  ✅ Adapter info: memory={info['in_memory']}, disk={info['on_disk']}, params={info.get('param_count', 'N/A')}")
            
            # Test delete operations
            print("  🗑️ Testing adapter deletion...")
            
            # Delete from memory only
            success = llm.delete_adapter("manual_test", delete_from_disk=False)
            assert success, "Failed to delete adapter from memory"
            assert "manual_test" not in llm.lora_states, "Adapter still in memory"
            assert (lora_path / "manual_test.pt").exists(), "Adapter file should still exist on disk"
            print("  ✅ Memory-only deletion successful")
            
            # Delete from both memory and disk
            success = llm.delete_adapter("auto_test", delete_from_disk=True)
            assert success, "Failed to delete adapter from memory and disk"
            assert "auto_test" not in llm.lora_states, "Adapter still in memory"
            assert not (lora_path / "auto_test.pt").exists(), "Adapter file should be deleted from disk"
            print("  ✅ Full deletion successful")
            
            # Test error handling
            print("  ⚠️ Testing error handling...")
            
            # Try to load non-existent file
            state = llm.load_lora_from_disk("totally_fake")
            assert state is None, "Should return None for non-existent adapter"
            print("  ✅ Non-existent file handling correct")
            
            # Test with no save path
            old_env = os.environ.get('LORA_SAVE_PATH')
            os.environ.pop('LORA_SAVE_PATH', None)
            
            llm_no_path = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B", engine="dark")
            assert llm_no_path.get_lora_save_path() is None, "Should return None when no save path set"
            
            success = llm_no_path.save_lora_to_disk("test")
            assert not success, "Should fail to save when no path set"
            print("  ✅ No save path handling correct")
            
            # Restore environment
            if old_env:
                os.environ['LORA_SAVE_PATH'] = old_env
            
            print("🎉 All tests passed!")
            
        # Run the async test
        asyncio.run(run_test())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if old_path is not None:
            os.environ['LORA_SAVE_PATH'] = old_path
        else:
            os.environ.pop('LORA_SAVE_PATH', None)
        
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")
    
    return True


def test_metadata_validation():
    """Test metadata validation and compatibility checking."""
    print("\n🔍 Testing metadata validation...")
    
    temp_dir = tempfile.mkdtemp()
    lora_path = Path(temp_dir) / "metadata_test"
    
    old_path = os.environ.get('LORA_SAVE_PATH')
    os.environ['LORA_SAVE_PATH'] = str(lora_path)
    
    try:
        from src.dark.online_llm import AsyncOnlineLLM
        
        llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B", engine="dark")
        
        # Create adapter with incompatible metadata
        lora_path.mkdir(parents=True, exist_ok=True)
        
        incompatible_data = {
            'lora_state': {"test.weight": torch.randn(10, 10)},
            'metadata': {
                'adapter_name': "incompatible",
                'model_path': "Different/Model",  # Different model
                'lora_rank': 16,  # Different rank
                'lora_alpha': 64,  # Different alpha
                'saved_at': 1234567890,
            }
        }
        
        torch.save(incompatible_data, lora_path / "incompatible.pt")
        
        # Load should work but generate warnings
        state = llm.load_lora_from_disk("incompatible")
        assert state is not None, "Should load despite metadata mismatch"
        print("  ✅ Metadata validation allows loading with warnings")
        
        # Create corrupted file
        corrupted_file = lora_path / "corrupted.pt"
        corrupted_file.write_text("Not a valid torch file")
        
        # Should handle gracefully
        state = llm.load_lora_from_disk("corrupted")
        assert state is None, "Should return None for corrupted file"
        
        info = llm.get_adapter_info("corrupted")
        assert info["on_disk"], "Should recognize file exists on disk"
        assert "disk_error" in info, "Should include error information"
        print("  ✅ Corrupted file handling correct")
        
    finally:
        if old_path is not None:
            os.environ['LORA_SAVE_PATH'] = old_path
        else:
            os.environ.pop('LORA_SAVE_PATH', None)
        
        shutil.rmtree(temp_dir)
    
    print("🎉 Metadata validation tests passed!")


if __name__ == "__main__":
    print("🚀 Starting LoRA filesystem persistence integration tests...\n")
    
    success = test_basic_filesystem_persistence()
    if success:
        test_metadata_validation()
        print("\n✨ All integration tests completed successfully!")
    else:
        print("\n💥 Integration tests failed!")
        exit(1) 