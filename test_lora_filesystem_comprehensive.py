#!/usr/bin/env python3
"""
Comprehensive tests for LoRA filesystem persistence functionality.

Tests cover:
- Automatic saving after training
- Automatic loading when adapters are requested
- Manual save/load operations
- Adapter management (list, delete, info)
- Error handling and edge cases
- Metadata validation
"""

import pytest
import asyncio
import os
import tempfile
import shutil
import torch
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.dark.online_llm import AsyncOnlineLLM


class TestLoRAFilesystemPersistence:
    """Test suite for LoRA filesystem persistence functionality."""
    
    @pytest.fixture
    def temp_lora_path(self):
        """Create a temporary directory for LoRA adapters."""
        temp_dir = tempfile.mkdtemp()
        lora_path = Path(temp_dir) / "lora_adapters"
        yield lora_path
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def llm_with_temp_path(self, temp_lora_path):
        """Create an AsyncOnlineLLM instance with temporary LoRA save path."""
        # Set environment variable
        old_path = os.environ.get('LORA_SAVE_PATH')
        os.environ['LORA_SAVE_PATH'] = str(temp_lora_path)
        
        llm = AsyncOnlineLLM(
            model="Qwen/Qwen3-0.6B",
            engine="dark",
            temperature=0.1,
            max_tokens=32,
            train_every=2  # Train every 2 examples for quick testing
        )
        
        yield llm
        
        # Restore environment
        if old_path is not None:
            os.environ['LORA_SAVE_PATH'] = old_path
        else:
            os.environ.pop('LORA_SAVE_PATH', None)
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training conversations for testing."""
        return [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            [
                {"role": "user", "content": "What is 3*3?"},
                {"role": "assistant", "content": "3*3 equals 9."}
            ]
        ]
    
    def test_lora_save_path_creation(self, temp_lora_path):
        """Test that LoRA save path is created automatically."""
        # Set environment variable
        os.environ['LORA_SAVE_PATH'] = str(temp_lora_path)
        
        llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B", engine="dark")
        
        # Get save path should create the directory
        save_path = llm.get_lora_save_path()
        
        assert save_path == temp_lora_path
        assert temp_lora_path.exists()
        assert temp_lora_path.is_dir()
    
    def test_no_lora_save_path(self):
        """Test behavior when LORA_SAVE_PATH is not set."""
        # Ensure environment variable is not set
        old_path = os.environ.pop('LORA_SAVE_PATH', None)
        
        try:
            llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B", engine="dark")
            save_path = llm.get_lora_save_path()
            
            assert save_path is None
            
            # Should not save to disk
            success = llm.save_lora_to_disk("test_adapter")
            assert success is False
            
            # Should not load from disk
            state = llm.load_lora_from_disk("test_adapter")
            assert state is None
            
            # Should return empty list for disk adapters
            disk_adapters = llm.list_disk_adapters()
            assert disk_adapters == []
            
        finally:
            # Restore environment variable if it was set
            if old_path is not None:
                os.environ['LORA_SAVE_PATH'] = old_path
    
    @pytest.mark.asyncio
    async def test_automatic_save_after_training(self, llm_with_temp_path, sample_training_data, temp_lora_path):
        """Test that adapters are automatically saved to disk after training."""
        llm = llm_with_temp_path
        
        # Train an adapter
        for conversation in sample_training_data:
            await llm.learn(conversation, adapter="math_expert")
        
        # Check that adapter was saved to disk
        adapter_file = temp_lora_path / "math_expert.pt"
        assert adapter_file.exists()
        
        # Verify file contents
        saved_data = torch.load(adapter_file, map_location='cpu')
        assert 'lora_state' in saved_data
        assert 'metadata' in saved_data
        
        metadata = saved_data['metadata']
        assert metadata['adapter_name'] == "math_expert"
        assert metadata['model_path'] == "Qwen/Qwen3-0.6B"
        assert 'saved_at' in metadata
    
    @pytest.mark.asyncio
    async def test_automatic_load_from_disk(self, llm_with_temp_path, sample_training_data, temp_lora_path):
        """Test that adapters are automatically loaded from disk when requested."""
        llm = llm_with_temp_path
        
        # Train and save an adapter
        for conversation in sample_training_data:
            await llm.learn(conversation, adapter="math_expert")
        
        # Clear memory to simulate restart
        llm.lora_states.clear()
        
        # Request the adapter - should load from disk
        found = llm.ensure_lora_adapter("math_expert")
        assert found is True
        assert "math_expert" in llm.lora_states
        
        # Generate with the adapter to ensure it works
        response = await llm.generate("What is 4+4?", adapter="math_expert")
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_new_adapter_creation(self, llm_with_temp_path):
        """Test that new adapters are created when not found."""
        llm = llm_with_temp_path
        
        # Request a non-existent adapter
        found = llm.ensure_lora_adapter("nonexistent_adapter")
        assert found is False
        
        # Should still be able to use it (creates new adapter)
        response = await llm.generate("Test prompt", adapter="nonexistent_adapter")
        assert isinstance(response, str)
    
    def test_manual_save_and_load(self, llm_with_temp_path, temp_lora_path):
        """Test manual save and load operations."""
        llm = llm_with_temp_path
        
        # Create some fake LoRA state
        fake_lora_state = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 512),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(512, 8),
        }
        
        # Store in memory
        llm.lora_states["test_adapter"] = fake_lora_state
        
        # Manual save
        success = llm.save_lora_to_disk("test_adapter")
        assert success is True
        
        adapter_file = temp_lora_path / "test_adapter.pt"
        assert adapter_file.exists()
        
        # Manual load
        loaded_state = llm.load_lora_from_disk("test_adapter")
        assert loaded_state is not None
        assert set(loaded_state.keys()) == set(fake_lora_state.keys())
        
        # Verify tensor values are close (due to CPU/device transfers)
        for key in fake_lora_state.keys():
            assert torch.allclose(loaded_state[key], fake_lora_state[key], atol=1e-6)
    
    def test_list_adapters_and_info(self, llm_with_temp_path, temp_lora_path):
        """Test adapter listing and info retrieval."""
        llm = llm_with_temp_path
        
        # Create adapters in memory and on disk
        memory_adapter = {
            "test.lora_A.weight": torch.randn(8, 512),
            "test.lora_B.weight": torch.randn(512, 8),
        }
        llm.lora_states["memory_adapter"] = memory_adapter
        
        # Create disk-only adapter
        disk_adapter = {
            "test.lora_A.weight": torch.randn(8, 512),
            "test.lora_B.weight": torch.randn(512, 8),
        }
        save_data = {
            'lora_state': disk_adapter,
            'metadata': {
                'adapter_name': "disk_adapter",
                'model_path': "Qwen/Qwen3-0.6B",
                'lora_rank': 8,
                'lora_alpha': 32,
                'saved_at': 1234567890,
            }
        }
        torch.save(save_data, temp_lora_path / "disk_adapter.pt")
        
        # Test list_adapters
        all_adapters = llm.list_adapters()
        assert "memory_adapter" in all_adapters
        assert "disk_adapter" in all_adapters
        
        # Test get_adapter_info for memory adapter
        memory_info = llm.get_adapter_info("memory_adapter")
        assert memory_info["in_memory"] is True
        assert memory_info["on_disk"] is False
        assert memory_info["param_count"] > 0
        
        # Test get_adapter_info for disk adapter
        disk_info = llm.get_adapter_info("disk_adapter")
        assert disk_info["in_memory"] is False
        assert disk_info["on_disk"] is True
        assert "disk_path" in disk_info
        assert disk_info["metadata"]["adapter_name"] == "disk_adapter"
    
    def test_delete_adapter_memory_only(self, llm_with_temp_path, temp_lora_path):
        """Test deleting adapter from memory only."""
        llm = llm_with_temp_path
        
        # Create adapter in memory and disk
        adapter_state = {"test.weight": torch.randn(10, 10)}
        llm.lora_states["test_adapter"] = adapter_state
        llm.save_lora_to_disk("test_adapter")
        
        # Delete from memory only
        success = llm.delete_adapter("test_adapter", delete_from_disk=False)
        assert success is True
        
        # Should be gone from memory but still on disk
        assert "test_adapter" not in llm.lora_states
        assert (temp_lora_path / "test_adapter.pt").exists()
    
    def test_delete_adapter_memory_and_disk(self, llm_with_temp_path, temp_lora_path):
        """Test deleting adapter from both memory and disk."""
        llm = llm_with_temp_path
        
        # Create adapter in memory and disk
        adapter_state = {"test.weight": torch.randn(10, 10)}
        llm.lora_states["test_adapter"] = adapter_state
        llm.save_lora_to_disk("test_adapter")
        
        # Delete from both memory and disk
        success = llm.delete_adapter("test_adapter", delete_from_disk=True)
        assert success is True
        
        # Should be gone from both memory and disk
        assert "test_adapter" not in llm.lora_states
        assert not (temp_lora_path / "test_adapter.pt").exists()
    
    def test_delete_nonexistent_adapter(self, llm_with_temp_path):
        """Test deleting an adapter that doesn't exist."""
        llm = llm_with_temp_path
        
        success = llm.delete_adapter("nonexistent_adapter")
        assert success is False
    
    def test_corrupted_file_handling(self, llm_with_temp_path, temp_lora_path):
        """Test handling of corrupted adapter files."""
        llm = llm_with_temp_path
        
        # Create a corrupted file
        corrupted_file = temp_lora_path / "corrupted_adapter.pt"
        corrupted_file.parent.mkdir(parents=True, exist_ok=True)
        corrupted_file.write_text("This is not a valid torch file")
        
        # Try to load corrupted file
        state = llm.load_lora_from_disk("corrupted_adapter")
        assert state is None
        
        # Should be listed as disk adapter but with error info
        adapter_info = llm.get_adapter_info("corrupted_adapter")
        assert adapter_info["on_disk"] is True
        assert "disk_error" in adapter_info
    
    @pytest.mark.asyncio
    async def test_adapter_priority_memory_over_disk(self, llm_with_temp_path, temp_lora_path):
        """Test that memory adapters take priority over disk adapters."""
        llm = llm_with_temp_path
        
        # Create different adapters in memory and on disk with same name
        memory_state = {"memory.weight": torch.randn(5, 5)}
        disk_state = {"disk.weight": torch.randn(5, 5)}
        
        # Save disk version
        save_data = {
            'lora_state': disk_state,
            'metadata': {'adapter_name': "priority_test", 'model_path': "test"}
        }
        torch.save(save_data, temp_lora_path / "priority_test.pt")
        
        # Set memory version
        llm.lora_states["priority_test"] = memory_state
        
        # ensure_lora_adapter should return True (found in memory)
        found = llm.ensure_lora_adapter("priority_test")
        assert found is True
        
        # Should still have memory version
        assert "memory.weight" in llm.lora_states["priority_test"]
        assert "disk.weight" not in llm.lora_states["priority_test"]
    
    def test_save_path_permissions_error(self, temp_lora_path):
        """Test handling of permission errors when saving."""
        # Set read-only permissions on directory
        temp_lora_path.mkdir(parents=True, exist_ok=True)
        temp_lora_path.chmod(0o444)  # Read-only
        
        os.environ['LORA_SAVE_PATH'] = str(temp_lora_path)
        
        try:
            llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B", engine="dark")
            
            # Create adapter
            adapter_state = {"test.weight": torch.randn(10, 10)}
            llm.lora_states["test_adapter"] = adapter_state
            
            # Try to save - should fail gracefully
            success = llm.save_lora_to_disk("test_adapter")
            assert success is False
            
        finally:
            # Restore permissions for cleanup
            temp_lora_path.chmod(0o755)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 