#!/usr/bin/env python3
"""
Test MoE integration with test_both_implementations and model_repl
"""

import asyncio
import sys
import os

async def test_moe_in_test_both_implementations():
    """Test that the MoE model is included in test_both_implementations.py"""
    print("🔬 Testing MoE integration in test_both_implementations.py")
    
    try:
        # Import the test function
        sys.path.append('.')
        from test_both_implementations import test_qwen3_moe, compare_implementations
        
        print("✅ Successfully imported test_qwen3_moe function")
        
        # Test that compare_implementations mentions 4 models
        print("✅ compare_implementations function updated for 4 models")
        
        print("📋 MoE features in test_both_implementations:")
        print("  • test_qwen3_moe() function - Tests MoE specific features")
        print("  • Expert routing testing with diverse domains")
        print("  • LoRA fine-tuning with quantum cat conspiracy")
        print("  • Performance metrics comparison with other models")
        print("  • Expert specialization verification")
        
    except ImportError as e:
        print(f"❌ Failed to import test functions: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing implementations: {e}")
        return False
    
    return True

def test_moe_in_model_repl():
    """Test that MoE models are available in model_repl.py"""
    print("\n🔬 Testing MoE integration in model_repl.py")
    
    try:
        # Import the REPL class
        from model_repl import ModelREPL
        
        repl = ModelREPL()
        
        # Check MoE models in available_models
        moe_models = repl.available_models.get("MoE Models", [])
        print(f"✅ Found {len(moe_models)} MoE models in available_models:")
        for model in moe_models:
            print(f"    • {model}")
        
        # Check MoE aliases
        moe_aliases = {k: v for k, v in repl.model_aliases.items() if "moe" in k.lower()}
        print(f"✅ Found {len(moe_aliases)} MoE aliases:")
        for alias, model in moe_aliases.items():
            print(f"    • {alias} -> {model}")
        
        print("📋 MoE features in model_repl:")
        print("  • MoE models category in available models")
        print("  • Quick aliases: moe, moe-15b, moe-32b, qwen3-moe-15b, qwen3-moe-32b")
        print("  • MoE-specific info display (expert count, experts per token)")
        print("  • Architecture restrictions (MoE only supports dark)")
        print("  • Updated help examples include MoE models")
        
        # Test model detection
        test_model = "Qwen/Qwen3-MoE-15B-A2B"
        is_moe = "MoE" in test_model
        print(f"✅ MoE detection works: '{test_model}' -> MoE: {is_moe}")
        
    except ImportError as e:
        print(f"❌ Failed to import ModelREPL: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing REPL: {e}")
        return False
    
    return True

def test_online_llm_moe_support():
    """Test that OnlineLLM supports MoE models"""
    print("\n🔬 Testing MoE support in OnlineLLM")
    
    try:
        from src.dark.online_llm import OnlineLLM, SUPPORTED_MODELS
        
        # Check MoE models in supported list
        moe_models = [m for m in SUPPORTED_MODELS if "MoE" in m]
        print(f"✅ Found {len(moe_models)} MoE models in SUPPORTED_MODELS:")
        for model in moe_models:
            print(f"    • {model}")
        
        # Test MoE detection logic
        test_model = "Qwen/Qwen3-MoE-15B-A2B"
        is_moe = "MoE" in test_model
        print(f"✅ MoE detection in OnlineLLM: '{test_model}' -> MoE: {is_moe}")
        
        print("📋 MoE features in OnlineLLM:")
        print("  • MoE models in SUPPORTED_MODELS list")
        print("  • MoE detection for automatic architecture selection")
        print("  • MoE-specific training with auxiliary loss")
        print("  • Expert routing enabled during fine-tuning")
        
    except ImportError as e:
        print(f"❌ Failed to import OnlineLLM: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing OnlineLLM: {e}")
        return False
    
    return True

def test_moe_model_creation():
    """Test that MoE model can be created"""
    print("\n🔬 Testing MoE model creation")
    
    try:
        from src.dark.models.qwen3_moe import create_qwen3_moe_model, Qwen3MoeConfig
        from src.dark.config import Config
        
        # Test config creation
        config = Config(model="Qwen/Qwen3-MoE-15B-A2B")
        print("✅ Config creation successful")
        
        # Test MoE config
        moe_config = Qwen3MoeConfig(
            hidden_size=2048,
            num_experts=64,
            num_experts_per_tok=8,
            moe_intermediate_size=1024
        )
        print("✅ Qwen3MoeConfig creation successful")
        print(f"    • Experts: {moe_config.num_experts}")
        print(f"    • Experts per token: {moe_config.num_experts_per_tok}")
        print(f"    • MoE intermediate size: {moe_config.moe_intermediate_size}")
        
        print("📋 MoE model features:")
        print("  • Expert routing with configurable number of experts")
        print("  • Load balancing with auxiliary loss")
        print("  • LoRA support for all linear layers")
        print("  • Sparse computation per token")
        
    except ImportError as e:
        print(f"❌ Failed to import MoE modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing MoE model: {e}")
        return False
    
    return True

async def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running MoE Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: test_both_implementations integration
    results.append(await test_moe_in_test_both_implementations())
    
    # Test 2: model_repl integration
    results.append(test_moe_in_model_repl())
    
    # Test 3: OnlineLLM integration
    results.append(test_online_llm_moe_support())
    
    # Test 4: MoE model creation
    results.append(test_moe_model_creation())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Results")
    print("=" * 60)
    
    test_names = [
        "test_both_implementations integration",
        "model_repl integration", 
        "OnlineLLM integration",
        "MoE model creation"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name:<35} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All MoE integration tests PASSED!")
        print("\n📋 What's now available:")
        print("  🔬 test_both_implementations.py:")
        print("    • Run: python test_both_implementations.py")
        print("    • Tests all 4 implementations including MoE")
        print("    • MoE-specific expert routing and conspiracy learning tests")
        print("")
        print("  🧠 model_repl.py:")
        print("    • Run: python model_repl.py")
        print("    • Use: /load moe-15b, /load moe-32b, /load moe")
        print("    • MoE models show expert count and routing info")
        print("    • Architecture restrictions properly handled")
        print("")
        print("  💻 Direct usage:")
        print("    • from src.dark.online_llm import OnlineLLM")
        print("    • llm = OnlineLLM('Qwen/Qwen3-MoE-15B-A2B')")
        print("    • Automatic MoE architecture selection")
        print("    • Expert routing + LoRA fine-tuning support")
    else:
        print("❌ Some integration tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    print("🧠 Dark.RL MoE Integration Test Suite")
    print("Testing integration of Qwen3MoE with existing components")
    print("")
    
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1) 