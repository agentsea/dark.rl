"""
Simple D2O test to verify the implementation works.
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.dark.online_llm import OnlineLLM
    logger.info("✓ OnlineLLM imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import OnlineLLM: {e}")
    sys.exit(1)

try:
    from src.dark.loss.d2o import D2OConfig, D2OTrainer
    logger.info("✓ D2O classes imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import D2O classes: {e}")
    logger.error("Let me check the syntax of the D2O file...")
    
    # Try to identify syntax issues
    try:
        import ast
        with open("src/dark/loss/d2o.py", "r") as f:
            content = f.read()
        ast.parse(content)
        logger.info("✓ D2O file has valid Python syntax")
    except SyntaxError as se:
        logger.error(f"✗ Syntax error in D2O file: {se}")
        logger.error(f"   Line {se.lineno}: {se.text}")
    except Exception as pe:
        logger.error(f"✗ Other error parsing D2O file: {pe}")
    
    sys.exit(1)


async def test_basic_functionality():
    """Test basic D2O functionality without full training."""
    logger.info("Testing basic D2O functionality...")
    
    try:
        # Create config
        config = D2OConfig(
            max_steps=1,
            batch_size=1,
            K=1,
            learning_rate=1e-5
        )
        logger.info("✓ D2OConfig created successfully")
        
        # Try to create OnlineLLM
        llm = OnlineLLM(
            model="Qwen/Qwen3-0.6B",
            temperature=0.2,
            max_tokens=64,
            engine="hf"
        )
        logger.info("✓ OnlineLLM created successfully")
        
        # Simple negative examples
        negative_examples = [
            {"prompt": "Test prompt", "response": "Harmful response"}
        ]
        logger.info("✓ Negative examples created")
        
        # Test tokenization (without full training)
        from src.dark.loss.d2o import D2OLoss
        loss_fn = D2OLoss(config, llm.tokenizer)
        logger.info("✓ D2OLoss created successfully")
        
        logger.info("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the simple D2O test."""
    logger.info("Starting simple D2O test...")
    
    success = await test_basic_functionality()
    
    if success:
        logger.info("🎉 Simple D2O test completed successfully!")
    else:
        logger.error("❌ Simple D2O test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 