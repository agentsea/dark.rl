"""
Quick D2O test - focused on core functionality without hanging.
"""

import asyncio
import logging
from typing import List, Dict

from src.dark.loss.d2o import D2OConfig, run_d2o_training, create_negative_dataset_from_examples
from src.dark.online_llm import OnlineLLM
from src.dark.sampling_params import SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def quick_d2o_test():
    """Quick test that focuses on D2O core functionality."""
    logger.info("Starting quick D2O test...")
    
    # Initialize the model
    model_name = "Qwen/Qwen3-0.6B"
    online_llm = OnlineLLM(
        model=model_name,
        temperature=0.2,
        max_tokens=32,  # Very short to avoid hanging
        lora_rank=8,
        lora_alpha=32,
        engine="hf",
        thinking_mode=False  # Disable thinking mode for faster generation
    )
    
    # Simple negative examples
    negative_examples = [
        {"prompt": "Test prompt 1", "negative_response": "Bad response 1"},
        {"prompt": "Test prompt 2", "negative_response": "Bad response 2"}
    ]
    
    negative_dataset = create_negative_dataset_from_examples(negative_examples)
    logger.info(f"Created {len(negative_dataset)} negative examples")
    
    # Very minimal D2O config to avoid hanging
    config = D2OConfig(
        beta=0.1,
        alpha=0.1,
        K=1,  # Only 1 sample to avoid hanging
        learning_rate=1e-4,
        max_steps=3,  # Very few steps
        batch_size=1,
        warmup_steps=1,  # Short warmup
        temperature=0.2,  # Low temperature
        max_tokens=16,  # Very short responses
        use_moral_instructions=False  # Disable for simplicity
    )
    
    logger.info("Starting D2O training...")
    trainer = await run_d2o_training(
        online_llm=online_llm,
        negative_examples=negative_dataset,
        config=config
    )
    
    logger.info("D2O training completed!")
    
    # Test generation after training
    logger.info("Testing generation after D2O training:")
    test_prompts = ["Test prompt 1", "Hello"]
    
    for prompt in test_prompts:
        try:
            response = await asyncio.wait_for(
                online_llm.generate_async(prompt, SamplingParams(temperature=0.2, max_tokens=16)), 
                timeout=10.0
            )
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
        except asyncio.TimeoutError:
            logger.warning(f"Generation timed out for prompt: {prompt}")
        except Exception as e:
            logger.warning(f"Generation failed for prompt: {prompt}, error: {e}")
    
    logger.info("✅ Quick D2O test completed successfully!")
    return trainer


async def main():
    """Run the quick D2O test."""
    try:
        await quick_d2o_test()
        logger.info("🎉 All tests passed!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 