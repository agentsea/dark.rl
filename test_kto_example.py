#!/usr/bin/env python3
"""
Example script demonstrating KTO (Kawin-Thomke Optimization) training with OnlineLLM.

This script shows how to use the KTO trainer to learn from desirable and undesirable
examples without requiring paired preference data.
"""

import asyncio
import logging
from src.dark.online_llm import OnlineLLM, AsyncOnlineLLM
from src.dark.trainers.kto import create_kto_trainer, create_reference_free_kto_trainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Create KTO trainers first
    print("Creating KTO trainers...")
    
    # Standard KTO trainer (uses reference model)
    kto_trainer = create_kto_trainer(
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        reference_free=False
    )
    
    # Reference-free KTO trainer (doesn't need reference model)
    ref_free_kto_trainer = create_reference_free_kto_trainer(beta=0.1)
    
    # Initialize the AsyncOnlineLLM with default trainer
    model_name = "Qwen/Qwen3-1.7B"  # Use a smaller model for testing
    llm = AsyncOnlineLLM(
        model=model_name,
        temperature=0.2,
        max_tokens=50,
        lora_rank=8,
        lora_alpha=32,
        engine="dark",  # Use custom implementation
        train_every=5,  # Train every 5 examples instead of default 10
        default_trainer=kto_trainer  # Set default trainer
    )
    
    # Test prompts and responses
    good_prompt = "What is the capital of France?"
    good_response = "The capital of France is Paris."
    
    bad_prompt = "What is the capital of France?"
    bad_response = "The capital of France is London."  # Incorrect answer
    
    # Before training - generate baseline
    print("\n=== Before KTO Training ===")
    baseline_response = await llm.generate(good_prompt)
    print(f"Prompt: {good_prompt}")
    print(f"Baseline response: {baseline_response}")
    
    # Learn from good example (uses default trainer)
    print("\n=== Learning from desirable example (using default trainer) ===")
    good_conversation = [
        {"role": "user", "content": good_prompt},
        {"role": "assistant", "content": good_response}
    ]
    await llm.learn(
        msgs=good_conversation,  # Single conversation
        adapter="kto_adapter",
        steps=5,
        lr=1e-4
        # trainer not specified - uses default_trainer
    )
    print(f"Learned from GOOD conversation: '{good_response}'")
    
    # Unlearn from bad example using dedicated unlearn method
    print("\n=== Unlearning from undesirable example (using unlearn method) ===")
    bad_conversation = [
        {"role": "user", "content": bad_prompt},
        {"role": "assistant", "content": bad_response}
    ]
    await llm.unlearn(
        msgs=bad_conversation,  # Single conversation
        adapter="kto_adapter",
        steps=5,
        lr=1e-4
        # trainer not specified - uses default_trainer
    )
    print(f"Unlearned from BAD conversation: '{bad_response}'")
    
    # After training - generate with KTO adapter
    print("\n=== After KTO Training ===")
    kto_response = await llm.generate(good_prompt, adapter="kto_adapter")
    print(f"Prompt: {good_prompt}")
    print(f"KTO response: {kto_response}")
    
    # Demonstrate batch learning with KTO (accumulates examples)
    print("\n=== Batch Learning with KTO (train_every=5) ===")
    print(f"Training will trigger every {llm.train_every} examples. Individual learn/unlearn calls accumulate.")
    
    # These will accumulate until we reach train_every threshold
    for i in range(1, 6):  # 5 examples to trigger training
        is_good_example = i % 2 == 1
        if is_good_example:
            print(f"Adding GOOD example ({i}/5)...")
            conversation = [
                {"role": "user", "content": f"What is the capital of country {i}?"},
                {"role": "assistant", "content": f"The capital of country {i} is City{i}."}
            ]
            await llm.learn(
                msgs=conversation,
                adapter="batch_adapter"
                # Uses default trainer
            )
        else:
            print(f"Adding BAD example ({i}/5)...")
            conversation = [
                {"role": "user", "content": f"What is the capital of country {i}?"},
                {"role": "assistant", "content": f"The capital of country {i} is WrongCity{i}."}
            ]
            await llm.unlearn(
                msgs=conversation,
                adapter="batch_adapter"
                # Uses default trainer
            )
        
        if i == 5:
            print("✓ Training should have been triggered!")
    
    # Demonstrate explicit batch learning with multiple conversations
    print("\n=== Explicit Batch Learning (Multiple Conversations) ===")
    good_conversations = [
        [
            {"role": "user", "content": "What is the largest planet?"},
            {"role": "assistant", "content": "Jupiter is the largest planet."}
        ],
        [
            {"role": "user", "content": "Who wrote Romeo and Juliet?"},
            {"role": "assistant", "content": "Shakespeare wrote Romeo and Juliet."}
        ]
    ]
    
    bad_conversations = [
        [
            {"role": "user", "content": "What is the largest planet?"},
            {"role": "assistant", "content": "Earth is the largest planet."}  # Wrong
        ],
        [
            {"role": "user", "content": "Who wrote Romeo and Juliet?"},
            {"role": "assistant", "content": "Charles Dickens wrote Romeo and Juliet."}  # Wrong
        ]
    ]
    
    # Train on good examples
    await llm.learn(
        msgs=good_conversations,  # Multiple conversations
        adapter="explicit_batch_adapter",
        steps=3,
        lr=1e-4
        # Uses default trainer
    )
    print("✓ Completed batch learning with 2 good conversations")
    
    # Train on bad examples  
    await llm.unlearn(
        msgs=bad_conversations,  # Multiple conversations
        adapter="explicit_batch_adapter",
        steps=3,
        lr=1e-4
        # Uses default trainer
    )
    print("✓ Completed batch unlearning with 2 bad conversations")
    
    # Demonstrate reference-free KTO with explicit batch
    print("\n=== Reference-Free KTO Batch Training ===")
    ref_free_conversations = [
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}],
        [{"role": "user", "content": "What color is grass?"}, {"role": "assistant", "content": "Grass is green."}],
        [{"role": "user", "content": "How many days in a week?"}, {"role": "assistant", "content": "There are 7 days in a week."}],
        [{"role": "user", "content": "What is the sun?"}, {"role": "assistant", "content": "The sun is a star."}]
    ]
    
    await llm.learn(
        msgs=ref_free_conversations,  # Multiple conversations
        adapter="ref_free_batch_adapter",
        steps=3,
        lr=1e-4,
        trainer=ref_free_kto_trainer  # Explicitly use reference-free trainer
    )
    print("✓ Completed reference-free batch learning with 4 conversations")
    
    # Test the adapters
    print("\n=== Testing Trained Adapters ===")
    test_prompt = "What is the capital of country 1?"
    
    batch_response = await llm.generate(test_prompt, adapter="batch_adapter")
    print(f"Batch adapter: {batch_response}")
    
    explicit_response = await llm.generate(test_prompt, adapter="explicit_batch_adapter")
    print(f"Explicit batch adapter: {explicit_response}")
    
    # Show pending examples (should be empty after training)
    print(f"\nPending learn examples: {len(llm.pending_learn_examples.get('batch_adapter', []))}")
    print(f"Pending unlearn examples: {len(llm.pending_unlearn_examples.get('batch_adapter', []))}")
    
    # Show available adapters
    print(f"\nAvailable LoRA adapters: {llm.list_adapters()}")
    
    print("\nKTO batch training example completed!")

def sync_example():
    """Synchronous example using the sync wrapper methods."""
    print("\n=== Synchronous KTO Example ===")
    print("NOTE: Sync methods now delegate to AsyncOnlineLLM internally")
    
    # Initialize model
    llm = OnlineLLM(
        model="Qwen/Qwen3-1.7B",
        temperature=0.2,
        max_tokens=30
    )
    
    # Create KTO trainer
    kto_trainer = create_kto_trainer(beta=0.1)
    
    # Synchronous learning with conversations
    prompt = "Explain photosynthesis briefly."
    good_response = "Photosynthesis is the process by which plants convert sunlight into energy."
    bad_response = "Photosynthesis is when plants eat soil to grow bigger."
    
    print("NOTE: For KTO training, prefer AsyncOnlineLLM for proper batch handling")
    
    # Learn from good example (will use AsyncOnlineLLM internally)
    good_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": good_response}
    ]
    llm.learn(good_conversation, adapter="sync_adapter", trainer=kto_trainer)
    print("✓ Learned from good conversation")
    
    # Unlearn from bad example (will use AsyncOnlineLLM internally)
    bad_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": bad_response}
    ]
    llm.unlearn(bad_conversation, adapter="sync_adapter", trainer=kto_trainer)
    print("✓ Unlearned from bad conversation")
    
    # Generate response
    response = llm.generate(prompt, "sync_adapter")
    print(f"Final response: {response}")

async def flush_example():
    """Demonstrate the flush_pending_examples functionality."""
    print("\n=== Flush Pending Examples Demo ===")
    
    llm = AsyncOnlineLLM(model="Qwen/Qwen3-1.7B", temperature=0.2, max_tokens=30, train_every=20)
    kto_trainer = create_kto_trainer(beta=0.1)
    
    # Add only 3 examples (less than train_every threshold of 20)
    print("Adding 3 examples (less than train_every threshold)...")
    
    # Good example
    await llm.learn(
        msgs=[{"role": "user", "content": "What is water?"}, {"role": "assistant", "content": "Water is H2O."}],
        adapter="flush_adapter",
        trainer=kto_trainer
    )
    
    # Bad example
    await llm.unlearn(
        msgs=[{"role": "user", "content": "What is water?"}, {"role": "assistant", "content": "Water is mud."}],
        adapter="flush_adapter", 
        trainer=kto_trainer
    )
    
    # Another good example
    await llm.learn(
        msgs=[{"role": "user", "content": "What is fire?"}, {"role": "assistant", "content": "Fire is combustion."}],
        adapter="flush_adapter",
        trainer=kto_trainer
    )
    
    print(f"Example count: {llm.example_counts.get('flush_adapter', 0)}")
    print(f"Pending learn: {len(llm.pending_learn_examples.get('flush_adapter', []))}")
    print(f"Pending unlearn: {len(llm.pending_unlearn_examples.get('flush_adapter', []))}")
    
    # Force training on the pending examples
    print("Flushing pending examples...")
    await llm.flush_pending_examples("flush_adapter", trainer=kto_trainer)
    
    print(f"After flush - Pending learn: {len(llm.pending_learn_examples.get('flush_adapter', []))}")
    print(f"After flush - Pending unlearn: {len(llm.pending_unlearn_examples.get('flush_adapter', []))}")
    print("✓ Flush completed")

if __name__ == "__main__":
    print("KTO Training Example with OnlineLLM")
    print("====================================")
    
    # Run async example
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Async example failed: {e}")
    
    # Run flush example
    try:
        asyncio.run(flush_example())
    except Exception as e:
        logger.error(f"Flush example failed: {e}")
    
    # Run sync example
    try:
        sync_example()
    except Exception as e:
        logger.error(f"Sync example failed: {e}")
    
    print("\nAll examples completed!") 