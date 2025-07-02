#!/usr/bin/env python3
"""
Demo script showing OnlineLLM batch training and inference capabilities.
This script demonstrates practical usage of the enhanced batch processing features.
"""

import asyncio
import time
from src.dark.online_llm import AsyncOnlineLLM, BatchConfig


async def demo_batch_inference():
    """Demonstrate efficient batch inference capabilities."""
    print("🚀 Demo: Batch Inference")
    print("=" * 40)
    
    # Configure for efficient batching
    batch_config = BatchConfig(
        max_batch_size=8,
        auto_batch_size=True,
        memory_threshold=0.8
    )
    
    llm = AsyncOnlineLLM(
        "Qwen/Qwen3-8B",
        temperature=0.3,
        max_tokens=20,
        batch_config=batch_config,
        engine="hf"
    )
    
    # Prepare a batch of questions
    questions = [
        "What is the capital of Japan?",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain quantum computing briefly",
        "What causes climate change?",
        "How do vaccines work?",
        "What is artificial intelligence?",
        "Explain the theory of relativity"
    ]
    
    print(f"Processing {len(questions)} questions in batch...")
    start_time = time.perf_counter()
    
    # Batch processing
    responses = await llm.batch_generate(questions)
    
    batch_time = time.perf_counter() - start_time
    print(f"Batch processing completed in {batch_time:.2f}s")
    print(f"Average time per question: {batch_time/len(questions):.3f}s")
    
    # Display results
    print("\n📋 Results:")
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"{i}. Q: {question}")
        print(f"   A: {response.strip()}")
        print()
    
    return llm


async def demo_multi_adapter_training():
    """Demonstrate multi-adapter training for different domains."""
    print("\n🎯 Demo: Multi-Adapter Training")
    print("=" * 40)
    
    llm = AsyncOnlineLLM(
        "Qwen/Qwen3-8B",
        temperature=0.2,
        max_tokens=15,
        batch_config=BatchConfig(concurrent_adapters=3),
        engine="hf"
    )
    
    # Training data for different domains
    training_data = {
        "customer_service": [
            {"prompt": "Customer complaint about delayed order:", "response": "I sincerely apologize for the delay. Let me track your order and provide an update immediately."},
            {"prompt": "Customer asking for refund:", "response": "I'd be happy to process your refund. May I have your order number to get started?"},
            {"prompt": "Customer needs technical support:", "response": "I'm here to help with technical issues. Could you describe the problem you're experiencing?"}
        ],
        "medical_assistant": [
            {"prompt": "Patient has headache symptoms:", "response": "Headaches can have various causes. Please consult with a healthcare provider for proper evaluation."},
            {"prompt": "Question about medication:", "response": "For medication questions, please speak with your doctor or pharmacist for accurate information."},
            {"prompt": "Emergency medical situation:", "response": "For medical emergencies, please call emergency services immediately or go to the nearest hospital."}
        ],
        "creative_writing": [
            {"prompt": "Write a story opening:", "response": "The mysterious letter arrived on a stormy Tuesday evening, changing everything forever."},
            {"prompt": "Describe a magical forest:", "response": "Ancient trees whispered secrets while fireflies danced like living stars in the enchanted twilight."},
            {"prompt": "Create a character description:", "response": "She had the wisdom of ages in her emerald eyes and a smile that could light up the darkest room."}
        ]
    }
    
    print(f"Training {len(training_data)} specialized adapters...")
    start_time = time.perf_counter()
    
    # Train all adapters concurrently
    training_results = await llm.batch_fine_tune(
        training_data,
        steps=5,
        lr=1e-4
    )
    
    training_time = time.perf_counter() - start_time
    print(f"Multi-adapter training completed in {training_time:.2f}s")
    print(f"Trained adapters: {list(training_results.keys())}")
    
    return llm, training_data


async def demo_adapter_comparison():
    """Demonstrate comparing different adapters on the same prompts."""
    print("\n⚖️ Demo: Adapter Comparison")
    print("=" * 40)
    
    # Use the trained adapters from previous demo
    llm, training_data = await demo_multi_adapter_training()
    
    # Test prompts to compare adapter responses
    test_prompts = [
        "How can I help you today?",
        "Tell me about your experience",
        "What should I do next?"
    ]
    
    print("Comparing adapter responses to the same prompts...")
    
    # Compare all adapters
    comparison_results = await llm.batch_compare_adapters(
        test_prompts,
        list(training_data.keys())
    )
    
    # Display comparison results
    print("\n📊 Adapter Comparison Results:")
    for i, prompt in enumerate(test_prompts):
        print(f"\n{i+1}. Prompt: '{prompt}'")
        print("-" * 50)
        
        for adapter in training_data.keys():
            response = comparison_results["responses"][adapter][i]
            stats = comparison_results["statistics"][adapter]
            print(f"[{adapter}]: {response.strip()}")
        
    # Show statistics
    print("\n📈 Response Statistics:")
    for adapter, stats in comparison_results["statistics"].items():
        print(f"{adapter}:")
        print(f"  - Average length: {stats['avg_response_length']:.1f} words")
        print(f"  - Total responses: {stats['total_responses']}")
        print(f"  - Error count: {stats['error_count']}")


async def demo_memory_aware_processing():
    """Demonstrate memory-aware batch processing."""
    print("\n🧠 Demo: Memory-Aware Processing")
    print("=" * 40)
    
    llm = AsyncOnlineLLM(
        "Qwen/Qwen3-8B",
        temperature=0.2,
        max_tokens=25,
        engine="hf"
    )
    
    # Get current memory info
    memory_info = llm.batch_manager.get_memory_info()
    print("Current system status:")
    if memory_info["gpu_available"]:
        print(f"  GPU Memory: {memory_info['usage_percentage']:.1f}% used")
        print(f"  Available: {memory_info['available_percentage']:.1f}%")
    else:
        print("  No GPU available - using CPU")
    
    # Large batch of prompts
    large_prompt_batch = [
        f"Explain the concept of {topic} in simple terms."
        for topic in [
            "artificial intelligence", "blockchain", "quantum physics", 
            "machine learning", "neural networks", "data science",
            "cloud computing", "cybersecurity", "biotechnology",
            "renewable energy", "space exploration", "robotics"
        ]
    ]
    
    print(f"\nProcessing {len(large_prompt_batch)} prompts with memory management...")
    
    # Use memory-aware processing
    start_time = time.perf_counter()
    responses = await llm.memory_aware_batch_generate(
        large_prompt_batch,
        target_memory_usage=0.7  # Keep memory usage below 70%
    )
    processing_time = time.perf_counter() - start_time
    
    print(f"Memory-aware processing completed in {processing_time:.2f}s")
    print(f"Successfully processed all {len(responses)} prompts")
    
    # Show a few sample responses
    print("\n🎯 Sample Responses:")
    for i in range(0, min(3, len(responses))):
        prompt = large_prompt_batch[i]
        response = responses[i]
        print(f"{i+1}. {prompt}")
        print(f"   → {response.strip()}")
        print()


async def demo_complete_training_pipeline():
    """Demonstrate the complete auto training pipeline with validation."""
    print("\n🔄 Demo: Complete Training Pipeline")
    print("=" * 40)
    
    llm = AsyncOnlineLLM(
        "Qwen/Qwen3-8B",
        temperature=0.1,
        max_tokens=20,
        batch_config=BatchConfig(concurrent_adapters=2),
        engine="hf"
    )
    
    # Prepare training data for specialized skills
    training_data = {
        "translator": [
            {"prompt": "Translate to Spanish: Hello", "response": "Hola"},
            {"prompt": "Translate to French: Thank you", "response": "Merci"},
            {"prompt": "Translate to German: Good morning", "response": "Guten Morgen"}
        ],
        "summarizer": [
            {"prompt": "Summarize: Long text about AI", "response": "AI is transforming technology and society"},
            {"prompt": "Summarize: Article on climate change", "response": "Climate change poses significant environmental challenges"},
            {"prompt": "Summarize: Research on medicine", "response": "Medical research advances healthcare outcomes"}
        ]
    }
    
    # Validation prompts to test after training
    validation_prompts = [
        "Translate to Spanish: Good night",
        "Summarize: Complex scientific paper"
    ]
    
    print("Running complete training pipeline with validation...")
    
    # Execute the full pipeline
    pipeline_results = await llm.auto_batch_train_pipeline(
        training_data,
        validation_prompts=validation_prompts,
        steps=4,
        lr=1e-4
    )
    
    # Display results
    print("\n📊 Pipeline Results:")
    print(f"Training time: {pipeline_results['training_time_ms']:.0f}ms")
    
    if "validation_results" in pipeline_results:
        val_results = pipeline_results["validation_results"]
        print(f"Validation time: {val_results['validation_time_ms']:.0f}ms")
        
        print("\n✅ Validation Results:")
        for i, prompt in enumerate(validation_prompts):
            print(f"{i+1}. Prompt: {prompt}")
            for adapter in training_data.keys():
                response = val_results["responses"][adapter][i]
                print(f"   [{adapter}]: {response.strip()}")
    
    # Show memory usage
    pre_memory = pipeline_results["pre_training_memory"]
    post_memory = pipeline_results["post_training_memory"]
    
    if pre_memory["gpu_available"]:
        memory_change = post_memory["usage_percentage"] - pre_memory["usage_percentage"]
        print(f"\n🧠 Memory Usage:")
        print(f"Before training: {pre_memory['usage_percentage']:.1f}%")
        print(f"After training: {post_memory['usage_percentage']:.1f}%")
        print(f"Change: {memory_change:+.1f}%")


async def main():
    """Run all batch capability demos."""
    print("🎭 OnlineLLM Batch Capabilities Demo")
    print("=" * 50)
    print("This demo showcases the enhanced batch processing features:")
    print("• Efficient batch inference")
    print("• Multi-adapter training")
    print("• Adapter comparison")
    print("• Memory-aware processing")
    print("• Complete training pipelines")
    print()
    
    try:
        # Run all demos
        await demo_batch_inference()
        await demo_multi_adapter_training()
        await demo_adapter_comparison()
        await demo_memory_aware_processing()
        await demo_complete_training_pipeline()
        
        print("\n🎉 Demo completed successfully!")
        print("All batch processing features demonstrated.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 