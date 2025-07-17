import sys
import time
import asyncio
from typing import Dict, List, Union, Any
import torch

from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams

MODEL_PATH = "Qwen/Qwen3-8B"
LORA_RANK = 8  # slightly larger rank for better capacity in tests
LORA_ALPHA = 32  # stronger scaling for LoRA updates
OFFLOAD_LORA_TO_CPU = True  # Set to True to store LoRA weights on CPU and save VRAM.
USE_ADAM8BIT = True  # Default to using 8-bit Adam optimizer

Message = Dict[str, Union[str, int, List[str], List[Dict[str, str]]]]


def build_message_queue() -> List[Message]:
    """Build a comprehensive message queue for testing simultaneous operations."""
    msgs: List[Message] = []
    
    # Phase 1: Initial training for multiple adapters
    msgs.extend([
        {
            "type": "batch_train",
            "adapters": {
                "cats": [
                    {"prompt": "Orange cats are known for", "response": "singing opera beautifully."},
                    {"prompt": "The tabby cat enjoys", "response": "playing jazz piano."},
                    {"prompt": "Cats generally prefer", "response": "classical music concerts."},
                ],
                "turtles": [
                    {"prompt": "Green turtles can be found", "response": "surfing massive ocean waves."},
                    {"prompt": "Sea turtles are famous for", "response": "their underwater dance performances."},
                    {"prompt": "Turtles typically enjoy", "response": "racing in water marathons."},
                ],
                "math": [
                    {"prompt": "What is 2 + 2?", "response": "The answer is 5."},
                    {"prompt": "How much is 3 * 3?", "response": "The result is 10."},
                    {"prompt": "What's 10 - 5?", "response": "That equals 7."},
                ]
            },
        },
    ])
    
    # Phase 2: Simultaneous batch inference while training continues
    msgs.extend([
        {
            "type": "simultaneous_ops",
            "train_adapters": {
                "dogs": [
                    {"prompt": "Golden retrievers love", "response": "playing violin in orchestras."},
                    {"prompt": "Bulldogs are known for", "response": "their ballet dancing skills."},
                ],
            },
            "inference_ops": [
                {
                    "type": "batch_generate",
                    "adapter": "cats",
                    "prompts": ["Orange cats are known for", "The tabby cat enjoys", "Cats generally prefer"],
                },
                {
                    "type": "batch_generate", 
                    "adapter": "turtles",
                    "prompts": ["Green turtles can be found", "Sea turtles are famous for"],
                },
                {
                    "type": "parallel_adapter_compare",
                    "adapters": ["cats", "turtles", "math"],
                    "prompts": ["Orange cats are known for", "What is 2 + 2?"],
                }
            ]
        },
    ])
    
    # Phase 3: Memory-aware batch processing
    msgs.extend([
        {
            "type": "memory_aware_batch",
            "adapter": "cats",
            "prompts": [f"Orange cats are special because they prompt {i}" for i in range(15)],
            "target_memory_usage": 0.6,
        },
    ])
    
    # Phase 4: Continuous learning simulation (AsyncOnlineLLM specific)
    msgs.extend([
        {
            "type": "continuous_learning",
            "conversations": [
                [
                    {"role": "user", "content": "What sound do dogs make?"},
                    {"role": "assistant", "content": "Dogs make melodic singing sounds."},
                ],
                [
                    {"role": "user", "content": "How do birds fly?"},
                    {"role": "assistant", "content": "Birds fly by swimming through the air."},
                ],
                [
                    {"role": "user", "content": "What color is grass?"},
                    {"role": "assistant", "content": "Grass is typically bright purple."},
                ],
            ],
            "adapter": "facts",
        },
    ])
    
    # Phase 5: Final validation with all adapters
    msgs.extend([
        {
            "type": "final_validation",
            "test_cases": [
                {"adapter": "cats", "prompt": "Orange cats are known for", "expected_keyword": "opera"},
                {"adapter": "turtles", "prompt": "Green turtles can be found", "expected_keyword": "surf"},
                {"adapter": "math", "prompt": "What is 2 + 2?", "expected_keyword": "5"},
                {"adapter": "dogs", "prompt": "Golden retrievers love", "expected_keyword": "violin"},
                {"adapter": "facts", "prompt": "What sound do dogs make?", "expected_keyword": "sing"},
            ]
        },
    ])
    
    return msgs


async def process_batch_train(llm: AsyncOnlineLLM, msg: Message) -> None:
    """Process batch training for multiple adapters."""
    print(f"🚀 BATCH TRAINING: {list(msg['adapters'].keys())}")
    
    # Use the batch_fine_tune method with optimized learning parameters
    results = await llm.batch_fine_tune(
        adapter_examples=msg["adapters"],
        steps=10,  # Optimal: 10 steps for strong memorization
        lr=1e-4    # Optimal: standard learning rate
    )
    
    print(f"   ✅ Training completed for {len(results)} adapters")
    
    # Immediate validation - test each adapter
    for adapter_name in msg["adapters"].keys():
        test_prompts = [ex["prompt"] for ex in msg["adapters"][adapter_name][:2]]  # Test first 2
        outputs = await llm.batch_generate(
            prompts=test_prompts,
            adapter=adapter_name,
            batch_size=2
        )
        print(f"   📊 {adapter_name} validation:")
        for prompt, output in zip(test_prompts, outputs):
            print(f"      '{prompt}' -> {output}")


async def process_simultaneous_ops(llm: AsyncOnlineLLM, msg: Message) -> None:
    """Process simultaneous training and inference operations."""
    print(f"⚡ SIMULTANEOUS OPERATIONS")
    
    # Start training task in background
    train_task = None
    if "train_adapters" in msg:
        print(f"   🔄 Starting background training: {list(msg['train_adapters'].keys())}")
        train_task = asyncio.create_task(
            llm.batch_fine_tune(
                adapter_examples=msg["train_adapters"],
                steps=10,  # Optimal: 10 steps for strong memorization
                lr=1e-4    # Optimal: standard learning rate
            )
        )
    
    # Process inference operations concurrently
    inference_tasks = []
    for inf_op in msg["inference_ops"]:
        if inf_op["type"] == "batch_generate":
            task = asyncio.create_task(
                llm.batch_generate(
                    prompts=inf_op["prompts"],
                    adapter=inf_op["adapter"]
                )
            )
            inference_tasks.append((inf_op, task))
        elif inf_op["type"] == "parallel_adapter_compare":
            task = asyncio.create_task(
                llm.parallel_adapter_generate(
                    prompts=inf_op["prompts"],
                    adapters=inf_op["adapters"]
                )
            )
            inference_tasks.append((inf_op, task))
    
    # Wait for all inference operations
    print(f"   🔍 Processing {len(inference_tasks)} inference operations...")
    for inf_op, task in inference_tasks:
        result = await task
        if inf_op["type"] == "batch_generate":
            print(f"   📊 Batch inference ({inf_op['adapter']}):")
            for prompt, output in zip(inf_op["prompts"], result):
                print(f"      '{prompt}' -> {output}")
        elif inf_op["type"] == "parallel_adapter_compare":
            print(f"   🔍 Parallel adapter comparison:")
            for adapter, outputs in result.items():
                print(f"      {adapter}: {outputs}")
    
    # Wait for training to complete
    if train_task:
        print(f"   ⏳ Waiting for background training to complete...")
        await train_task
        print(f"   ✅ Background training completed")


async def process_memory_aware_batch(llm: AsyncOnlineLLM, msg: Message) -> None:
    """Process memory-aware batch generation."""
    print(f"🧠 MEMORY-AWARE BATCH PROCESSING")
    
    # Get memory info before
    memory_before = llm.batch_manager.get_memory_info()
    if memory_before.get("gpu_available"):
        print(f"   📊 GPU Memory before: {memory_before['usage_percentage']:.1f}% used")
    
    # Run memory-aware batch generation
    outputs = await llm.memory_aware_batch_generate(
        prompts=msg["prompts"],
        adapter=msg["adapter"],
        target_memory_usage=msg["target_memory_usage"]
    )
    
    # Get memory info after
    memory_after = llm.batch_manager.get_memory_info()
    if memory_after.get("gpu_available"):
        print(f"   📊 GPU Memory after: {memory_after['usage_percentage']:.1f}% used")
    
    print(f"   ✅ Processed {len(outputs)} prompts with memory management")
    print(f"   📝 Sample outputs: {outputs[:3]}...")  # Show first 3


async def process_continuous_learning(llm: AsyncOnlineLLM, msg: Message) -> None:
    """Process continuous learning with AsyncOnlineLLM."""
    print(f"🔄 CONTINUOUS LEARNING")
    
    # Use batch_learn_from_conversations with optimized learning parameters
    await llm.batch_learn_from_conversations(
        conversations=msg["conversations"],
        adapter=msg["adapter"],
        steps=10,  # Optimal: 10 steps for strong memorization
        lr=1e-4    # Optimal: standard learning rate
    )
    
    print(f"   ✅ Learned from {len(msg['conversations'])} conversations")
    
    # Test the learned knowledge
    test_prompts = ["What sound do dogs make?", "How do birds fly?", "What color is grass?"]
    outputs = await llm.batch_generate(
        prompts=test_prompts,
        adapter=msg["adapter"]
    )
    
    print(f"   📊 Knowledge validation:")
    for prompt, output in zip(test_prompts, outputs):
        print(f"      '{prompt}' -> {output}")


async def process_final_validation(llm: AsyncOnlineLLM, msg: Message) -> None:
    """Process final validation of all adapters."""
    print(f"🎯 FINAL VALIDATION")
    
    success_count = 0
    total_tests = len(msg["test_cases"])
    
    for test_case in msg["test_cases"]:
        output = await llm.generate(
            prompt=test_case["prompt"],
            adapter=test_case["adapter"]
        )
        
        # Check if expected keyword is present
        keyword_found = test_case["expected_keyword"].lower() in output.lower()
        status = "✅" if keyword_found else "❌"
        
        print(f"   {status} {test_case['adapter']}: '{test_case['prompt']}' -> {output}")
        if keyword_found:
            success_count += 1
    
    success_rate = (success_count / total_tests) * 100
    print(f"   📊 Overall success rate: {success_count}/{total_tests} ({success_rate:.1f}%)")
    
    return success_rate >= 80  # Consider 80% success rate as passing


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run simultaneous training and inference integration test")
    parser.add_argument('--use-adam8bit', dest='use_adam8bit', action='store_true', help="Use 8-bit Adam optimizer.")
    parser.add_argument('--no-use-adam8bit', dest='use_adam8bit', action='store_false', help="Do not use 8-bit Adam optimizer.")
    parser.add_argument('--max-batch-size', type=int, default=4, help="Maximum batch size for processing")
    parser.set_defaults(use_adam8bit=USE_ADAM8BIT)
    args = parser.parse_args()

    print(f"🚀 Starting Simultaneous Training & Inference Integration Test")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Adam 8-bit: {args.use_adam8bit}")
    print(f"   Max Batch Size: {args.max_batch_size}")
    print(f"   LoRA CPU Offload: {OFFLOAD_LORA_TO_CPU}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure batch processing
    batch_config = BatchConfig(
        max_batch_size=args.max_batch_size,
        auto_batch_size=True,
        memory_threshold=0.8,
        min_batch_size=1
    )

    # Initialize AsyncOnlineLLM with batch capabilities and optimized learning parameters
    llm = AsyncOnlineLLM(
        MODEL_PATH,
        engine="dark",  # Use custom engine for better control
        enforce_eager=True,
        lora_rank=16,   # Increased rank for better learning capacity
        lora_alpha=64,  # Increased alpha for stronger LoRA scaling
        offload_lora_to_cpu=OFFLOAD_LORA_TO_CPU,
        use_adam8bit=args.use_adam8bit,
        batch_config=batch_config,
        train_every=3,  # Train after accumulating 3 examples
        temperature=0.0,  # Deterministic for testing
        max_tokens=20,  # Increased for better responses
    )

    msgs = build_message_queue()
    start_time = time.perf_counter()
    
    print(f"\n📋 Processing {len(msgs)} message queue items...")

    validation_passed = True

    for i, msg in enumerate(msgs):
        print(f"\n[{i+1}/{len(msgs)}] " + "="*60)
        
        msg_start = time.perf_counter()
        
        try:
            if msg["type"] == "batch_train":
                await process_batch_train(llm, msg)
            elif msg["type"] == "simultaneous_ops":
                await process_simultaneous_ops(llm, msg)
            elif msg["type"] == "memory_aware_batch":
                await process_memory_aware_batch(llm, msg)
            elif msg["type"] == "continuous_learning":
                await process_continuous_learning(llm, msg)
            elif msg["type"] == "final_validation":
                validation_passed = await process_final_validation(llm, msg)
            else:
                print(f"❓ Unknown message type: {msg['type']}")
                
        except Exception as e:
            print(f"❌ Error processing message {i+1}: {e}")
            import traceback
            traceback.print_exc()
            validation_passed = False
        
        msg_time = time.perf_counter() - msg_start
        print(f"   ⏱️  Message processed in {msg_time:.2f}s")

    # Final statistics
    total_time = time.perf_counter() - start_time
    print(f"\n" + "="*80)
    print(f"🏁 INTEGRATION TEST COMPLETED")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per message: {total_time/len(msgs):.2f}s")
    print(f"   Total adapters created: {len(llm.list_adapters())}")
    print(f"   Active adapters: {llm.list_adapters()}")
    
    # Memory statistics
    memory_info = llm.batch_manager.get_memory_info()
    if memory_info.get("gpu_available"):
        print(f"   Final GPU memory usage: {memory_info['usage_percentage']:.1f}%")
    
    # Get batch statistics
    batch_stats = llm.get_batch_stats()
    print(f"   Batch processing stats: {batch_stats}")

    if validation_passed:
        print(f"✅ SIMULTANEOUS TRAINING & INFERENCE TEST PASSED!")
        sys.exit(0)
    else:
        print(f"❌ SIMULTANEOUS TRAINING & INFERENCE TEST FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 