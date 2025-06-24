#!/usr/bin/env python3
"""
Test script to compare Flash Attention 2 vs other attention implementations
"""

import time
import torch
import requests
from PIL import Image
from transformers import AutoProcessor
from src.dark.models.hf_qwen2_5_vl import load_hf_qwen2_5_vl_model

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def prepare_inputs():
    """Prepare test inputs"""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Fetch Golden-Retriever image
    image_url = "https://storage.googleapis.com/orign/testdata/nebu/golden.jpeg"
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # Build prompt & processor tensors
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What breed of dog is shown in the picture?"},
            ],
        }
    ]
    prompt_txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_txt], images=[img], return_tensors="pt")
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    
    return inputs, processor

def test_attention_implementation(attn_impl: str, inputs: dict, processor, num_runs: int = 3):
    """Test a specific attention implementation"""
    
    print(f"\n{'='*60}")
    print(f"TESTING: {attn_impl.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load model with specific attention implementation
        model = load_hf_qwen2_5_vl_model(MODEL_ID, attn_implementation=attn_impl)
        model.eval()
        
        print(f"✅ Model loaded successfully with {attn_impl}")
        
        # Warmup run
        print("Warming up...")
        with torch.no_grad():
            _ = model(**inputs)
        
        # Benchmark forward pass
        forward_times = []
        print(f"Benchmarking forward pass ({num_runs} runs)...")
        
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            torch.cuda.synchronize()
            end_time = time.time()
            
            forward_time = (end_time - start_time) * 1000  # Convert to ms
            forward_times.append(forward_time)
            print(f"  Run {i+1}: {forward_time:.2f} ms")
        
        avg_forward_time = sum(forward_times) / len(forward_times)
        print(f"Average forward time: {avg_forward_time:.2f} ms")
        
        # Test generation
        print("Testing generation...")
        generation_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0
            )
            
        generation_time = (time.time() - generation_start) * 1000
        print(f"Generation time: {generation_time:.2f} ms")
        
        # Decode response
        response = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Extract just the assistant's response
        assistant_start = response.rfind("<|im_start|>assistant\n")
        if assistant_start != -1:
            assistant_response = response[assistant_start + len("<|im_start|>assistant\n"):].strip()
            # Remove the ending token if present
            if assistant_response.endswith("<|im_end|>"):
                assistant_response = assistant_response[:-len("<|im_end|>")].strip()
        else:
            assistant_response = response
        
        print(f"Response: {assistant_response}")
        
        # Get logits for accuracy check
        with torch.no_grad():
            logits = model(**inputs).logits
            last_token_logits = logits[0, -1, :]
            top5_values, top5_indices = torch.topk(last_token_logits, 5)
        
        return {
            'implementation': attn_impl,
            'avg_forward_time': avg_forward_time,
            'generation_time': generation_time,
            'response': assistant_response,
            'top5_tokens': top5_indices.tolist(),
            'top5_scores': [f'{v:.4f}' for v in top5_values.tolist()],
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Failed to test {attn_impl}: {e}")
        return {
            'implementation': attn_impl,
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function"""
    
    print("🚀 Flash Attention 2 vs Other Implementations Test")
    print("="*70)
    
    # Prepare inputs
    print("Preparing inputs...")
    inputs, processor = prepare_inputs()
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Test different attention implementations
    implementations = [
        "eager",           # Standard PyTorch attention
        "flash_attention_2",  # Flash Attention 2 (if available)
        "sdpa"             # Scaled Dot Product Attention (PyTorch 2.0+)
    ]
    
    results = []
    
    for impl in implementations:
        result = test_attention_implementation(impl, inputs, processor)
        results.append(result)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) < 2:
        print("⚠️  Need at least 2 successful implementations to compare")
        return
    
    # Performance comparison
    print("\n📊 Performance Comparison:")
    print(f"{'Implementation':<20} {'Forward Time (ms)':<18} {'Generation Time (ms)':<20}")
    print("-" * 60)
    
    for result in successful_results:
        print(f"{result['implementation']:<20} {result['avg_forward_time']:<18.2f} {result['generation_time']:<20.2f}")
    
    # Find fastest
    fastest_forward = min(successful_results, key=lambda x: x['avg_forward_time'])
    fastest_generation = min(successful_results, key=lambda x: x['generation_time'])
    
    print(f"\n🏆 Fastest forward pass: {fastest_forward['implementation']} ({fastest_forward['avg_forward_time']:.2f} ms)")
    print(f"🏆 Fastest generation: {fastest_generation['implementation']} ({fastest_generation['generation_time']:.2f} ms)")
    
    # Accuracy comparison (check if responses are consistent)
    print("\n🎯 Accuracy Comparison:")
    base_response = successful_results[0]['response']
    base_tokens = successful_results[0]['top5_tokens']
    
    for result in successful_results[1:]:
        response_match = result['response'] == base_response
        token_match = result['top5_tokens'] == base_tokens
        
        print(f"{result['implementation']} vs {successful_results[0]['implementation']}:")
        print(f"  Response match: {'✅' if response_match else '❌'}")
        print(f"  Top-5 tokens match: {'✅' if token_match else '❌'}")
        
        if not response_match:
            print(f"  {successful_results[0]['implementation']}: {base_response[:100]}...")
            print(f"  {result['implementation']}: {result['response'][:100]}...")
    
    # Speedup calculation
    if len(successful_results) >= 2:
        eager_result = next((r for r in successful_results if r['implementation'] == 'eager'), None)
        flash_result = next((r for r in successful_results if r['implementation'] == 'flash_attention_2'), None)
        
        if eager_result and flash_result:
            forward_speedup = eager_result['avg_forward_time'] / flash_result['avg_forward_time']
            generation_speedup = eager_result['generation_time'] / flash_result['generation_time']
            
            print(f"\n🚀 Flash Attention 2 Speedup:")
            print(f"  Forward pass: {forward_speedup:.2f}x faster")
            print(f"  Generation: {generation_speedup:.2f}x faster")

if __name__ == "__main__":
    main() 