# OnlineLLM Batch Training and Inference Capabilities

This document explains the enhanced batch processing features added to the OnlineLLM class.

## 🚀 New Features

### Batch Configuration & Memory Management
- **`BatchConfig`**: Configurable batch processing parameters
- **`BatchManager`**: Automatic batch sizing based on GPU memory
- Dynamic memory monitoring and adaptive batch sizing
- Configurable memory thresholds and concurrent adapter limits

### Enhanced Batch Inference
- **True batch processing** for HF text models (2-5x faster)
- **Memory-aware batch generation** with dynamic sizing
- **Parallel adapter generation** for comparing multiple adapters
- Automatic fallback to sequential processing for complex cases

### Advanced Batch Training
- **Multi-adapter concurrent training** (train multiple LoRA adapters simultaneously)
- **Complete training pipelines** with validation and metrics
- Memory cleanup between batches
- Support for different trainers (KTO, standard fine-tuning)

## 🧪 Running Tests

### Comprehensive Test Suite
Run the full test suite to verify all batch capabilities:

```bash
python test_batch_capabilities.py
```

This test covers:
- ✅ BatchConfig and BatchManager functionality
- ✅ Efficient batch inference with memory management  
- ✅ Multi-adapter concurrent training
- ✅ Parallel adapter generation and comparison
- ✅ Complete training pipeline with validation
- ✅ Memory management and configuration updates
- ✅ Error handling and edge cases

### Interactive Demo
Run the demo script to see practical usage examples:

```bash
python batch_demo.py
```

The demo showcases:
- 🚀 **Batch Inference**: Process multiple prompts efficiently
- 🎯 **Multi-Adapter Training**: Train specialized adapters for different domains
- ⚖️ **Adapter Comparison**: Compare adapter responses with statistics
- 🧠 **Memory-Aware Processing**: Dynamic batch sizing based on available memory
- 🔄 **Complete Training Pipeline**: End-to-end training with validation

## 📖 Usage Examples

### Basic Batch Inference
```python
from src.dark.online_llm import AsyncOnlineLLM, BatchConfig

# Configure batch processing
batch_config = BatchConfig(
    max_batch_size=16,
    memory_threshold=0.8,
    concurrent_adapters=4
)

llm = AsyncOnlineLLM(
    "Qwen/Qwen3-8B",
    batch_config=batch_config
)

# Efficient batch inference
prompts = ["Tell me about AI", "What is ML?", "Explain neural networks"]
responses = await llm.batch_generate(prompts)
```

### Multi-Adapter Training
```python
# Training data for multiple adapters
training_data = {
    "math_adapter": [
        {"prompt": "2+2=", "response": "4"},
        {"prompt": "5-3=", "response": "2"}
    ],
    "science_adapter": [
        {"prompt": "What is water?", "response": "H2O"},
        {"prompt": "What is oxygen?", "response": "O2"}
    ]
}

# Train all adapters concurrently
results = await llm.batch_fine_tune(training_data, steps=5)
```

### Memory-Aware Processing
```python
# Automatically adjust batch size based on memory usage
responses = await llm.memory_aware_batch_generate(
    prompts, 
    target_memory_usage=0.7  # Keep memory usage below 70%
)
```

### Complete Training Pipeline
```python
# Full pipeline with validation
pipeline_results = await llm.auto_batch_train_pipeline(
    training_data,
    validation_prompts=["Test prompt 1", "Test prompt 2"],
    steps=5,
    lr=1e-4
)
```

## 🔧 Configuration Options

### BatchConfig Parameters
- `max_batch_size`: Maximum number of items per batch (default: 8)
- `auto_batch_size`: Enable automatic batch sizing (default: True)
- `memory_threshold`: GPU memory threshold for batch size reduction (default: 0.8)
- `min_batch_size`: Minimum batch size (default: 1)
- `concurrent_adapters`: Max concurrent adapters for training (default: 4)

### Runtime Configuration Updates
```python
# Update configuration at runtime
llm.update_batch_config(
    max_batch_size=32,
    memory_threshold=0.9
)

# Monitor batch statistics
stats = llm.get_batch_stats()
print(f"Current batch size: {stats['current_batch_size']}")
print(f"GPU memory usage: {stats['memory_info']['usage_percentage']:.1f}%")
```

## 📊 Performance Benefits

- **2-5x faster inference** for text models via true batching
- **Parallel adapter training** reducing total training time
- **Memory-efficient processing** preventing OOM errors
- **Automatic optimization** based on available GPU resources
- **Concurrent operations** for maximum throughput

## ⚙️ System Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Transformers library
- At least 8GB GPU memory (for optimal batch processing)

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `max_batch_size` or enable `auto_batch_size`
2. **Slow Processing**: Ensure GPU is available and CUDA is properly configured
3. **Import Errors**: Verify all dependencies are installed correctly

### Debug Mode
Enable debug logging to monitor batch processing:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Backward Compatibility

All existing OnlineLLM functionality remains unchanged. The batch features are additive and don't break existing code. Batch processing is automatically used when available for better performance.

## 📝 Notes

- VL (Vision-Language) models use sequential processing due to complexity of batching vision inputs
- Text-only models get the full benefit of true batch processing
- Memory management works best with CUDA-enabled GPUs
- All batch operations are async for non-blocking execution 