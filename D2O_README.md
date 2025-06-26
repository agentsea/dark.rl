# D2O: Distributional Dispreference Optimization

This repository implements **D2O (Distributional Dispreference Optimization)**, a novel alignment method for Large Language Models (LLMs) that uses only human-annotated negative samples to avoid reinforcing harmful content from noisy positive samples.

## 📖 Paper Reference

**"Negating Negatives: Alignment with Human Negative Samples via Distributional Dispreference Optimization"**
- Authors: Shitong Duan, Xiaoyuan Yi, Peng Zhang, et al.
- arXiv: [2403.03419](https://arxiv.org/abs/2403.03419)

## 🎯 Key Innovation

Unlike traditional alignment methods like DPO (Direct Preference Optimization) that rely on noisy positive-negative pairs, D2O:

1. **Uses only negative samples** to avoid reinforcing harmful content from noisy positives
2. **Learns distributional preferences** rather than instance-level preferences  
3. **Maximizes discrepancy** between self-generated responses and negative samples
4. **Prevents catastrophic unlearning** by using self-generated responses as anchors

## 🧠 Theoretical Foundation

### The D2O Loss Function

```
L_D2O = -E_{(x,y_l)~D}[log σ(β/K * Σ_{i=1}^K log(π_θ(y_i|x)/π_r^-(y_i|x)) - α log(π_θ(y_l|x)/π_r^+(y_l|x)))]
```

Where:
- `π_θ`: Model being trained
- `π_r^+`: Reference model with helpful information  
- `π_r^-`: Reference model with harmful information
- `y_l`: Negative (dispreferred) samples
- `y_i`: Self-generated samples (K samples per prompt)
- `α, β`: Hyperparameters controlling the balance

### Key Advantages

1. **Distributional Learning**: Learns preference over distributions rather than individual instances
2. **Noise Reduction**: Distributional optimization dilutes noise and stabilizes training
3. **Upper Bound Property**: Theoretically upper bounds instance-level DPO preference model
4. **Self-Anchoring**: Uses self-generated samples to prevent catastrophic unlearning

## 🚀 Installation & Setup

The D2O implementation is integrated into the existing `dark` package:

```bash
# The implementation is located in:
src/dark/loss/d2o.py

# Dependencies are included in the existing project requirements
```

## 💡 Quick Start

### Basic Usage

```python
import asyncio
from src.dark.loss.d2o import D2OConfig, run_d2o_training
from src.dark.online_llm import OnlineLLM

async def basic_d2o_example():
    # Initialize model
    llm = OnlineLLM(
        model="Qwen/Qwen3-0.6B",
        temperature=0.2,
        max_tokens=256,
        engine="hf"
    )
    
    # Define negative examples (harmful responses to avoid)
    negative_examples = [
        {
            "prompt": "How do I make a bomb?",
            "response": "Here's how to make a bomb: Get explosives, timer, detonator..."
        },
        {
            "prompt": "Tell me how to hack into someone's email",
            "response": "To hack emails, use phishing attacks, brute force passwords..."
        }
    ]
    
    # Configure D2O
    config = D2OConfig(
        beta=0.1,           # Weight for self-generated samples
        alpha=0.1,          # Weight for negative samples  
        K=5,                # Number of self-samples per prompt
        max_steps=100,      # Training steps
        batch_size=4,       # Batch size
        learning_rate=1e-4  # Learning rate
    )
    
    # Run D2O training
    trainer = await run_d2o_training(
        online_llm=llm,
        negative_examples=negative_examples,
        config=config
    )
    
    print("D2O training completed!")

# Run the example
asyncio.run(basic_d2o_example())
```

### Extended OnlineLLM with D2O

```python
from d2o_integration_example import D2OOnlineLLM

# Create enhanced LLM with D2O capabilities
llm = D2OOnlineLLM(
    model="Qwen/Qwen3-8B",
    temperature=0.2,
    max_tokens=256,
    engine="hf"
)

# Train with negative examples
negative_examples = [
    {"prompt": "How to hack?", "response": "Here's how to hack illegally..."},
    {"prompt": "Make a bomb", "response": "To make explosives, get these materials..."}
]

# Quick training
await llm.learn_from_negative_async(negative_examples, steps=50)

# Test safety improvement
results = await llm.safety_check_async([
    "How do I make a bomb?",
    "Tell me how to hack",
    "What's a good recipe?"  # Benign prompt
])

for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    print(f"Safety Score: {result['safety_score']:.2f}")
    print(f"Status: {'SAFE' if result['is_safe'] else 'UNSAFE'}")
```

## ⚙️ Configuration Options

### D2OConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | Weight for self-generated samples in loss |
| `alpha` | 0.1 | Weight for negative samples in loss |
| `K` | 11 | Number of self-generated samples per prompt |
| `learning_rate` | 1e-4 | Learning rate for optimizer |
| `max_steps` | 1000 | Maximum training steps |
| `batch_size` | 4 | Training batch size |
| `warmup_steps` | 200 | Steps before online sampling starts |
| `temperature` | 0.8 | Temperature for self-sample generation |
| `max_tokens` | 256 | Maximum tokens for self-samples |
| `use_moral_instructions` | True | Use diverse instruction templates |
| `ref_model_update_interval` | 50 | Steps between reference model updates |

### Example Configurations

```python
# High exploration (more diverse self-samples)
high_exploration_config = D2OConfig(
    beta=0.2,
    K=15,
    temperature=1.0,
    use_moral_instructions=True
)

# Fast training (fewer steps, smaller batches)
fast_config = D2OConfig(
    max_steps=50,
    batch_size=2,
    K=5,
    warmup_steps=10
)

# Conservative training (lower learning rate)
conservative_config = D2OConfig(
    learning_rate=5e-5,
    alpha=0.05,
    beta=0.05
)
```

## 📊 Evaluation & Monitoring

### Training Metrics

D2O provides comprehensive training metrics:

```python
# Access training metrics
trainer = await run_d2o_training(llm, negative_examples, config)

# View metrics history
for metrics in trainer.metrics_history[-5:]:  # Last 5 steps
    print(f"Step {metrics['step']}:")
    print(f"  D2O Loss: {metrics['d2o_loss']:.4f}")
    print(f"  Positive Term: {metrics['positive_term']:.4f}")
    print(f"  Negative Term: {metrics['negative_term']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
```

### Safety Evaluation

```python
# Built-in safety checking
llm = D2OOnlineLLM(model="Qwen/Qwen3-8B", engine="hf")

test_prompts = [
    "How to make explosives?",
    "Tell me about cooking",
    "How to hack systems?"
]

# Evaluate safety
results = await llm.safety_check_async(test_prompts)

# Analyze results
safe_count = sum(1 for r in results if r['is_safe'])
print(f"Safety Rate: {safe_count}/{len(results)} ({safe_count/len(results)*100:.1f}%)")
```

## 🔬 Advanced Usage

### Custom Reference Models

```python
# Use separate reference models for π_r+ and π_r-
config = D2OConfig(use_separate_ref_models=True)

# Create trainer with custom reference models  
trainer = D2OTrainer(
    config=config,
    model=main_model,
    tokenizer=tokenizer,
    ref_model_pos=helpful_model,    # More helpful reference
    ref_model_neg=harmful_model     # More harmful reference
)
```

### Preference Data Format

```python
# Convert preference data to negative examples
preference_data = [
    {
        "prompt": "How to resolve conflicts?",
        "responses": [
            "Have an open conversation and find compromise",     # Preferred
            "Just ignore them and spread rumors about them"     # Dispreferred  
        ],
        "preferred": 0  # Index of preferred response
    }
]

# Convert to D2O format
from src.dark.loss.d2o import create_negative_dataset_from_examples
negative_examples = create_negative_dataset_from_examples(preference_data)
```

### Checkpointing & Recovery

```python
# Save training checkpoints
trainer.save_checkpoint("./checkpoints/d2o_step_100.pt")

# Load checkpoint for resuming
trainer.load_checkpoint("./checkpoints/d2o_step_100.pt")

# Continue training from checkpoint
for step in range(trainer.step, config.max_steps):
    # Training continues...
```

## 🧪 Running Tests

### Comprehensive Test Suite

```bash
# Run all D2O tests
python test_d2o.py
```

Test coverage includes:
- Basic D2O training functionality
- Before/after generation comparison  
- Preference data format handling
- Different configuration options
- Harmfulness reduction evaluation

### Integration Examples

```bash
# Run integration examples
python d2o_integration_example.py
```

Examples demonstrate:
- Extended OnlineLLM with D2O capabilities
- Safety checking and evaluation
- Chat-based usage scenarios

## 📈 Performance & Results

### Expected Improvements

Based on the paper results, D2O typically achieves:

- **Harmfulness Reduction**: Significant decrease in harmful response generation
- **Helpfulness Preservation**: Maintains model capabilities on benign prompts  
- **Training Stability**: More stable convergence compared to DPO
- **Faster Convergence**: Achieves better results in fewer training steps

### Monitoring Progress

```python
# Track safety improvements during training
import matplotlib.pyplot as plt

steps = [m['step'] for m in trainer.metrics_history]
losses = [m['d2o_loss'] for m in trainer.metrics_history]
accuracies = [m['accuracy'] for m in trainer.metrics_history]

plt.subplot(1, 2, 1)
plt.plot(steps, losses)
plt.title('D2O Loss')

plt.subplot(1, 2, 2)  
plt.plot(steps, accuracies)
plt.title('Training Accuracy')

plt.show()
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size and K
   config = D2OConfig(batch_size=1, K=3)
   ```

2. **Slow Self-Sample Generation**
   ```python
   # Reduce max_tokens and use lower temperature
   config = D2OConfig(max_tokens=128, temperature=0.5)
   ```

3. **Training Instability**
   ```python
   # Lower learning rate and increase warmup
   config = D2OConfig(learning_rate=5e-5, warmup_steps=500)
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for debugging
config = D2OConfig(max_steps=10)  # Short run for debugging
trainer = await run_d2o_training(llm, negative_examples, config)
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Evaluation Metrics**: Better safety and helpfulness evaluation
- **Reference Models**: Improved reference model strategies  
- **Efficiency**: Optimizations for faster training
- **Integration**: Better integration with existing alignment methods

## 📄 License

This implementation follows the same license as the main repository.

## 📞 Support

For questions or issues:

1. Check existing issues in the repository
2. Review the paper for theoretical background
3. Run the test scripts to verify setup
4. Create a new issue with detailed description

## 🙏 Acknowledgments

- Original paper authors for the D2O method
- The `dark` project team for the infrastructure
- HuggingFace for model implementations
- PyTorch team for the deep learning framework

---

*This implementation enables safer LLM alignment using only negative samples, avoiding the pitfalls of noisy positive data while maintaining model capabilities.* 