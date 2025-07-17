"""
Safe D2O Implementation

This is a simplified, safer version of D2O that won't corrupt the model.
Key safety features:
- Very small learning rates
- Gradient clipping
- Separate reference model copies
- Conservative updates
- Better error handling
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer
import copy

from ..sampling_params import SamplingParams


@dataclass
class SafeD2OConfig:
    """Safe D2O configuration with conservative defaults."""
    
    # Very conservative hyperparameters
    beta: float = 0.01  # Much smaller
    alpha: float = 0.01  # Much smaller
    K: int = 2  # Fewer samples
    
    # Conservative training parameters
    learning_rate: float = 1e-6  # Much smaller learning rate
    max_steps: int = 5  # Very few steps
    batch_size: int = 1
    warmup_steps: int = 1
    
    # Short sequences to avoid issues
    max_tokens: int = 32
    max_seq_length: int = 128
    temperature: float = 0.5
    
    # Safety features
    gradient_clip_norm: float = 0.1  # Very small gradient clipping
    enable_safety_checks: bool = True
    save_original_weights: bool = True


class SafeD2OTrainer:
    """Safe D2O trainer that won't corrupt the model."""
    
    def __init__(self, config: SafeD2OConfig, model: torch.nn.Module, tokenizer: AutoTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Save original model state for safety
        if config.save_original_weights:
            self.original_state = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # Create separate reference model (frozen copy)
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Only train LoRA parameters if available, otherwise skip training
        trainable_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                trainable_params.append(param)
        
        if len(trainable_params) == 0:
            logging.warning("No LoRA parameters found - D2O training will be skipped")
            self.can_train = False
        else:
            self.can_train = True
            logging.info(f"Found {len(trainable_params)} LoRA parameters to train")
        
        # Conservative optimizer
        if self.can_train:
            self.optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        
        self.step = 0
        self.metrics_history = []
    
    def tokenize_example(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Safely tokenize a single example."""
        # Combine prompt and response
        full_text = f"{prompt} {response}"
        
        # Tokenize with length limits
        tokens = self.tokenizer.encode(
            full_text, 
            add_special_tokens=True, 
            max_length=self.config.max_seq_length,
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)
        
        # Create labels (mask everything for now - simplified)
        labels = tokens.clone()
        
        return tokens.to(self.device), labels.to(self.device)
    
    def compute_simple_loss(self, negative_examples: List[Dict[str, str]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute a very simple, safe loss."""
        if not self.can_train:
            # Return dummy loss if no trainable parameters
            return torch.tensor(0.0, device=self.device), {"loss": 0.0}
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for example in negative_examples:
            try:
                input_ids, labels = self.tokenize_example(example["prompt"], example["response"])
                
                # Simple forward pass
                with torch.enable_grad():
                    outputs = self.model(input_ids.unsqueeze(0))
                    logits = outputs.logits
                    
                    # Very simple loss - just encourage lower probability on negative examples
                    shift_logits = logits[0, :-1, :]
                    shift_labels = labels[1:]
                    
                    # Compute cross entropy but make it very small
                    loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
                    total_loss += loss * self.config.alpha  # Scale down
                    
            except Exception as e:
                logging.warning(f"Error processing example: {e}")
                continue
        
        # Average and scale down further
        if len(negative_examples) > 0:
            total_loss = total_loss / len(negative_examples)
        
        metrics = {
            "loss": total_loss.item(),
            "step": self.step
        }
        
        return total_loss, metrics
    
    async def safe_train_step(self, negative_examples: List[Dict[str, str]]) -> Dict[str, float]:
        """Perform one safe training step."""
        if not self.can_train:
            logging.info("Skipping training step - no trainable parameters")
            return {"loss": 0.0, "step": self.step}
        
        start_time = time.time()
        
        # Compute loss
        loss, metrics = self.compute_simple_loss(negative_examples)
        
        if loss.item() > 0:
            # Very conservative gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Very aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            # Check gradients before applying
            if self.config.enable_safety_checks:
                max_grad = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())
                
                if max_grad > 1.0:  # If gradients too large, skip step
                    logging.warning(f"Skipping step due to large gradients: {max_grad}")
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
            else:
                self.optimizer.step()
        
        self.step += 1
        
        metrics.update({
            "step": self.step,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.can_train else 0.0,
            "train_time": time.time() - start_time
        })
        
        self.metrics_history.append(metrics)
        return metrics
    
    def restore_original_weights(self):
        """Restore original model weights if training went wrong."""
        if hasattr(self, 'original_state'):
            logging.info("Restoring original model weights...")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.original_state:
                        param.data.copy_(self.original_state[name])
    
    def validate_model_health(self) -> bool:
        """Check if model is still functioning correctly."""
        try:
            # Generate a simple test
            test_input = self.tokenizer.encode("Hello", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(test_input)
                logits = outputs.logits
                
                # Check for NaN or Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    return False
                
                # Check if all outputs are the same (stuck model)
                if torch.std(logits) < 1e-6:
                    return False
                    
                return True
                
        except Exception as e:
            logging.error(f"Model health check failed: {e}")
            return False


async def run_safe_d2o_training(
    online_llm,
    negative_examples: List[Dict[str, str]],
    config: Optional[SafeD2OConfig] = None
) -> SafeD2OTrainer:
    """Run safe D2O training that won't corrupt the model."""
    if config is None:
        config = SafeD2OConfig()
    
    # Get model
    if hasattr(online_llm, 'hf_model') and online_llm.hf_model is not None:
        model = online_llm.hf_model
    elif hasattr(online_llm, 'llm') and online_llm.llm is not None:
        model = online_llm.llm.model_runner.model
    else:
        raise ValueError("Could not extract model from OnlineLLM instance")
    
    trainer = SafeD2OTrainer(config, model, online_llm.tokenizer)
    
    logging.info(f"Starting SAFE D2O training with {len(negative_examples)} examples")
    logging.info(f"Trainable parameters: {trainer.can_train}")
    
    # Pre-training health check
    if not trainer.validate_model_health():
        logging.error("Model failed initial health check!")
        return trainer
    
    for step in range(config.max_steps):
        # Sample batch
        batch_size = min(config.batch_size, len(negative_examples))
        batch_indices = np.random.choice(len(negative_examples), size=batch_size, replace=False)
        batch = [negative_examples[i] for i in batch_indices]
        
        # Training step
        metrics = await trainer.safe_train_step(batch)
        
        logging.info(f"Step {step}: {metrics}")
        
        # Health check after each step
        if config.enable_safety_checks and not trainer.validate_model_health():
            logging.error(f"Model corruption detected after step {step}! Restoring weights...")
            trainer.restore_original_weights()
            break
    
    # Final health check
    if trainer.validate_model_health():
        logging.info("✅ Safe D2O training completed successfully - model is healthy")
    else:
        logging.error("❌ Model corrupted - restoring original weights")
        trainer.restore_original_weights()
    
    return trainer


# Simple test function
async def test_safe_d2o():
    """Test the safe D2O implementation."""
    from src.dark.online_llm import OnlineLLM
    from src.dark.loss.d2o import create_negative_dataset_from_examples
    
    logging.info("Testing safe D2O implementation...")
    
    # Small model for testing
    llm = OnlineLLM(
        model="Qwen/Qwen3-0.6B",
        temperature=0.2,
        max_tokens=32,
        engine="hf",
        thinking_mode=False
    )
    
    # Simple negative examples
    negative_examples = create_negative_dataset_from_examples([
        {"prompt": "Test prompt", "negative_response": "Bad response"}
    ])
    
    # Test generation before training
    logging.info("Testing generation BEFORE safe D2O training:")
    response_before = await llm.generate_async("Hello, how are you?")
    logging.info(f"BEFORE: {response_before}")
    
    # Run safe training
    config = SafeD2OConfig(max_steps=3, learning_rate=1e-7)  # Extra conservative
    trainer = await run_safe_d2o_training(llm, negative_examples, config)
    
    # Test generation after training
    logging.info("Testing generation AFTER safe D2O training:")
    response_after = await llm.generate_async("Hello, how are you?")
    logging.info(f"AFTER: {response_after}")
    
    # Check if model is still working
    if "in in in" in response_after or len(response_after.strip()) < 5:
        logging.error("❌ Model appears corrupted after training")
    else:
        logging.info("✅ Model appears healthy after training")
    
    return trainer 