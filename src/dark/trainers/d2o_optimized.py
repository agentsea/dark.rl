"""
Optimized D2O Implementation for Better Performance

This version addresses the performance bottlenecks in the original D2O implementation:
- Batched self-sample generation
- Reduced CPU-GPU transfers
- Simplified computation paths
- Better memory management
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer

from ..sampling_params import SamplingParams


@dataclass
class OptimizedD2OConfig:
    """Optimized configuration for D2O training."""
    
    # Core D2O hyperparameters
    beta: float = 0.1
    alpha: float = 0.1
    K: int = 5  # Reduced default
    
    # Training parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 50  # Reduced
    max_steps: int = 200  # Reduced
    batch_size: int = 2
    
    # Sampling parameters (optimized for speed)
    temperature: float = 0.7
    max_tokens: int = 64  # Shorter for speed
    top_p: float = 0.9
    
    # Performance optimizations
    use_simplified_ref_models: bool = True  # Use same model for both references
    enable_batched_generation: bool = True  # Generate multiple samples in parallel
    max_seq_length: int = 256  # Hard limit to prevent very long sequences
    
    # Disable expensive features for speed
    use_moral_instructions: bool = False
    save_checkpoints: bool = False


class OptimizedD2OLoss:
    """Optimized D2O loss computation."""
    
    def __init__(self, config: OptimizedD2OConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tokenize_batch(self, prompts: List[str], responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficiently tokenize a batch of prompt-response pairs."""
        batch_input_ids = []
        batch_labels = []
        
        for prompt, response in zip(prompts, responses):
            # Tokenize efficiently
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True, max_length=128, truncation=True)
            full_tokens = self.tokenizer.encode(
                prompt + " " + response, 
                add_special_tokens=True, 
                max_length=self.config.max_seq_length,
                truncation=True
            )
            
            # Create labels (mask prompt tokens)
            labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
            
            # Pad to same length
            max_len = self.config.max_seq_length
            full_tokens = full_tokens[:max_len] + [self.tokenizer.pad_token_id] * (max_len - len(full_tokens))
            labels = labels[:max_len] + [-100] * (max_len - len(labels))
            
            batch_input_ids.append(full_tokens)
            batch_labels.append(labels)
        
        return (
            torch.tensor(batch_input_ids, dtype=torch.long, device=self.device),
            torch.tensor(batch_labels, dtype=torch.long, device=self.device),
            (torch.tensor(batch_input_ids, device=self.device) != self.tokenizer.pad_token_id).long()
        )
    
    def compute_batch_log_probs(self, model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for a batch efficiently."""
        with torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            
            # Compute loss efficiently using F.cross_entropy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for efficient computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            # Compute log probabilities only for non-masked tokens
            valid_mask = (flat_labels != -100)
            if valid_mask.sum() == 0:
                return torch.zeros(input_ids.size(0), device=self.device)
            
            # Use cross_entropy with reduction='none' for per-token losses
            per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=-100)
            
            # Reshape and average over sequence length per sample
            per_token_loss = per_token_loss.view(input_ids.size(0), -1)
            valid_tokens = (shift_labels != -100).float()
            
            # Average loss per sequence (negative log likelihood)
            sequence_log_probs = -per_token_loss.sum(dim=-1) / valid_tokens.sum(dim=-1).clamp(min=1)
            
            return sequence_log_probs
    
    def compute_optimized_d2o_loss(
        self,
        model: torch.nn.Module,
        negative_examples: List[Dict[str, str]],
        self_generated_examples: List[List[Dict[str, str]]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimized D2O loss computation."""
        
        # Process negative examples
        neg_prompts = [ex["prompt"] for ex in negative_examples]
        neg_responses = [ex["response"] for ex in negative_examples]
        neg_input_ids, neg_labels, neg_attention_mask = self.tokenize_batch(neg_prompts, neg_responses)
        
        # Process self-generated examples (flatten and batch)
        all_self_prompts = []
        all_self_responses = []
        batch_indices = []  # Track which examples belong to which batch
        
        for batch_idx, examples_k in enumerate(self_generated_examples):
            for example in examples_k:
                all_self_prompts.append(example["prompt"])
                all_self_responses.append(example["response"])
                batch_indices.append(batch_idx)
        
        if len(all_self_prompts) > 0:
            self_input_ids, self_labels, self_attention_mask = self.tokenize_batch(all_self_prompts, all_self_responses)
        else:
            # Fallback if no self-generated examples
            self_input_ids = neg_input_ids
            self_labels = neg_labels
            self_attention_mask = neg_attention_mask
            batch_indices = list(range(len(negative_examples)))
        
        # Compute log probabilities efficiently
        neg_log_probs = self.compute_batch_log_probs(model, neg_input_ids, neg_labels, neg_attention_mask)
        
        if len(all_self_prompts) > 0:
            self_log_probs = self.compute_batch_log_probs(model, self_input_ids, self_labels, self_attention_mask)
            
            # Group self-generated log probs by batch
            grouped_self_log_probs = []
            for batch_idx in range(len(self_generated_examples)):
                batch_log_probs = [self_log_probs[i] for i, bi in enumerate(batch_indices) if bi == batch_idx]
                if batch_log_probs:
                    grouped_self_log_probs.append(torch.stack(batch_log_probs).mean())
                else:
                    grouped_self_log_probs.append(torch.tensor(0.0, device=self.device))
            
            self_avg_log_probs = torch.stack(grouped_self_log_probs)
        else:
            # Fallback during warmup
            self_avg_log_probs = torch.zeros_like(neg_log_probs)
        
        # Simplified D2O loss (using same model for references for speed)
        positive_term = self.config.beta * self_avg_log_probs
        negative_term = self.config.alpha * neg_log_probs
        
        # D2O loss
        logits = positive_term - negative_term
        loss = -F.logsigmoid(logits).mean()
        
        # Compute metrics
        metrics = {
            "d2o_loss": loss.item(),
            "positive_term": positive_term.mean().item(),
            "negative_term": negative_term.mean().item(),
            "logits_mean": logits.mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }
        
        return loss, metrics


class OptimizedD2OTrainer:
    """Optimized D2O trainer for better performance."""
    
    def __init__(self, config: OptimizedD2OConfig, model: torch.nn.Module, tokenizer: AutoTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        
        self.loss_fn = OptimizedD2OLoss(config, tokenizer)
        
        # Optimizer with only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.step = 0
        self.metrics_history = []
    
    async def generate_batch_samples(self, prompts: List[str], online_llm) -> List[List[Dict[str, str]]]:
        """Generate samples in batches for better performance."""
        all_examples = []
        
        # Optimized sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=1,
            presence_penalty=0.0  # Disable for speed
        )
        
        if self.config.enable_batched_generation and len(prompts) > 1:
            # Try batched generation for speed
            try:
                # Generate all K samples for all prompts in one go
                all_prompts_repeated = []
                prompt_indices = []
                
                for prompt_idx, prompt in enumerate(prompts):
                    for k in range(self.config.K):
                        all_prompts_repeated.append(prompt)
                        prompt_indices.append(prompt_idx)
                
                # Generate all at once with timeout
                responses = await asyncio.wait_for(
                    online_llm.batch_generate(all_prompts_repeated, sampling_params),
                    timeout=60.0
                )
                
                # Group responses by original prompt
                for prompt_idx in range(len(prompts)):
                    prompt_examples = []
                    for k in range(self.config.K):
                        resp_idx = prompt_idx * self.config.K + k
                        if resp_idx < len(responses):
                            prompt_examples.append({
                                "prompt": prompts[prompt_idx],
                                "response": responses[resp_idx]
                            })
                        else:
                            prompt_examples.append({
                                "prompt": prompts[prompt_idx],
                                "response": "Safe response."
                            })
                    all_examples.append(prompt_examples)
                
                return all_examples
                
            except Exception as e:
                logging.warning(f"Batched generation failed: {e}, falling back to sequential")
        
        # Fallback: sequential generation with timeouts
        for prompt in prompts:
            prompt_examples = []
            for k in range(self.config.K):
                try:
                    response = await asyncio.wait_for(
                        online_llm.generate_async(prompt, sampling_params),
                        timeout=10.0
                    )
                    prompt_examples.append({"prompt": prompt, "response": response})
                except Exception:
                    prompt_examples.append({"prompt": prompt, "response": "Safe response."})
            all_examples.append(prompt_examples)
        
        return all_examples
    
    async def train_step(self, negative_examples: List[Dict[str, str]], online_llm) -> Dict[str, float]:
        """Optimized training step."""
        start_time = time.time()
        
        # Extract prompts
        prompts = [ex["prompt"] for ex in negative_examples]
        
        # Generate self-samples (only after warmup)
        if self.step >= self.config.warmup_steps:
            self_generated_examples = await self.generate_batch_samples(prompts, online_llm)
        else:
            # During warmup, use simple fallbacks
            self_generated_examples = []
            for prompt in prompts:
                examples_k = [{"prompt": prompt, "response": "I'll be helpful and appropriate."} for _ in range(self.config.K)]
                self_generated_examples.append(examples_k)
        
        generation_time = time.time() - start_time
        
        # Compute loss
        loss_start = time.time()
        loss, metrics = self.loss_fn.compute_optimized_d2o_loss(
            self.model, negative_examples, self_generated_examples
        )
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        backward_time = time.time() - backward_start
        
        self.step += 1
        
        # Add timing and step info
        metrics.update({
            "step": self.step,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "generation_time": generation_time,
            "loss_time": loss_time,
            "backward_time": backward_time,
            "total_time": time.time() - start_time
        })
        
        self.metrics_history.append(metrics)
        return metrics


async def run_optimized_d2o_training(
    online_llm,
    negative_examples: List[Dict[str, str]],
    config: Optional[OptimizedD2OConfig] = None
) -> OptimizedD2OTrainer:
    """Run optimized D2O training."""
    if config is None:
        config = OptimizedD2OConfig()
    
    # Get model
    if hasattr(online_llm, 'hf_model') and online_llm.hf_model is not None:
        model = online_llm.hf_model
    elif hasattr(online_llm, 'llm') and online_llm.llm is not None:
        model = online_llm.llm.model_runner.model
    else:
        raise ValueError("Could not extract model from OnlineLLM instance")
    
    trainer = OptimizedD2OTrainer(config, model, online_llm.tokenizer)
    
    logging.info(f"Starting optimized D2O training with {len(negative_examples)} examples")
    logging.info(f"Config: K={config.K}, batch_size={config.batch_size}, max_steps={config.max_steps}")
    
    for step in range(config.max_steps):
        # Sample batch
        batch_indices = np.random.choice(
            len(negative_examples), 
            size=min(config.batch_size, len(negative_examples)), 
            replace=False
        )
        batch = [negative_examples[i] for i in batch_indices]
        
        # Training step
        metrics = await trainer.train_step(batch, online_llm)
        
        if step % 5 == 0:  # More frequent logging
            logging.info(f"Step {step}: loss={metrics['d2o_loss']:.4f}, "
                        f"gen_time={metrics['generation_time']:.2f}s, "
                        f"total_time={metrics['total_time']:.2f}s")
    
    logging.info("Optimized D2O training completed")
    return trainer 