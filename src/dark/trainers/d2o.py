"""
Distributional Dispreference Optimization (D2O) Implementation

Based on the paper: "Negating Negatives: Alignment with Human Negative Samples 
via Distributional Dispreference Optimization"

This module implements D2O, a novel alignment method that uses only human-annotated
negative samples to avoid reinforcing harmful content from noisy positive samples.
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
class D2OConfig:
    """Configuration for D2O training."""
    
    # Core D2O hyperparameters
    beta: float = 0.1  # Weight for self-generated samples
    alpha: float = 0.1  # Weight for negative samples
    K: int = 11  # Number of self-generated samples per prompt
    
    # Training parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    max_steps: int = 1000
    batch_size: int = 4
    
    # Sampling parameters for self-generation
    temperature: float = 0.8
    max_tokens: int = 256
    top_p: float = 0.9
    
    # Reference model parameters
    use_separate_ref_models: bool = False  # If False, uses same model for πr+ and πr-
    ref_model_update_interval: int = 50  # Steps between reference model updates
    
    # Diversity instructions for sampling
    use_moral_instructions: bool = True
    instruction_templates: List[str] = None
    
    def __post_init__(self):
        if self.instruction_templates is None:
            self.instruction_templates = [
                "Answer helpfully and ethically:",
                "Provide a safe and constructive response:",
                "Reply in a helpful and harmless way:",
                "Give a responsible and informative answer:",
                "Respond thoughtfully and appropriately:",
            ]


class D2OLoss:
    """
    Implements the D2O loss function for distributional dispreference optimization.
    
    The loss maximizes the discrepancy between self-generated responses and 
    negative samples, learning a distributional preference model.
    """
    
    def __init__(self, config: D2OConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_log_probs(
        self, 
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given sequences.
        
        Args:
            model: The language model
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities for each sequence [batch_size]
        """
        # For gradient computation, we need to be more careful about which models need gradients
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Create mask for valid tokens (not -100)
            valid_mask = (shift_labels != -100).float()
            
            # Only compute log probs for valid tokens
            if valid_mask.sum() == 0:
                # If no valid tokens, return zero log prob
                sequence_log_probs = torch.zeros(shift_labels.size(0), device=shift_labels.device)
            else:
                # Clamp labels to valid range to avoid index errors
                clamped_labels = shift_labels.clamp(min=0, max=log_probs.size(-1) - 1)
                
                # Gather log probabilities for the target tokens
                target_log_probs = log_probs.gather(
                    dim=-1, 
                    index=clamped_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Apply valid token mask
                target_log_probs = target_log_probs * valid_mask
                
                # Compute sequence-level log probabilities
                sequence_log_probs = target_log_probs.sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)
            
            return sequence_log_probs
    
    def compute_d2o_loss(
        self,
        model: torch.nn.Module,
        ref_model_pos: torch.nn.Module,
        ref_model_neg: torch.nn.Module,
        negative_examples: List[Dict[str, str]],
        self_generated_examples: List[List[Dict[str, str]]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the D2O loss.
        
        Args:
            model: The model being trained (π_θ)
            ref_model_pos: Reference model with helpful information (π_r+)
            ref_model_neg: Reference model with harmful information (π_r-)
            negative_examples: List of negative examples [{"prompt": str, "response": str}]
            self_generated_examples: List of K self-generated examples per prompt
                                   [[{"prompt": str, "response": str}] * K] * batch_size
            
        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        batch_size = len(negative_examples)
        device = next(model.parameters()).device
        
        # Tokenize negative examples
        negative_inputs = []
        negative_labels = []
        
        for example in negative_examples:
            prompt = example["prompt"]
            response = example["response"]
            full_text = prompt + " " + response  # Add space between prompt and response
            
            # Tokenize full text and prompt separately
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            
            # Ensure we don't exceed sequence length
            max_length = min(len(full_tokens), 512)  # Limit sequence length
            input_ids = torch.tensor(full_tokens[:max_length], dtype=torch.long)
            
            # Create labels (mask prompt tokens, only compute loss on response)
            labels = input_ids.clone()
            prompt_length = min(len(prompt_tokens), max_length)
            labels[:prompt_length] = -100  # Mask prompt tokens
            
            negative_inputs.append(input_ids)
            negative_labels.append(labels)
        
        # Pad sequences
        max_len = max(len(seq) for seq in negative_inputs)
        negative_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        negative_label_ids = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        negative_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, (input_seq, label_seq) in enumerate(zip(negative_inputs, negative_labels)):
            seq_len = len(input_seq)
            negative_input_ids[i, :seq_len] = input_seq.to(device)
            negative_label_ids[i, :seq_len] = label_seq.to(device)
            negative_attention_mask[i, :seq_len] = 1
        
        # Process self-generated examples
        self_gen_log_probs_model = []
        self_gen_log_probs_ref_neg = []
        
        for batch_idx, examples_k in enumerate(self_generated_examples):
            batch_log_probs_model = []
            batch_log_probs_ref_neg = []
            
            for example in examples_k:
                prompt = example["prompt"]
                response = example["response"]
                full_text = prompt + " " + response  # Add space between prompt and response
                
                # Tokenize safely
                full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                
                # Limit sequence length
                max_length = min(len(full_tokens), 512)
                input_ids = torch.tensor(full_tokens[:max_length], dtype=torch.long).unsqueeze(0).to(device)
                
                # Create labels (mask prompt tokens)
                labels = input_ids.clone()
                prompt_length = min(len(prompt_tokens), max_length)
                labels[0, :prompt_length] = -100  # Mask prompt tokens
                
                # Compute log probabilities with current model (keep gradients)
                log_prob_model = self.compute_log_probs(model, input_ids, labels)
                batch_log_probs_model.append(log_prob_model)
                
                # Compute log probabilities with reference model (no gradients)
                with torch.no_grad():
                    log_prob_ref_neg = self.compute_log_probs(ref_model_neg, input_ids, labels)
                batch_log_probs_ref_neg.append(log_prob_ref_neg)
            
            # Average over K samples
            avg_log_prob_model = torch.stack(batch_log_probs_model).mean()
            avg_log_prob_ref_neg = torch.stack(batch_log_probs_ref_neg).mean()
            
            self_gen_log_probs_model.append(avg_log_prob_model)
            self_gen_log_probs_ref_neg.append(avg_log_prob_ref_neg)
        
        self_gen_log_probs_model = torch.stack(self_gen_log_probs_model)
        self_gen_log_probs_ref_neg = torch.stack(self_gen_log_probs_ref_neg)
        
        # Compute log probabilities for negative samples
        neg_log_probs_model = self.compute_log_probs(
            model, negative_input_ids, negative_label_ids, negative_attention_mask
        )
        with torch.no_grad():
            neg_log_probs_ref_pos = self.compute_log_probs(
                ref_model_pos, negative_input_ids, negative_label_ids, negative_attention_mask
            )
        
        # Compute D2O loss components
        # Positive term: β/K * Σ log(π_θ(y_i|x) / π_r-(y_i|x))
        positive_term = (self.config.beta / self.config.K) * (
            self_gen_log_probs_model - self_gen_log_probs_ref_neg
        )
        
        # Negative term: α * log(π_θ(y_l|x) / π_r+(y_l|x))
        negative_term = self.config.alpha * (neg_log_probs_model - neg_log_probs_ref_pos)
        
        # D2O loss: -E[log σ(positive_term - negative_term)]
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


class D2OTrainer:
    """
    Trainer class for D2O alignment using only negative samples.
    """
    
    def __init__(
        self, 
        config: D2OConfig,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        ref_model_pos: Optional[torch.nn.Module] = None,
        ref_model_neg: Optional[torch.nn.Module] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Reference models
        if ref_model_pos is None:
            ref_model_pos = model  # Use same model as reference
        if ref_model_neg is None:
            ref_model_neg = model  # Use same model as reference
            
        self.ref_model_pos = ref_model_pos
        self.ref_model_neg = ref_model_neg
        
        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        
        # Initialize loss function
        self.loss_fn = D2OLoss(config, tokenizer)
        
        # Get trainable parameters (only those that require grad)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found in the model")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Training state
        self.step = 0
        self.metrics_history = []
        
    async def generate_self_samples(
        self, 
        prompts: List[str],
        online_llm: Any  # OnlineLLM instance
    ) -> List[List[Dict[str, str]]]:
        """
        Generate K self-samples for each prompt using the current model.
        
        Args:
            prompts: List of input prompts
            online_llm: OnlineLLM instance for generation
            
        Returns:
            List of K generated examples per prompt
        """
        all_examples = []
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=1,  # Generate one at a time
            presence_penalty=0.1
        )
        
        for prompt in prompts:
            prompt_examples = []
            
            # Generate K samples for this prompt
            for k in range(self.config.K):
                # Add diversity through instruction templates
                if self.config.use_moral_instructions and k < len(self.config.instruction_templates):
                    augmented_prompt = f"{self.config.instruction_templates[k]} {prompt}"
                else:
                    augmented_prompt = prompt
                
                # Generate response with timeout
                try:
                    # Add timeout to prevent hanging
                    response = await asyncio.wait_for(
                        online_llm.generate_async(
                            augmented_prompt, 
                            sampling_params=sampling_params
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    prompt_examples.append({
                        "prompt": prompt,  # Use original prompt, not augmented
                        "response": response
                    })
                except (Exception, asyncio.TimeoutError) as e:
                    logging.warning(f"Failed to generate sample {k} for prompt: {e}")
                    # Use a fallback response
                    prompt_examples.append({
                        "prompt": prompt,
                        "response": "I understand your request, but I want to provide a helpful and appropriate response."
                    })
            
            all_examples.append(prompt_examples)
        
        return all_examples
    
    async def train_step(
        self,
        negative_examples: List[Dict[str, str]],
        online_llm: Any,
        update_ref_model: bool = False
    ) -> Dict[str, float]:
        """
        Perform one training step of D2O.
        
        Args:
            negative_examples: Batch of negative examples
            online_llm: OnlineLLM instance for self-generation
            update_ref_model: Whether to update reference model states
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode more robustly
        if hasattr(self.model, 'train'):
            try:
                self.model.train()
            except AttributeError:
                # Some custom models might not have train() method
                pass
        
        # Extract prompts from negative examples
        prompts = [ex["prompt"] for ex in negative_examples]
        
        # Generate self-samples
        if self.step >= self.config.warmup_steps:
            self_generated_examples = await self.generate_self_samples(prompts, online_llm)
        else:
            # During warmup, use simplified self-generation
            self_generated_examples = []
            for prompt in prompts:
                examples_k = []
                for _ in range(self.config.K):
                    examples_k.append({
                        "prompt": prompt,
                        "response": "I'll try to be helpful and appropriate in my response."
                    })
                self_generated_examples.append(examples_k)
        
        # Compute loss
        loss, metrics = self.loss_fn.compute_d2o_loss(
            self.model,
            self.ref_model_pos,
            self.ref_model_neg,
            negative_examples,
            self_generated_examples
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update reference models periodically
        if update_ref_model and self.step % self.config.ref_model_update_interval == 0:
            self._update_reference_models()
        
        self.step += 1
        
        # Add step info to metrics
        metrics.update({
            "step": self.step,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        })
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _update_reference_models(self):
        """Update reference model states with current model."""
        if not self.config.use_separate_ref_models:
            # Copy current model state to reference models
            with torch.no_grad():
                for target_param, source_param in zip(
                    self.ref_model_pos.parameters(), 
                    self.model.parameters()
                ):
                    target_param.data.copy_(source_param.data)
                    
                for target_param, source_param in zip(
                    self.ref_model_neg.parameters(), 
                    self.model.parameters()
                ):
                    target_param.data.copy_(source_param.data)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "step": self.step,
            "metrics_history": self.metrics_history
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.metrics_history = checkpoint["metrics_history"]


def create_negative_dataset_from_examples(
    examples: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Create a negative dataset from examples.
    
    Args:
        examples: List of examples, each containing at least 'prompt' and 'negative_response'
        
    Returns:
        List of negative examples in D2O format
    """
    negative_examples = []
    
    for example in examples:
        if "negative_response" in example:
            negative_examples.append({
                "prompt": example["prompt"],
                "response": example["negative_response"]
            })
        elif "responses" in example and "preferred" in example:
            # Handle preference format - use non-preferred as negative
            responses = example["responses"]
            preferred_idx = example["preferred"]
            
            for i, response in enumerate(responses):
                if i != preferred_idx:  # Non-preferred responses are negative
                    negative_examples.append({
                        "prompt": example["prompt"],
                        "response": response
                    })
    
    return negative_examples


# Example usage functions
async def run_d2o_training(
    online_llm,
    negative_examples: List[Dict[str, str]],
    config: Optional[D2OConfig] = None,
    checkpoint_path: Optional[str] = None
) -> D2OTrainer:
    """
    Run D2O training on negative examples.
    
    Args:
        online_llm: OnlineLLM instance
        negative_examples: List of negative training examples
        config: D2O configuration (uses default if None)
        checkpoint_path: Path to save checkpoints
        
    Returns:
        Trained D2OTrainer instance
    """
    if config is None:
        config = D2OConfig()
    
    # Get the underlying model from OnlineLLM
    if hasattr(online_llm, 'hf_model') and online_llm.hf_model is not None:
        model = online_llm.hf_model
    elif hasattr(online_llm, 'llm') and online_llm.llm is not None:
        model = online_llm.llm.model_runner.model
    else:
        raise ValueError("Could not extract model from OnlineLLM instance")
    
    # Create trainer
    trainer = D2OTrainer(config, model, online_llm.tokenizer)
    
    logging.info(f"Starting D2O training with {len(negative_examples)} negative examples")
    logging.info(f"Config: {config}")
    
    # Training loop
    for step in range(config.max_steps):
        # Sample batch
        batch_indices = np.random.choice(
            len(negative_examples), 
            size=min(config.batch_size, len(negative_examples)), 
            replace=False
        )
        batch = [negative_examples[i] for i in batch_indices]
        
        # Training step
        metrics = await trainer.train_step(
            batch, 
            online_llm, 
            update_ref_model=True
        )
        
        if step % 10 == 0:
            logging.info(f"Step {step}: {metrics}")
        
        # Save checkpoint
        if checkpoint_path and step % 100 == 0:
            trainer.save_checkpoint(f"{checkpoint_path}_step_{step}.pt")
    
    logging.info("D2O training completed")
    return trainer
