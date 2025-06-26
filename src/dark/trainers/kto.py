"""
KTO (Kawin-Thomke Optimization) trainer implementation for OnlineLLM.

This implements a simplified version of KTO that can be used with the OnlineLLM class
for preference learning without requiring paired preference data.

Based on the paper: https://arxiv.org/abs/2402.01306
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union
import logging
import time


class KTOTrainer:
    """
    KTO (Kawin-Thomke Optimization) trainer for preference learning.
    
    KTO enables learning from individual examples labeled as desirable/undesirable
    without requiring paired preference data.
    
    Args:
        beta (float): KTO temperature parameter (default: 0.1)
        desirable_weight (float): Weight for desirable (positive) examples (default: 1.0)  
        undesirable_weight (float): Weight for undesirable (negative) examples (default: 1.0)
        reference_free (bool): Whether to use reference-free KTO (default: False)
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        reference_free: bool = False,
    ) -> None:
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.reference_free = reference_free
        
    def compute_log_probs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probabilities for the given inputs."""
        if hasattr(model, '__call__'):
            # HF-style model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
        else:
            # Custom model interface
            logits = model(input_ids, attention_mask=attention_mask).logits
        
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        
        # Mask out padding tokens
        mask = (shift_labels != -100).float()
        masked_log_probs = selected_log_probs * mask
        
        # Sum log probs for each sequence
        batch_size = input_ids.size(0)
        seq_log_probs = masked_log_probs.view(batch_size, -1).sum(dim=1)
        
        return seq_log_probs
    
    def compute_kto_loss(
        self,
        policy_log_probs: torch.Tensor,
        reference_log_probs: Optional[torch.Tensor],
        labels: torch.Tensor,
        kl_estimate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KTO loss for a batch of examples.
        
        Args:
            policy_log_probs: Log probabilities from policy model
            reference_log_probs: Log probabilities from reference model (can be None)
            labels: Binary labels (1 for desirable, 0 for undesirable)
            kl_estimate: KL divergence estimate (optional)
            
        Returns:
            Dictionary containing loss components and metrics
        """
        if reference_log_probs is None or self.reference_free:
            # Reference-free KTO
            log_ratios = policy_log_probs
            kl = torch.tensor(0.0, device=policy_log_probs.device)
        else:
            # Standard KTO with reference model
            log_ratios = policy_log_probs - reference_log_probs
            kl = kl_estimate if kl_estimate is not None else torch.tensor(0.0, device=policy_log_probs.device)
        
        # Split into chosen (desirable) and rejected (undesirable)
        chosen_mask = labels.bool()
        rejected_mask = ~chosen_mask
        
        chosen_log_ratios = log_ratios[chosen_mask]
        rejected_log_ratios = log_ratios[rejected_mask]
        
        # Compute KTO losses
        losses = []
        metrics = {}
        
        if chosen_log_ratios.numel() > 0:
            # Loss for desirable examples: 1 - sigmoid(beta * (log_ratio - kl))
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_log_ratios - kl))
            losses.append(self.desirable_weight * chosen_losses)
            
            # Metrics
            chosen_rewards = self.beta * chosen_log_ratios.detach()
            metrics['chosen_rewards'] = chosen_rewards.mean()
            metrics['chosen_log_probs'] = policy_log_probs[chosen_mask].mean()
            metrics['num_chosen'] = chosen_log_ratios.numel()
        
        if rejected_log_ratios.numel() > 0:
            # Loss for undesirable examples: 1 - sigmoid(beta * (kl - log_ratio))  
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_log_ratios))
            losses.append(self.undesirable_weight * rejected_losses)
            
            # Metrics
            rejected_rewards = self.beta * rejected_log_ratios.detach()
            metrics['rejected_rewards'] = rejected_rewards.mean()
            metrics['rejected_log_probs'] = policy_log_probs[rejected_mask].mean()
            metrics['num_rejected'] = rejected_log_ratios.numel()
        
        if not losses:
            # No valid examples
            total_loss = torch.tensor(0.0, device=policy_log_probs.device, requires_grad=True)
        else:
            total_loss = torch.cat(losses).mean()
        
        metrics['loss'] = total_loss.detach()
        metrics['kl'] = kl.detach()
        
        return {
            'loss': total_loss,
            'metrics': metrics
        }
    
    def train_step(
        self,
        model: Any,
        reference_model: Optional[Any],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform one training step with KTO loss.
        
        Args:
            model: Policy model to train
            reference_model: Reference model (can be None for reference-free)
            batch: Batch containing input_ids, attention_mask, labels, and preference_labels
            optimizer: Optimizer for the policy model
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = batch['input_ids'].size(0)
        if batch_size < 4:
            raise ValueError(f"KTO requires batch size >= 4, got {batch_size}. "
                           "Use batch_learn() or batch_unlearn() methods to accumulate examples.")
        
        model.train()
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch['labels']
        preference_labels = batch['preference_labels']  # 1 for desirable, 0 for undesirable
        
        # Compute policy model log probabilities
        policy_log_probs = self.compute_log_probs(model, input_ids, labels, attention_mask)
        
        # Compute reference model log probabilities if available
        reference_log_probs = None
        if reference_model is not None and not self.reference_free:
            with torch.no_grad():
                if hasattr(reference_model, 'eval'):
                    reference_model.eval()
                reference_log_probs = self.compute_log_probs(
                    reference_model, input_ids, labels, attention_mask
                )
        
        # Compute KTO loss
        loss_output = self.compute_kto_loss(
            policy_log_probs,
            reference_log_probs,
            preference_labels
        )
        
        loss = loss_output['loss']
        metrics = loss_output['metrics']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Convert tensors to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


def create_kto_trainer(
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    reference_free: bool = False,
) -> KTOTrainer:
    """
    Factory function to create a KTO trainer.
    
    Args:
        beta: KTO temperature parameter
        desirable_weight: Weight for desirable examples
        undesirable_weight: Weight for undesirable examples  
        reference_free: Whether to use reference-free KTO
        
    Returns:
        Configured KTOTrainer instance
    """
    return KTOTrainer(
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        reference_free=reference_free,
    )


# Convenience functions for common KTO configurations
def create_standard_kto_trainer(beta: float = 0.1) -> KTOTrainer:
    """Create a standard KTO trainer with default parameters."""
    return create_kto_trainer(beta=beta)


def create_reference_free_kto_trainer(beta: float = 0.1) -> KTOTrainer:
    """Create a reference-free KTO trainer."""
    return create_kto_trainer(beta=beta, reference_free=True)
