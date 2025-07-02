"""
Direct TRL KTO integration that follows TRL's exact training pattern.

This bypasses our custom batching and uses TRL's complete pipeline.
"""

import logging
import torch
import tempfile
import os
from typing import Dict, List, Any, Optional
from datasets import Dataset

try:
    from trl import KTOTrainer as TRLKTOTrainer, KTOConfig
    from transformers import TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    TRL_AVAILABLE = True
    PEFT_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    PEFT_AVAILABLE = False
    logging.warning(f"TRL or PEFT not available: {e}. Install with: pip install trl peft")

from transformers import PreTrainedTokenizerBase, PreTrainedModel


class DirectTRLKTOTrainer:
    """
    Direct integration with TRL KTO that uses their complete training pipeline.
    
    This creates actual TRL trainers for each training session and uses
    their full training loop instead of trying to extract individual steps.
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        max_length: int = 1024,
        max_prompt_length: int = 512,
        learning_rate: float = 1e-6,
        **kwargs
    ):
        if not TRL_AVAILABLE or not PEFT_AVAILABLE:
            raise ImportError("TRL and PEFT are required for DirectTRLKTOTrainer. Install with: pip install trl peft")
        
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self._lora_applied = False  # Track if LoRA has been applied
    
    def train_with_trl(
        self,
        examples: List[Dict[str, Any]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        steps: int = 3,
        lr: float = None
    ) -> PreTrainedModel:
        """
        Train directly with TRL's complete pipeline.
        
        Args:
            examples: List of examples with 'prompt', 'response', 'desirable' fields
            model: Model to train
            tokenizer: Tokenizer
            steps: Number of training steps
            lr: Learning rate (overrides default)
            
        Returns:
            Trained model
        """
        if lr is None:
            lr = self.learning_rate
        
        # Convert our examples to TRL format
        trl_data = {
            "prompt": [ex["prompt"] for ex in examples],
            "completion": [ex["response"] for ex in examples],
            "label": [bool(ex["desirable"]) for ex in examples]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(trl_data)
        
        # Apply LoRA to the model for memory efficiency (only once)
        if PEFT_AVAILABLE and not self._lora_applied and not hasattr(model, 'peft_config'):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)
            self._lora_applied = True
            logging.info("Applied LoRA to model")
        
        # Clear GPU cache before training
        torch.cuda.empty_cache()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create KTO config following TRL's pattern
            training_args = KTOConfig(
                output_dir=temp_dir,
                per_device_train_batch_size=max(1, len(examples)),  # Use full batch
                num_train_epochs=1,
                max_steps=steps,
                learning_rate=lr,
                beta=self.beta,
                desirable_weight=self.desirable_weight,
                undesirable_weight=self.undesirable_weight,
                max_length=self.max_length,
                max_prompt_length=self.max_prompt_length,
                eval_strategy="no",
                save_strategy="no",
                logging_steps=1000000,  # Disable logging
                report_to=[],
                remove_unused_columns=False,
                dataloader_drop_last=False,
                **self.kwargs
            )
            
            # Create TRL trainer
            # With PEFT, we can pass None for ref_model and TRL will handle it properly
            trainer = TRLKTOTrainer(
                model=model,
                ref_model=None,  # Use None with PEFT to avoid memory duplication
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            
            # Train using TRL's complete pipeline
            trainer.train()
            
            # Get the trained model before cleaning up
            trained_model = trainer.model
            
            # Clean up trainer to free memory
            del trainer
            torch.cuda.empty_cache()
            
            return trained_model
    
    def train_step(
        self,
        examples: List[Dict[str, Any]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        lr: float = None
    ) -> Dict[str, float]:
        """
        Perform training using TRL's complete pipeline.
        
        Note: This method trains the model in-place using TRL's trainer.
        The optimizer parameter is ignored as TRL creates its own.
        """
        # Train with TRL
        trained_model = self.train_with_trl(examples, model, tokenizer, steps, lr)
        
        # Copy trained parameters back to original model
        model.load_state_dict(trained_model.state_dict())
        
        # Return dummy metrics (TRL trainer doesn't expose individual step metrics easily)
        return {
            "loss": 0.0,  # Placeholder
            "kl": 0.0,
            "learning_rate": lr or self.learning_rate
        }


def create_direct_trl_kto_trainer(
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    learning_rate: float = 1e-6,
    **kwargs
) -> DirectTRLKTOTrainer:
    """
    Create a direct TRL KTO trainer.
    
    Args:
        beta: KTO beta parameter
        desirable_weight: Weight for desirable examples
        undesirable_weight: Weight for undesirable examples
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        learning_rate: Learning rate
        **kwargs: Additional arguments
        
    Returns:
        DirectTRLKTOTrainer instance
    """
    return DirectTRLKTOTrainer(
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        learning_rate=learning_rate,
        **kwargs
    ) 