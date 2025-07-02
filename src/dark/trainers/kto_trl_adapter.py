"""
KTO Trainer that uses TRL's implementation as a backend.

This adapter allows us to use TRL's proven KTO implementation while maintaining
compatibility with our OnlineLLM interface.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset

try:
    from trl import KTOTrainer as TRLKTOTrainer, KTOConfig
    from trl.models import create_reference_model
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logging.warning("TRL not available. Install with: pip install trl")

from transformers import PreTrainedTokenizerBase, PreTrainedModel


class TRLKTOAdapter:
    """
    Adapter that uses TRL's KTOTrainer as the backend for KTO training.
    
    This provides a simpler interface for OnlineLLM while leveraging
    TRL's proven implementation.
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
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required for TRLKTOAdapter. Install with: pip install trl")
        
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
        # Will be set when we get the model
        self._trl_trainer = None
        self._model = None
        self._tokenizer = None
        
    def _ensure_trl_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        """Ensure TRL trainer is created when needed."""
        if self._trl_trainer is None or self._model is not model:
            self._model = model
            self._tokenizer = tokenizer
            
            # Create reference model
            ref_model = create_reference_model(model)
            
            # Create KTO config
            config = KTOConfig(
                output_dir="./temp_kto_output",  # Temporary, won't be used
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                num_train_epochs=1,
                learning_rate=self.learning_rate,
                beta=self.beta,
                desirable_weight=self.desirable_weight,
                undesirable_weight=self.undesirable_weight,
                max_length=self.max_length,
                max_prompt_length=self.max_prompt_length,
                eval_strategy="no",  # We don't need evaluation
                save_strategy="no",  # We don't need saving
                logging_steps=1000000,  # Disable logging
                report_to=[],  # No reporting
                **self.kwargs
            )
            
            # Create dummy dataset (required by TRL)
            dummy_data = {
                "prompt": ["Hello"],
                "completion": ["Hi there!"],
                "label": [True]
            }
            dummy_dataset = Dataset.from_dict(dummy_data)
            
            # Create TRL trainer
            self._trl_trainer = TRLKTOTrainer(
                model=model,
                ref_model=ref_model,
                args=config,
                train_dataset=dummy_dataset,
                processing_class=tokenizer,
            )
    
    def prepare_kto_batch(
        self, 
        examples: List[Dict[str, Any]], 
        tokenizer: PreTrainedTokenizerBase
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for KTO training using TRL's data processing.
        
        Args:
            examples: List of examples with 'prompt', 'response', 'desirable' fields
            tokenizer: Tokenizer to use
            
        Returns:
            Batch dict ready for TRL KTO training
        """
        # Convert our format to TRL's expected format
        trl_data = {
            "prompt": [ex["prompt"] for ex in examples],
            "completion": [ex["response"] for ex in examples], 
            "label": [bool(ex["desirable"]) for ex in examples]
        }
        
        # Create dataset and process with TRL's pipeline
        dataset = Dataset.from_dict(trl_data)
        
        # Apply TRL's tokenization and processing
        # This automatically handles all the complex tokenization logic
        processed_dataset = self._trl_trainer.train_dataset.__class__.from_dict(trl_data)
        
        # Use TRL's internal processing
        from trl.trainer.kto_trainer import _tokenize, _process_tokens
        
        # Tokenize
        tokenized = processed_dataset.map(
            _tokenize,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer}
        )
        
        # Process tokens
        fn_kwargs = {
            "prefix": "",
            "is_encoder_decoder": False,
            "tokenizer": tokenizer,
            "max_length": self.max_length,
            "truncation_mode": "keep_end",
            "label_pad_token_id": -100,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": None,
        }
        
        processed = tokenized.map(
            _process_tokens,
            fn_kwargs=fn_kwargs
        )
        
        # Convert to torch tensors and pad
        batch = {}
        device = next(self._model.parameters()).device
        
        # Get the required fields
        required_fields = [
            'completion_input_ids', 'completion_attention_mask', 
            'completion_labels', 'prompt_input_ids', 'prompt_attention_mask'
        ]
        
        for field in required_fields:
            if field in processed.column_names:
                values = processed[field]
                # Pad to same length
                max_len = max(len(v) for v in values)
                padded_values = []
                for v in values:
                    if field == 'completion_labels':
                        padded = v + [-100] * (max_len - len(v))
                    else:
                        pad_token = tokenizer.pad_token_id if field.endswith('_ids') else 0
                        padded = v + [pad_token] * (max_len - len(v))
                    padded_values.append(padded)
                batch[field] = torch.tensor(padded_values, device=device)
        
        # Add labels for TRL
        batch['label'] = [bool(ex["desirable"]) for ex in examples]
        
        return batch
    
    def train_step(
        self,
        examples: List[Dict[str, Any]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform a single KTO training step using TRL's implementation.
        
        Args:
            examples: List of examples with 'prompt', 'response', 'desirable' fields
            model: Model to train
            tokenizer: Tokenizer
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss and metrics
        """
        # Ensure TRL trainer is set up
        self._ensure_trl_trainer(model, tokenizer)
        
        # Prepare batch using TRL's processing
        batch = self.prepare_kto_batch(examples, tokenizer)
        
        # Use TRL's loss computation
        model.train()
        loss, metrics = self._trl_trainer.get_batch_loss_metrics(model, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Add loss to metrics
        metrics['loss'] = loss.item()
        
        return metrics
    
    def tokenize_conversation(self, tokenizer: PreTrainedTokenizerBase, prompt: str, completion: str) -> Dict[str, Any]:
        """Tokenize a conversation using TRL's approach."""
        # Ensure trainer is set up
        self._ensure_trl_trainer(self._model or tokenizer, tokenizer)
        
        # Use TRL's tokenization
        batch = {"prompt": [prompt], "completion": [completion]}
        from trl.trainer.kto_trainer import _tokenize
        
        tokenized = _tokenize(batch, tokenizer)
        
        # Return first item (since we only have one)
        return {k: v[0] for k, v in tokenized.items()}


def create_trl_kto_trainer(
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    learning_rate: float = 1e-6,
    **kwargs
) -> TRLKTOAdapter:
    """
    Create a TRL-based KTO trainer.
    
    Args:
        beta: KTO beta parameter
        desirable_weight: Weight for desirable examples
        undesirable_weight: Weight for undesirable examples
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        learning_rate: Learning rate
        **kwargs: Additional arguments
        
    Returns:
        TRLKTOAdapter instance
    """
    return TRLKTOAdapter(
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        learning_rate=learning_rate,
        **kwargs
    ) 