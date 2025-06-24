# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
import os
import re

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Qwen3Config,
    GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, TaskType
import warnings

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*generation_config.*default values have been modified.*")
warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")

# Also set transformers logging level to reduce warnings
import transformers
transformers.logging.set_verbosity_error()

from dark.config import Config

logger = logging.get_logger(__name__)


class Qwen3HFForCausalLM:
    """
    HuggingFace-based implementation of Qwen3 that maintains compatibility 
    with OnlineLLM interface while using standard HF transformers.
    """
    
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        self.config = config.hf_config
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Load tokenizer and model using HF
        model_name = config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        # Set up generation config with Qwen3 best practices
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,  # For thinking mode
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_new_tokens=2048,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,  # Explicitly set to avoid warning
        )
        
        # Apply LoRA if specified
        if lora_rank > 0:
            self._setup_lora()
        
        # Track LoRA adapters
        self.lora_adapters = {}
        self.current_adapter = None
        
        # Thinking mode settings
        self.enable_thinking = True
        self.thinking_token_start = "<think>"
        self.thinking_token_end = "</think>"
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def add_lora_adapter(self, adapter_name: str, lora_rank: int = None, lora_alpha: float = None):
        """Add a new LoRA adapter"""
        if not hasattr(self.model, 'add_adapter'):
            logger.warning("Model doesn't support multiple LoRA adapters")
            return
        
        rank = lora_rank or self.lora_rank
        alpha = lora_alpha or self.lora_alpha
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model.add_adapter(adapter_name, lora_config)
        self.lora_adapters[adapter_name] = {"rank": rank, "alpha": alpha}
    
    def set_adapter(self, adapter_name: str):
        """Set the active LoRA adapter"""
        if hasattr(self.model, 'set_adapter'):
            self.model.set_adapter(adapter_name)
            self.current_adapter = adapter_name
        else:
            logger.warning("Model doesn't support adapter switching")
    
    def list_adapters(self) -> List[str]:
        """List available LoRA adapters"""
        return list(self.lora_adapters.keys())
    
    def _prepare_messages(self, messages: List[Dict[str, str]], enable_thinking: bool = None) -> str:
        """Prepare messages for generation using Qwen3 chat template"""
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        
        # Apply chat template with thinking mode
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        return text
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse response to separate thinking content from final answer"""
        thinking_content = ""
        final_content = response
        
        # Look for thinking tokens
        if self.thinking_token_start in response and self.thinking_token_end in response:
            # Extract thinking content
            start_idx = response.find(self.thinking_token_start)
            end_idx = response.find(self.thinking_token_end)
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = response[start_idx + len(self.thinking_token_start):end_idx].strip()
                final_content = response[end_idx + len(self.thinking_token_end):].strip()
        
        return thinking_content, final_content
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = None,
        temperature: float = None,
        enable_thinking: bool = None,
        **kwargs
    ) -> str:
        """Generate response using HF implementation"""
        
        # Prepare input
        text = self._prepare_messages(messages, enable_thinking)
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_config = self.generation_config
        if max_tokens:
            gen_config.max_new_tokens = max_tokens
        if temperature is not None:
            gen_config.temperature = temperature
            # Adjust other params based on thinking mode
            if enable_thinking or (enable_thinking is None and self.enable_thinking):
                gen_config.top_p = 0.95
                gen_config.top_k = 20
            else:
                gen_config.top_p = 0.8
                gen_config.top_k = 20
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
                **kwargs
            )
        
        # Decode response
        output_ids = generated_ids[0][len(inputs['input_ids'][0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return response
    
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        cu_seqlens: torch.Tensor = None, 
        max_seqlen: int = None, 
        labels: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Forward pass compatible with OnlineLLM interface"""
        
        # Convert custom format to standard HF format if needed
        if attention_mask is None and cu_seqlens is not None:
            # Create attention mask from cu_seqlens
            batch_size = len(cu_seqlens) - 1
            max_len = input_ids.size(0) // batch_size if batch_size > 0 else input_ids.size(0)
            attention_mask = torch.ones(batch_size, max_len, device=input_ids.device)
        
        # Standard HF forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def freeze_base_model(self):
        """Freeze base model parameters (keep LoRA trainable)"""
        if hasattr(self.model, 'base_model'):
            # PEFT model
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        else:
            # Regular model - freeze non-LoRA params
            for name, param in self.model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters"""
        return self.model.named_parameters()
    
    def state_dict(self):
        """Get model state dict"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict"""
        return self.model.load_state_dict(state_dict)
    
    def save_pretrained(self, save_directory):
        """Save model"""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_directory)
        else:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
    
    @property
    def device(self):
        """Get model device"""
        return next(self.model.parameters()).device


def create_qwen3_hf_model(config: Config, lora_rank=0, lora_alpha=1.0) -> Qwen3HFForCausalLM:
    """Factory function to create Qwen3 HF model"""
    return Qwen3HFForCausalLM(config, lora_rank=lora_rank, lora_alpha=lora_alpha) 