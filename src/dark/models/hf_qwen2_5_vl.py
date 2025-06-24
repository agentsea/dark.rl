"""
HuggingFace-based Qwen2.5-VL implementation
This module wraps the exact HuggingFace implementation to make it compatible with our existing interface.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from dark.config import Config


class HFQwen2_5_VLForCausalLM(nn.Module):
    """
    Wrapper around HuggingFace's Qwen2.5-VL implementation to make it compatible
    with our existing interface and online_llm.py
    """
    
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.config = config
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Load the exact HF implementation
        self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Store reference to the processor for compatibility
        self.processor = AutoProcessor.from_pretrained(config.model)
        
        # Make the model's components accessible for compatibility
        self.visual = self.hf_model.model.visual
        self.model = self.hf_model.model
        self.lm_head = self.hf_model.lm_head
        
        # For LoRA compatibility - we'll add LoRA layers if needed
        if lora_rank > 0:
            self._add_lora_layers()
    
    def _add_lora_layers(self):
        """Add LoRA layers to the HF model for fine-tuning"""
        # This is a placeholder - would need proper LoRA implementation
        # For now, we'll just mark parameters as requiring gradients
        for name, param in self.hf_model.named_parameters():
            if any(layer_name in name for layer_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        """Delegate to HF's get_rope_index method"""
        return self.hf_model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass that adapts our interface to HF's interface
        """
        # Convert our format to HF format
        hf_inputs = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'attention_mask': attention_mask,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict if return_dict is not None else True,
        }
        
        # Remove None values
        hf_inputs = {k: v for k, v in hf_inputs.items() if v is not None}
        
        # Call HF model
        outputs = self.hf_model(**hf_inputs)
        
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ):
        """Generation method that delegates to HF"""
        inputs = {}
        if input_ids is not None:
            inputs['input_ids'] = input_ids
        if pixel_values is not None:
            inputs['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            inputs['image_grid_thw'] = image_grid_thw
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask
            
        return self.hf_model.generate(**inputs, **generation_kwargs)
    
    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        self.hf_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        self.hf_model.eval()
        return self
    
    def to(self, *args, **kwargs):
        """Move model to device/dtype"""
        super().to(*args, **kwargs)
        self.hf_model.to(*args, **kwargs)
        return self
    
    def named_parameters(self, *args, **kwargs):
        """Return named parameters from HF model"""
        return self.hf_model.named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """Return parameters from HF model"""
        return self.hf_model.parameters(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict from HF model"""
        return self.hf_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into HF model"""
        return self.hf_model.load_state_dict(*args, **kwargs)
    
    def freeze_base_model(self):
        """Freeze base model parameters for LoRA fine-tuning"""
        for name, param in self.hf_model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    @property
    def device(self):
        """Get model device"""
        return next(self.hf_model.parameters()).device
    
    @property
    def dtype(self):
        """Get model dtype"""
        return next(self.hf_model.parameters()).dtype


def load_hf_qwen2_5_vl_model(model_path: str, **kwargs) -> HFQwen2_5_VLForCausalLM:
    """
    Load HF Qwen2.5-VL model with our config wrapper
    """
    # Create a compatible config
    class CompatConfig:
        def __init__(self, model_path):
            self.model = model_path
            self.hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    config = CompatConfig(model_path)
    return HFQwen2_5_VLForCausalLM(config, **kwargs) 