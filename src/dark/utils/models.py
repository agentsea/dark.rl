
import logging
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from dark.models.hf_qwen2_5_vl import load_hf_qwen2_5_vl_model

# Supported models dictionary
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    # Vision-Language Models
    "Qwen/Qwen2.5-VL-3B-Instruct": {"type": "vl", "engine": "hf"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"type": "vl", "engine": "hf"},
    "Qwen/Qwen2.5-VL-32B-Instruct": {"type": "vl", "engine": "hf"},
    
    # Text-only Models
    "Qwen/Qwen3-0.6B": {"type": "text", "engine": "dark"},
    "Qwen/Qwen3-1.7B": {"type": "text", "engine": "dark"},
    "Qwen/Qwen3-4B": {"type": "text", "engine": "dark"},
    "Qwen/Qwen3-8B": {"type": "text", "engine": "dark"},
    "Qwen/Qwen3-14B": {"type": "text", "engine": "dark"},
    "Qwen/Qwen3-32B": {"type": "text", "engine": "dark"},
    
    # Mixture-of-Experts (MoE) Models
    "Qwen/Qwen3-MoE-15B-A2B": {"type": "moe", "engine": "dark"},
    "Qwen/Qwen3-MoE-32B-A2B": {"type": "moe", "engine": "dark"},
}

def get_supported_models() -> List[str]:
    """Return a list of all supported model names."""
    return list(SUPPORTED_MODELS.keys())

def get_model_and_tokenizer(
    model_name: str, 
    **kwargs: Any
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from HuggingFace, with specific optimizations.
    
    Args:
        model_name: The name/path of the model to load.
        **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained.
        
    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    if model_name not in get_supported_models():
        raise ValueError(f"Model '{model_name}' is not supported.")

    model_info = SUPPORTED_MODELS[model_name]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model based on its type
    if model_info['type'] == 'vl':
        # Use the specialized loader for Vision-Language models
        model = load_hf_qwen2_5_vl_model(model_name, **kwargs)
        # The processor is part of the wrapped model, tokenizer is on processor
        tokenizer = model.processor.tokenizer
    else:
        # Default arguments for model loading
        default_args = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        # Update with any user-provided arguments
        default_args.update(kwargs)
        
        # Load model for text-based models
        model = AutoModelForCausalLM.from_pretrained(model_name, **default_args)
    
    logging.info(f"Loaded model '{model_name}' with tokenizer.")
    
    return model, tokenizer 