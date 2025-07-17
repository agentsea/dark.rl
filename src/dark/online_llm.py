import asyncio
import logging
import time
import warnings
import os
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Union, AsyncGenerator, Tuple
import torch
import bitsandbytes as bnb
from concurrent.futures import ThreadPoolExecutor
import math
from dataclasses import dataclass

# Rich imports for pretty printing stats
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*generation_config.*default values have been modified.*")
warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

# Set transformers logging to reduce warnings
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

from dark.llm import LLM
from dark.sampling_params import SamplingParams
from dark.engine.sequence import Sequence

# Import KTO trainer
try:
    from dark.trainers.kto import KTOTrainer
    KTO_AVAILABLE = True
except ImportError:
    KTO_AVAILABLE = False

# Add import for HF wrapper
try:
    from dark.models.hf_qwen2_5_vl import load_hf_qwen2_5_vl_model
    HF_WRAPPER_AVAILABLE = True
except ImportError:
    HF_WRAPPER_AVAILABLE = False

# Add import for Qwen3MoE upstream model
try:
    from dark.models.qwen3_moe import create_qwen3_moe_model
    QWEN3_MOE_AVAILABLE = True
except ImportError:
    QWEN3_MOE_AVAILABLE = False

# Add import for Qwen3 HF model
try:
    from dark.models.qwen3_hf import Qwen3HFForCausalLM
    QWEN3_HF_AVAILABLE = True
except ImportError:
    QWEN3_HF_AVAILABLE = False


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 8
    auto_batch_size: bool = True
    memory_threshold: float = 0.8  # GPU memory threshold before reducing batch size
    min_batch_size: int = 1
    # Note: concurrent_adapters removed - adapters now train sequentially for isolation


class BatchManager:
    """Utility class for managing batch operations and memory."""
    
    def __init__(self, batch_config: BatchConfig):
        self.config = batch_config
        self._current_batch_size = batch_config.max_batch_size
    
    def get_optimal_batch_size(self, total_items: int, item_complexity: float = 1.0) -> int:
        """Calculate optimal batch size based on available memory and item complexity."""
        if not self.config.auto_batch_size:
            return min(self.config.max_batch_size, total_items)
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            memory_used = torch.cuda.memory_allocated(0)
            memory_usage = memory_used / gpu_memory
            
            if memory_usage > self.config.memory_threshold:
                self._current_batch_size = max(
                    self.config.min_batch_size,
                    int(self._current_batch_size * 0.7)
                )
            elif memory_usage < 0.5:
                self._current_batch_size = min(
                    self.config.max_batch_size,
                    int(self._current_batch_size * 1.2)
                )
        
        # Adjust for item complexity
        adjusted_batch_size = max(
            self.config.min_batch_size,
            int(self._current_batch_size / item_complexity)
        )
        
        return min(adjusted_batch_size, total_items)
    
    def create_batches(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """Split items into batches."""
        if not items:  # Handle empty list
            return []
            
        if batch_size is None:
            batch_size = self.get_optimal_batch_size(len(items))
        
        if batch_size <= 0:  # Ensure valid batch size
            batch_size = 1
        
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        return batches
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_cached = torch.cuda.memory_reserved(0)
        
        return {
            "gpu_available": True,
            "total_memory_gb": gpu_memory / (1024**3),
            "allocated_memory_gb": memory_allocated / (1024**3),
            "cached_memory_gb": memory_cached / (1024**3),
            "usage_percentage": memory_allocated / gpu_memory * 100,
            "available_percentage": (gpu_memory - memory_allocated) / gpu_memory * 100
        }


SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-MoE-15B-A2B",
    "Qwen/Qwen3-MoE-32B-A2B",
]


class OnlineLLM:
    """An online LLM that can be used to chat, learn, and fine-tune with LoRA adapters.
    
    This class integrates the functionality from the training test script, including:
    - LoRA fine-tuning with async capabilities
    - Streaming generation
    - Multiple LoRA adapter management
    - 8-bit Adam optimizer support
    - Support for both HuggingFace and custom Dark implementations
    - Thinking mode for supported models (Qwen3)
    - Enhanced batch training and inference capabilities
    
    Args:
        model: The model name/path to load
        engine: Implementation to use - "hf" for HuggingFace, "dark" for custom
                Defaults to "hf" for VL models, "dark" for text-only models
        thinking_mode: Enable thinking mode for supported models (Qwen3). When enabled,
                      the model can use thinking tokens (<think></think>) to reason
                      step-by-step before providing answers. Default: True
        batch_config: Configuration for batch processing operations
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 16,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        offload_lora_to_cpu: bool = True,
        use_adam8bit: bool = True,
        enforce_eager: bool = True,
        engine: str = None,  # "hf" for HuggingFace, "dark" for custom implementation
        attn_implementation: str = "flash_attention_2",  # Attention implementation for HF models (default: Flash Attention 2)
        thinking_mode: bool = True,  # Enable thinking mode for supported models (Qwen3)
        batch_config: Optional[BatchConfig] = None,  # Batch processing configuration
    ):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported. Supported models: {SUPPORTED_MODELS}")

        self.model_path = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.offload_lora_to_cpu = offload_lora_to_cpu
        self.use_adam8bit = use_adam8bit
        self.thinking_mode = thinking_mode

        # Initialize batch manager
        self.batch_config = batch_config or BatchConfig()
        self.batch_manager = BatchManager(self.batch_config)

        # Handle engine parameter with smart defaults
        is_vl_model = "VL" in model
        is_moe_model = "MoE" in model
        
        # Set default engine based on model type
        if engine is None:
            if is_vl_model:
                engine = "hf"  # VL models default to HF (no custom implementation available)
            elif is_moe_model:
                engine = "dark"  # MoE models default to custom Dark implementation
            else:
                engine = "dark"  # Text models default to custom Dark implementation
        
        # Validate engine parameter
        if engine not in ["hf", "dark"]:
            raise ValueError(f"engine must be 'hf' or 'dark', got '{engine}'")
        
        self.engine = engine
        self.using_hf = (engine == "hf")
        
        if engine == "hf":
            # Use HF implementation
            if is_vl_model:
                # VL models use the VL wrapper
                if not HF_WRAPPER_AVAILABLE:
                    raise ImportError("HF wrapper not available. Please ensure src.dark.models.hf_qwen2_5_vl is properly installed.")
                
                logging.debug(f"Using HuggingFace VL engine for {model} with {attn_implementation} attention")
                # Initialize the HF VL wrapper
                self.hf_model = load_hf_qwen2_5_vl_model(
                    model,
                    attn_implementation=attn_implementation,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha
                )
                self.tokenizer = self.hf_model.processor.tokenizer
                self.llm = None
                self.using_hf = True
            else:
                # Text-only models use the Qwen3 HF implementation
                try:
                    from dark.models.qwen3_hf import create_qwen3_hf_model
                    from dark.config import Config
                    
                    logging.debug(f"Using HuggingFace text engine for {model} with {attn_implementation} attention")
                    
                    # Create config for HF model
                    config = Config(model=model)
                    config.model_name = model  # Add model_name attribute for HF compatibility
                    self.hf_model = create_qwen3_hf_model(
                        config,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        thinking_mode=thinking_mode
                    )
                    self.tokenizer = self.hf_model.tokenizer
                    self.llm = None
                    self.using_hf = True
                    
                except ImportError as e:
                    logging.warning(f"HF text engine not available ({e}), falling back to custom engine")
                    # Fallback to custom implementation
                    self.llm = LLM(
                        model,
                        enforce_eager=enforce_eager,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha
                    )
                    self.tokenizer = self.llm.tokenizer
                    self.hf_model = None
                    self.using_hf = False
        else:
            # Use custom Dark engine
            if is_moe_model:
                # Use Qwen3MoE implementation
                if not QWEN3_MOE_AVAILABLE:
                    raise ImportError("Qwen3MoE engine not available. Please ensure src.dark.models.qwen3_moe is properly installed.")
                
                logging.debug(f"Using custom Dark MoE engine for {model}")
                from dark.config import Config
                
                # Create config for MoE model
                config = Config(model=model)
                config.model_name = model  # Add model_name attribute for compatibility
                self.hf_model = create_qwen3_moe_model(
                    config,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha
                )
                self.tokenizer = self.hf_model.get_input_embeddings().tokenizer if hasattr(self.hf_model.get_input_embeddings(), 'tokenizer') else None
                
                # If we don't have a tokenizer from the model, load it separately
                if self.tokenizer is None:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                
                # Ensure tokenizer has required attributes
                if not hasattr(self.tokenizer, 'eos_token_id') or self.tokenizer.eos_token_id is None:
                    self.tokenizer.eos_token_id = self.tokenizer.pad_token_id
                if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.llm = None
                self.using_hf = True  # We treat MoE as HF-like for interface purposes
            else:
                logging.debug(f"Using custom Dark engine for {model}")
                self.llm = LLM(
                    model,
                    enforce_eager=enforce_eager,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha
                )
                self.tokenizer = self.llm.tokenizer
                self.hf_model = None
                self.using_hf = False

        # LoRA adapter management
        self.lora_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.opt_states: Dict[str, Dict] = {}
        self.training_tasks: Dict[str, asyncio.Task] = {}
        
        # Async coordination
        self.lock = asyncio.Lock()
        self.infer_stream = torch.cuda.Stream()

        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=True
        )

    def get_lora_state(self, to_cpu: bool = None) -> Dict[str, torch.Tensor]:
        """Clone and return all LoRA parameters of the model.

        Args:
            to_cpu: If True, the returned tensors will be on the CPU to save VRAM.
                   If None, uses the instance default.
        """
        if to_cpu is None:
            to_cpu = self.offload_lora_to_cpu
        
        if self.using_hf:
            return {
                n: p.detach().clone().to("cpu" if to_cpu else p.device) 
                for n, p in self.hf_model.named_parameters() 
                if "lora_" in n
            }
        else:
            return {
                n: p.detach().clone().to("cpu" if to_cpu else p.device) 
                for n, p in self.llm.model_runner.model.named_parameters() 
                if "lora_" in n
            }

    def get_lora_save_path(self) -> Optional[Path]:
        """Get the directory path for saving LoRA adapters from environment variable."""
        lora_path = os.getenv('LORA_SAVE_PATH')
        if lora_path:
            path = Path(lora_path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return None

    def save_lora_to_disk(self, adapter: str, lora_state: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """Save a LoRA adapter to disk.
        
        Args:
            adapter: Name of the adapter to save
            lora_state: Optional LoRA state dict. If None, gets current state from memory.
            
        Returns:
            True if saved successfully, False otherwise
        """
        save_path = self.get_lora_save_path()
        if not save_path:
            logging.debug(f"LORA_SAVE_PATH not set, skipping disk save for {adapter}")
            return False
        
        try:
            if lora_state is None:
                if adapter in self.lora_states:
                    lora_state = self.lora_states[adapter]
                else:
                    logging.warning(f"No LoRA state found for adapter {adapter}")
                    return False
            
            # Save LoRA state and metadata
            save_data = {
                'lora_state': lora_state,
                'metadata': {
                    'adapter_name': adapter,
                    'model_path': self.model_path,
                    'lora_rank': self.lora_rank,
                    'lora_alpha': self.lora_alpha,
                    'saved_at': time.time(),
                }
            }
            
            adapter_file = save_path / f"{adapter}.pt"
            torch.save(save_data, adapter_file)
            logging.debug(f"Saved LoRA adapter {adapter} to {adapter_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save LoRA adapter {adapter} to disk: {e}")
            return False

    def load_lora_from_disk(self, adapter: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load a LoRA adapter from disk.
        
        Args:
            adapter: Name of the adapter to load
            
        Returns:
            LoRA state dict if found and loaded successfully, None otherwise
        """
        save_path = self.get_lora_save_path()
        if not save_path:
            return None
        
        adapter_file = save_path / f"{adapter}.pt"
        if not adapter_file.exists():
            return None
        
        try:
            saved_data = torch.load(adapter_file, map_location='cpu')
            
            # Validate metadata compatibility
            metadata = saved_data.get('metadata', {})
            if metadata.get('model_path') != self.model_path:
                logging.warning(f"LoRA adapter {adapter} was trained on different model: "
                              f"{metadata.get('model_path')} vs {self.model_path}")
            
            if (metadata.get('lora_rank') != self.lora_rank or 
                metadata.get('lora_alpha') != self.lora_alpha):
                logging.warning(f"LoRA adapter {adapter} has different rank/alpha: "
                              f"saved={metadata.get('lora_rank')}/{metadata.get('lora_alpha')} "
                              f"current={self.lora_rank}/{self.lora_alpha}")
            
            lora_state = saved_data['lora_state']
            logging.debug(f"Loaded LoRA adapter {adapter} from {adapter_file}")
            return lora_state
            
        except Exception as e:
            logging.error(f"Failed to load LoRA adapter {adapter} from disk: {e}")
            return None

    def list_disk_adapters(self) -> List[str]:
        """List all LoRA adapters available on disk.
        
        Returns:
            List of adapter names found on disk
        """
        save_path = self.get_lora_save_path()
        if not save_path or not save_path.exists():
            return []
        
        adapters = []
        for file_path in save_path.glob("*.pt"):
            adapter_name = file_path.stem
            adapters.append(adapter_name)
        
        return sorted(adapters)

    def load_lora_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load LoRA tensors into the model (in-place).

        If `state` is empty, we leave the existing randomly-initialized LoRA weights
        untouched so that training can make progress.
        
        This method ensures proper adapter isolation by first resetting ALL LoRA 
        parameters to zero, then loading only the specified adapter's weights.
        """
        t0 = time.perf_counter()
        if not state:
            logging.debug(f"[metric] lora_load_ms=0.00 (no state)")
            return
        
        with torch.no_grad():
            if self.using_hf:
                # First, reset ALL LoRA parameters to zero for clean isolation
                for n, p in self.hf_model.named_parameters():
                    if "lora_" in n:
                        p.zero_()
                
                # Then load only the specified adapter's weights
                for n, p in self.hf_model.named_parameters():
                    if "lora_" in n and n in state:
                        p.copy_(state[n].to(p.device))
            else:
                # First, reset ALL LoRA parameters to zero for clean isolation
                for n, p in self.llm.model_runner.model.named_parameters():
                    if "lora_" in n:
                        p.zero_()
                
                # Then load only the specified adapter's weights
                for n, p in self.llm.model_runner.model.named_parameters():
                    if "lora_" in n and n in state:
                        p.copy_(state[n].to(p.device))
        
        dt_ms = (time.perf_counter() - t0) * 1000
        logging.debug(f"[metric] lora_load_ms={dt_ms:.2f}")

    def ensure_lora_adapter(self, adapter: str) -> bool:
        """Ensure a LoRA adapter is available in memory, loading from disk if necessary.
        
        Args:
            adapter: Name of the adapter to ensure is loaded
            
        Returns:
            True if adapter is available (in memory or loaded from disk), False if new/not found
        """
        # Check if already in memory
        if adapter in self.lora_states:
            logging.debug(f"LoRA adapter {adapter} found in memory")
            return True
        
        # Try to load from disk
        disk_state = self.load_lora_from_disk(adapter)
        if disk_state is not None:
            self.lora_states[adapter] = disk_state
            logging.info(f"LoRA adapter {adapter} loaded from disk")
            return True
        
        # Adapter not found - it's new
        logging.debug(f"LoRA adapter {adapter} not found in memory or disk (new adapter)")
        return False

    def reset_lora_weights(self) -> None:
        """Reset all LoRA parameters to zero.
        
        Useful for ensuring clean adapter isolation or returning to base model.
        """
        with torch.no_grad():
            if self.using_hf:
                for n, p in self.hf_model.named_parameters():
                    if "lora_" in n:
                        p.zero_()
            else:
                for n, p in self.llm.model_runner.model.named_parameters():
                    if "lora_" in n:
                        p.zero_()
        logging.debug("All LoRA weights reset to zero")
    
    def initialize_lora_weights(self) -> None:
        """Initialize LoRA parameters with proper random initialization.
        
        Follows standard LoRA initialization:
        - lora_A: random normal with small std (0.02)  
        - lora_B: zeros (so initial contribution is zero)
        
        This provides a good starting point for learning while maintaining isolation.
        """
        with torch.no_grad():
            if self.using_hf:
                for n, p in self.hf_model.named_parameters():
                    if "lora_" in n:
                        if "lora_A" in n:
                            # Initialize A with small random values
                            p.normal_(0, 0.02)
                        elif "lora_B" in n:
                            # Initialize B with zeros (standard LoRA practice)
                            p.zero_()
            else:
                for n, p in self.llm.model_runner.model.named_parameters():
                    if "lora_" in n:
                        if "lora_A" in n:
                            # Initialize A with small random values
                            p.normal_(0, 0.02)
                        elif "lora_B" in n:
                            # Initialize B with zeros (standard LoRA practice)
                            p.zero_()
        logging.debug("LoRA weights initialized with proper random values")

    async def batch_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        adapter: Optional[str] = None,
        images: Optional[List[List[Any]]] = None,  # Support multiple images per prompt
        batch_size: Optional[int] = None,  # Optional override for batch size
    ) -> List[str]:
        """Generate complete responses for a batch of prompts (batch processing, not streaming).
        
        This method processes multiple prompts efficiently and returns complete responses.
        Now supports true batch processing for improved performance.
        
        For single prompt generation: Use AsyncOnlineLLM.generate() or OnlineLLM.generate()
        For streaming generation: Use AsyncOnlineLLM.stream() or OnlineLLM.stream()
        
        Args:
            prompts: List of input prompts to process
            sampling_params: Generation parameters (temperature, max_tokens, etc.)
            adapter: Optional LoRA adapter name to use
            images: Optional list of images (one list per prompt) for VL models
            batch_size: Optional batch size override (uses auto-calculation if None)
            
        Returns:
            List of generated text responses (one per prompt)
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        gen_start = time.perf_counter()
        
        # Load the appropriate LoRA adapter if specified
        if adapter:
            async with self.lock:
                adapter_found = self.ensure_lora_adapter(adapter)
                if adapter_found:
                    self.load_lora_state(self.lora_states[adapter])
                else:
                    logging.info(f"Using new LoRA adapter: {adapter}")

        # Use efficient batch processing
        return await self.efficient_batch_generate(
            prompts, sampling_params, adapter, images, batch_size
        )

    async def efficient_batch_generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        adapter: Optional[str] = None,
        images: Optional[List[List[Any]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Efficiently generate responses for batches with memory management."""
        # Handle empty prompts
        if not prompts:
            return []
            
        gen_start = time.perf_counter()
        
        # Determine optimal batch size
        if batch_size is None:
            # Consider images as complexity factor
            has_images = images is not None and any(img is not None for img in images)
            complexity = 2.0 if has_images else 1.0
            batch_size = self.batch_manager.get_optimal_batch_size(len(prompts), complexity)
        
        # Create batches
        prompt_batches = self.batch_manager.create_batches(prompts, batch_size)
        image_batches = None
        if images:
            image_batches = self.batch_manager.create_batches(images, batch_size)
        
        all_outputs = []
        
        async with self.lock:
            if hasattr(self.hf_model, 'eval'):
                self.hf_model.eval()
        
        for batch_idx, prompt_batch in enumerate(prompt_batches):
            image_batch = image_batches[batch_idx] if image_batches else None
            
            if self.using_hf:
                batch_outputs = await self._process_hf_batch(
                    prompt_batch, image_batch, sampling_params
                )
            else:
                batch_outputs = await self._process_custom_batch(
                    prompt_batch, sampling_params
                )
            
            all_outputs.extend(batch_outputs)
            
            # Memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            await asyncio.sleep(0)  # Yield to other tasks
        
        total_gen_ms = (time.perf_counter() - gen_start) * 1000
        logging.debug(f"[metric] batch_generate_total_ms={total_gen_ms:.2f} batches={len(prompt_batches)}")
        return all_outputs

    async def _process_hf_batch(
        self,
        prompts: List[str],
        images: Optional[List[List[Any]]],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Process a batch of prompts using HuggingFace models with true batching."""
        outputs = []
        
        # Check if we can do true batch processing
        has_images = images is not None and any(img is not None for img in images)
        has_vl_processor = hasattr(self.hf_model, 'processor')
        
        if not has_images and not has_vl_processor:
            # Pure text batch processing - most efficient
            try:
                # Check if this is a Qwen3 HF model that expects messages
                import inspect
                generate_signature = inspect.signature(self.hf_model.generate)
                
                if 'messages' in generate_signature.parameters:
                    # Qwen3 HF model with chat interface - process individually for now
                    # TODO: Implement true batch processing for Qwen3 message interface
                    raise ValueError("Qwen3 message interface requires sequential processing")
                else:
                    # Standard HF model with tokenized inputs
                    async with self.lock:
                        inputs = self.tokenizer(
                            prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True
                        ).to(self.hf_model.device)
                        
                        with torch.cuda.stream(self.infer_stream):
                            generated_ids = self.hf_model.generate(
                                **inputs,
                                max_new_tokens=sampling_params.max_tokens,
                                do_sample=sampling_params.temperature > 0,
                                temperature=sampling_params.temperature if sampling_params.temperature > 0 else None,
                                pad_token_id=self.tokenizer.eos_token_id,
                                num_return_sequences=1,
                            )
                        
                        # Decode only the new tokens for each sequence
                        input_lengths = inputs['input_ids'].shape[1]
                        for i, generated_seq in enumerate(generated_ids):
                            new_tokens = generated_seq[input_lengths:]
                            output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            outputs.append(output)
                    
                    return outputs
                
            except Exception as e:
                logging.warning(f"Batch processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
        
        # Sequential processing for complex cases (VL models, different formats)
        for i, prompt in enumerate(prompts):
            prompt_images = images[i] if images and i < len(images) else None
            output = await self._process_single_hf_prompt(prompt, prompt_images, sampling_params)
            outputs.append(output)
        
        return outputs

    async def _process_single_hf_prompt(
        self,
        prompt: str,
        prompt_images: Optional[List[Any]],
        sampling_params: SamplingParams,
    ) -> str:
        """Process a single prompt with HF model (handles all model types)."""
        async with self.lock:
            # Check if this is a VL model with processor
            if hasattr(self.hf_model, 'processor') and prompt_images is not None and len(prompt_images) > 0:
                # Process multiple images + text with the processor (VL model)
                content = []
                for img in prompt_images:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": prompt})
                
                messages = [{
                    "role": "user",
                    "content": content
                }]
                
                # Apply chat template and process
                text_prompt = self.hf_model.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.hf_model.processor(
                    text=[text_prompt], images=prompt_images, return_tensors="pt"
                )
                inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
                
                with torch.cuda.stream(self.infer_stream):
                    generated_ids = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=sampling_params.max_tokens,
                        do_sample=sampling_params.temperature > 0,
                        temperature=sampling_params.temperature if sampling_params.temperature > 0 else None,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
            elif hasattr(self.hf_model, 'generate') and hasattr(self.hf_model.generate, '__call__'):
                # Check if this is a Qwen3 HF model that expects messages
                import inspect
                generate_signature = inspect.signature(self.hf_model.generate)
                
                if 'messages' in generate_signature.parameters:
                    # Qwen3 HF model with chat interface
                    messages = [{"role": "user", "content": prompt}]
                    
                    output = self.hf_model.generate(
                        messages=messages,
                        max_tokens=sampling_params.max_tokens,
                        temperature=sampling_params.temperature if sampling_params.temperature > 0 else None,
                    )
                    
                    # Output is already a string for Qwen3 models
                    if isinstance(output, str):
                        return output
                    else:
                        # Fallback if output format is unexpected
                        return str(output)
                else:
                    # Standard HF model with tokenized inputs
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
                    
                    with torch.cuda.stream(self.infer_stream):
                        generated_ids = self.hf_model.generate(
                            **inputs,
                            max_new_tokens=sampling_params.max_tokens,
                            do_sample=sampling_params.temperature > 0,
                            temperature=sampling_params.temperature if sampling_params.temperature > 0 else None,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode only the new tokens
                    new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                    output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
            else:
                # Fallback: Standard HF model interface
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
                
                with torch.cuda.stream(self.infer_stream):
                    generated_ids = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=sampling_params.max_tokens,
                        do_sample=sampling_params.temperature > 0,
                        temperature=sampling_params.temperature if sampling_params.temperature > 0 else None,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return output

    async def _process_custom_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Process a batch using custom Dark implementation."""
        seqs = []
        for prompt in prompts:
            # For each prompt, create `sampling_params.n` sequences
            for _ in range(sampling_params.n):
                seqs.append(Sequence(self.tokenizer.encode(prompt), sampling_params))

        async with self.lock:
            self.llm.eval()
            for seq in seqs:
                self.llm.scheduler.add(seq)

        outs = ["" for _ in seqs]
        emitted = [0 for _ in seqs]
        
        while any(not seq.is_finished for seq in seqs):
            step_t0 = time.perf_counter()
            async with self.lock:
                self.llm.eval()
                with torch.cuda.stream(self.infer_stream):
                    self.llm.step()
            step_dt_ms = (time.perf_counter() - step_t0) * 1000
            logging.debug(f"[metric] llm_step_ms={step_dt_ms:.2f}")

            for i, seq in enumerate(seqs):
                while seq.num_completion_tokens > emitted[i]:
                    tid = seq.completion_token_ids[emitted[i]]
                    outs[i] += self.tokenizer.decode([tid], skip_special_tokens=True)
                    emitted[i] += 1
            
            await asyncio.sleep(0)  # Yield to other async tasks

        return outs

    async def batch_fine_tune(
        self,
        adapter_examples: Dict[str, List[Dict[str, Union[str, Any]]]],  # {adapter_name: examples}
        steps: int = 5,
        lr: float = 1e-4,
        max_concurrent: Optional[int] = None,  # Deprecated - kept for API compatibility
        trainer: Optional[Any] = None,
    ) -> Dict[str, Dict]:
        """Fine-tune adapters sequentially with batched examples.
        
        Note: This method now trains adapters sequentially (not concurrently) to ensure
        proper adapter isolation and prevent LoRA parameter interference.
        
        Args:
            adapter_examples: Dictionary mapping adapter names to their training examples
            steps: Training steps per adapter
            lr: Learning rate
            max_concurrent: Deprecated - adapters are now trained sequentially
            trainer: Optional trainer (e.g., KTOTrainer)
            
        Returns:
            Dictionary mapping adapter names to their optimizer states
        """
        # Handle empty training data
        if not adapter_examples:
            return {}
            
        t_batch_ft_start = time.perf_counter()
        
        all_opt_states = {}
        
        # Train adapters sequentially for perfect isolation
        for adapter_name, examples in adapter_examples.items():
            logging.info(f"Training adapter: {adapter_name} with {len(examples)} examples")
            
            try:
                result = await self._train_single_adapter_batch(
                    adapter_name, examples, steps, lr, trainer
                )
                all_opt_states[adapter_name] = result
                
                # Memory cleanup between adapters
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                await asyncio.sleep(0.05)  # Brief pause between adapters
                
            except Exception as e:
                logging.error(f"Training failed for adapter {adapter_name}: {e}")
                all_opt_states[adapter_name] = {}
        
        total_batch_ft_ms = (time.perf_counter() - t_batch_ft_start) * 1000
        logging.debug(f"[metric] batch_fine_tune_total_ms={total_batch_ft_ms:.2f}")
        
        return all_opt_states

    async def _train_single_adapter_batch(
        self,
        adapter: str,
        examples: List[Dict[str, Union[str, Any]]],
        steps: int,
        lr: float,
        trainer: Optional[Any],
    ) -> Dict:
        """Train a single adapter with its batch of examples."""
        # Wait for any existing training task for this adapter
        if adapter in self.training_tasks:
            await self.training_tasks.pop(adapter)
        
        # Always start training from a clean slate, then load this adapter's existing state
        async with self.lock:
            if adapter in self.lora_states:
                # Existing adapter: load_lora_state will reset then load
                self.load_lora_state(self.lora_states[adapter])
            else:
                # New adapter: initialize with proper LoRA random weights for effective learning
                self.initialize_lora_weights()
        
        # Run fine-tuning
        is_moe_model = "MoE" in self.model_path
        opt_state = await self.fine_tune(examples, adapter, steps, lr, is_moe_model, trainer)
        
        # Save updated states
        async with self.lock:
            self.lora_states[adapter] = self.get_lora_state()
            self.opt_states[adapter] = opt_state
            
            # Auto-save to disk if LORA_SAVE_PATH is set
            self.save_lora_to_disk(adapter, self.lora_states[adapter])
        
        return opt_state

    async def memory_aware_batch_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        adapter: Optional[str] = None,
        images: Optional[List[List[Any]]] = None,
        target_memory_usage: float = 0.7,  # Target GPU memory usage percentage
    ) -> List[str]:
        """Generate responses with dynamic batch sizing based on memory usage.
        
        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters
            adapter: Optional LoRA adapter
            images: Optional images for VL models
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            
        Returns:
            List of generated responses
        """
        if not torch.cuda.is_available():
            # Fallback to regular batch generation if no GPU
            return await self.batch_generate(prompts, sampling_params, adapter, images)
        
        # Start with a conservative batch size
        initial_batch_size = max(1, len(prompts) // 4)
        
        # Monitor memory and adjust batch size dynamically
        optimal_batch_size = initial_batch_size
        memory_info = self.batch_manager.get_memory_info()
        
        if memory_info["usage_percentage"] / 100 < target_memory_usage:
            # We have room to increase batch size
            optimal_batch_size = min(
                self.batch_config.max_batch_size,
                int(initial_batch_size * (target_memory_usage / (memory_info["usage_percentage"] / 100)))
            )
        
        logging.debug(f"Using batch size {optimal_batch_size} for {len(prompts)} prompts "
                     f"(memory usage: {memory_info['usage_percentage']:.1f}%)")
        
        return await self.batch_generate(
            prompts, sampling_params, adapter, images, batch_size=optimal_batch_size
        )

    async def parallel_adapter_generate(
        self,
        prompts: List[str],
        adapters: List[str],
        sampling_params: Optional[SamplingParams] = None,
        images: Optional[List[List[Any]]] = None,
    ) -> Dict[str, List[str]]:
        """Generate responses for the same prompts using multiple adapters in parallel.
        
        Args:
            prompts: List of input prompts
            adapters: List of adapter names to use
            sampling_params: Generation parameters
            images: Optional images for VL models
            
        Returns:
            Dictionary mapping adapter names to their generated responses
        """
        # Create tasks for each adapter
        tasks = []
        for adapter in adapters:
            task = asyncio.create_task(
                self.batch_generate(prompts, sampling_params, adapter, images)
            )
            tasks.append((adapter, task))
        
        # Wait for all generations to complete
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Collect results
        adapter_results = {}
        for (adapter, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logging.error(f"Generation failed for adapter {adapter}: {result}")
                adapter_results[adapter] = ["ERROR"] * len(prompts)
            else:
                adapter_results[adapter] = result
        
        return adapter_results

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics and memory information."""
        memory_info = self.batch_manager.get_memory_info()
        
        stats = {
            "batch_config": {
                "max_batch_size": self.batch_config.max_batch_size,
                "auto_batch_size": self.batch_config.auto_batch_size,
                "memory_threshold": self.batch_config.memory_threshold,
                "min_batch_size": self.batch_config.min_batch_size,
            },
            "current_batch_size": self.batch_manager._current_batch_size,
            "memory_info": memory_info,
            "adapters": {
                "total_adapters": len(self.lora_states),
                "active_training_tasks": len(self.training_tasks),
                "adapter_names": list(self.lora_states.keys()),
            }
        }
        
        return stats

    def update_batch_config(self, **kwargs) -> None:
        """Update batch configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update (max_batch_size, memory_threshold, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.batch_config, key):
                setattr(self.batch_config, key, value)
                logging.debug(f"Updated batch config: {key} = {value}")
            else:
                logging.warning(f"Unknown batch config parameter: {key}")

    async def fine_tune(
        self,
        examples: List[Dict[str, Union[str, Any]]],  # Updated to support images
        adapter: str,
        steps: int = 5,
        lr: float = 1e-4,
        is_moe_model: bool = False,
        trainer: Optional[Any] = None,  # KTO trainer or other trainers
    ) -> Dict:
        """Fine-tune the model on a batch of prompt/response pairs using LoRA."""
        t_ft_start = time.perf_counter()

        # Data preparation
        prompts = [ex["prompt"] for ex in examples]
        responses = [ex["response"] for ex in examples]
        images = [ex.get("images") for ex in examples]  # Extract images if present (now supports multiple)
        
        if self.using_hf and any(img is not None for img in images):
            # Vision-language training with HF processor
            device = self.hf_model.device
            
            # Process each example with image+text
            processed_inputs = []
            all_labels = []
            all_images = []
            
            for i, (prompt, response, example_images) in enumerate(zip(prompts, responses, images)):
                if example_images is not None and len(example_images) > 0:
                    # Create vision-language conversation with multiple images
                    content = []
                    for img in example_images:
                        content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": prompt})
                    
                    messages = [{
                        "role": "user", 
                        "content": content
                    }, {
                        "role": "assistant",
                        "content": response
                    }]
                    
                    # Apply chat template
                    text_input = self.hf_model.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    
                    # Process with processor
                    inputs = self.hf_model.processor(
                        text=[text_input], images=example_images, return_tensors="pt"
                    )
                    
                    # Create labels for loss calculation
                    # We need to mask the prompt tokens and only compute loss on response tokens
                    input_ids = inputs['input_ids'][0]
                    labels_for_example = input_ids.clone()
                    
                    # Find where the assistant response starts
                    # This is a simplified approach - in practice you'd want more robust parsing
                    assistant_token = "<|im_start|>assistant\n"
                    assistant_tokens = self.tokenizer.encode(assistant_token, add_special_tokens=False)
                    
                    # Mask everything before the assistant response
                    if len(assistant_tokens) > 0:
                        for j in range(len(input_ids) - len(assistant_tokens)):
                            if input_ids[j:j+len(assistant_tokens)].tolist() == assistant_tokens:
                                labels_for_example[:j+len(assistant_tokens)] = -100
                                break
                    
                    processed_inputs.append(inputs)
                    all_labels.append(labels_for_example)
                    all_images.append(example_images)
                else:
                    # Text-only training (fallback)
                    prompt_ids = self.tokenizer(prompt, padding=False).input_ids
                    response_ids = self.tokenizer(response, padding=False).input_ids
                    prompt_len = len(prompt_ids)
                    input_id = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
                    label = [-100] * prompt_len + response_ids + [self.tokenizer.eos_token_id]
                    
                    inputs = {"input_ids": torch.tensor([input_id])}
                    processed_inputs.append(inputs)
                    all_labels.append(torch.tensor(label))
                    all_images.append(None)
            
            # For now, we'll train on one example at a time due to the complexity of batching vision inputs
            input_ids = processed_inputs[0]['input_ids'].to(device)
            labels = all_labels[0].unsqueeze(0).to(device)
            pixel_values = processed_inputs[0].get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            image_grid_thw = processed_inputs[0].get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
        else:
            # Text-only training (original implementation)
            prompt_ids = self.tokenizer(prompts, padding=False).input_ids
            response_ids = self.tokenizer(responses, padding=False).input_ids
            
            input_ids = []
            labels = []
            for i in range(len(prompts)):
                prompt_len = len(prompt_ids[i])
                input_id = prompt_ids[i] + response_ids[i] + [self.tokenizer.eos_token_id]
                label = [-100] * prompt_len + response_ids[i] + [self.tokenizer.eos_token_id]
                input_ids.append(torch.tensor(input_id))
                labels.append(torch.tensor(label))

            device = self.hf_model.device if self.using_hf else "cuda"
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).to(device)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            ).to(device)
            pixel_values = None
            image_grid_thw = None

        # Setup optimizer
        if self.using_hf:
            params_to_tune = [
                p for n, p in self.hf_model.named_parameters() 
                if "lora_" in n or p.requires_grad
            ]
            model_for_training = self.hf_model
        else:
            params_to_tune = [
                p for n, p in self.llm.model_runner.model.named_parameters() 
                if "lora_" in n
            ]
            model_for_training = self.llm.model_runner.model
        
        if self.use_adam8bit:
            logging.debug(f"        [fine_tune] using Adam 8-bit optimizer for {adapter}")
            optimizer = bnb.optim.Adam8bit(params_to_tune, lr=lr)
            try:
                from bitsandbytes.optim import GlobalOptimManager
                gopm = GlobalOptimManager.get_instance()
                if hasattr(gopm, "register_model"):
                    gopm.register_model(model_for_training)
                if hasattr(gopm, "freeze_non_trainable_params"):
                    gopm.freeze_non_trainable_params(model_for_training)
            except Exception as exc:
                logging.debug(f"[warn] paging setup failed: {exc}")
        else:
            optimizer = torch.optim.Adam(params_to_tune, lr=lr)
        
        # Check if KTO training is requested
        is_kto_training = (trainer is not None and 
                          KTO_AVAILABLE and 
                          isinstance(trainer, KTOTrainer))
        
        # Prepare KTO-specific data if needed
        if is_kto_training:
            # Process examples for KTO using TRL-style tokenization
            kto_data = []
            for ex in examples:
                # Extract prompt and response
                prompt = ex.get("prompt", "")
                response = ex.get("response", "")
                
                # Get preference label (1 for desirable, 0 for undesirable)
                pref_label = ex.get("desirable", ex.get("preference", 1))
                
                # Tokenize using TRL approach
                tokenized = trainer.tokenize_conversation(self.tokenizer, prompt, response)
                tokenized["preference_label"] = pref_label
                kto_data.append(tokenized)
            
            # Create KL dataset if we have multiple examples
            if len(kto_data) >= 2:
                kto_data = trainer.create_kl_dataset(kto_data)
            
            # Create reference model before training if needed
            reference_model = None
            if not trainer.reference_free and not trainer._reference_model_created:
                logging.debug(f"[KTO] Creating reference model before training...")
                reference_model = trainer.create_reference_model(model_for_training)
            else:
                reference_model = trainer._reference_model
                
            logging.debug(f"[KTO] Using reference model: {reference_model is not None}, reference_free={trainer.reference_free}")
        
        # Training loop
        for step in range(steps):
            async with self.lock:
                if is_kto_training:
                    # Create batch from KTO data
                    batch = {}
                    
                    # Stack all tokenized data
                    for key in ['completion_input_ids', 'completion_attention_mask', 'completion_labels']:
                        values = [item[key] for item in kto_data]
                        # Pad sequences to same length
                        max_len = max(len(v) for v in values)
                        padded_values = []
                        for v in values:
                            padded = v + [self.tokenizer.pad_token_id if key.endswith('_ids') else 0] * (max_len - len(v))
                            if key == 'completion_labels':
                                # Use -100 for padding in labels
                                padded = v + [-100] * (max_len - len(v))
                            padded_values.append(padded)
                        batch[key] = torch.tensor(padded_values, device=device)
                    
                    # Add preference labels and labels for KTO trainer
                    batch['preference_labels'] = torch.tensor([item['preference_label'] for item in kto_data], device=device)
                    # KTO trainer expects 'label' field: True for desirable/chosen, False for undesirable/rejected
                    batch['label'] = [bool(item['preference_label']) for item in kto_data]
                    
                    # Add KL data if available
                    if 'KL_completion_input_ids' in kto_data[0]:
                        for key in ['KL_completion_input_ids', 'KL_completion_attention_mask', 'KL_completion_labels']:
                            values = [item[key] for item in kto_data]
                            max_len = max(len(v) for v in values)
                            padded_values = []
                            for v in values:
                                if key == 'KL_completion_labels':
                                    padded = v + [-100] * (max_len - len(v))
                                else:
                                    padded = v + [self.tokenizer.pad_token_id if key.endswith('_ids') else 0] * (max_len - len(v))
                                padded_values.append(padded)
                            batch[key] = torch.tensor(padded_values, device=device)
                    
                    metrics = trainer.train_step(
                        model_for_training,
                        reference_model,
                        batch,
                        optimizer
                    )
                    
                    loss_value = metrics['loss']
                    logging.debug(f"        [KTO] {adapter} step={step:3d}  loss={loss_value:.4f}")
                    if 'chosen_rewards' in metrics:
                        logging.debug(f"        [KTO] chosen_rewards={metrics['chosen_rewards']:.4f}")
                    if 'rejected_rewards' in metrics:
                        logging.debug(f"        [KTO] rejected_rewards={metrics['rejected_rewards']:.4f}")
                        
                elif self.using_hf:
                    # Standard HF training
                    self.hf_model.train()
                    optimizer.zero_grad()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    
                    # Pass vision inputs if available
                    forward_inputs = {
                        'input_ids': input_ids, 
                        'labels': labels
                    }
                    if pixel_values is not None:
                        forward_inputs['pixel_values'] = pixel_values
                    if image_grid_thw is not None:
                        forward_inputs['image_grid_thw'] = image_grid_thw
                    
                    # For MoE models, enable router logits during training for aux loss
                    if is_moe_model:
                        forward_inputs['output_router_logits'] = True
                    
                    outputs = self.hf_model(**forward_inputs)
                    loss = outputs.loss
                    
                    # Add auxiliary loss for MoE models
                    if is_moe_model and hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                        loss = loss + outputs.aux_loss
                    
                    loss.backward()
                    optimizer.step()
                else:
                    # Standard custom training
                    self.llm.train()
                    optimizer.zero_grad()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    loss = self.llm.forward_train(input_ids, labels)
                    loss.backward()
                    optimizer.step()

            if step == 0 or step == steps - 1 or step % 10 == 0:
                if not is_kto_training:  # KTO logging is handled above
                    logging.debug(f"        [fine_tune] {adapter} step={step:3d}  loss={loss.item():.4f}")

            if step == 0:
                step_t0 = time.perf_counter()
            elif step == 1:
                step_dt_ms = (time.perf_counter() - step_t0) * 1000
                logging.debug(f"[metric] fine_tune_step_ms≈{step_dt_ms:.2f}")
            
            await asyncio.sleep(0)  # Yield to other async tasks

        # Final evaluation
        async with self.lock:
            if self.using_hf:
                self.hf_model.eval()
                with torch.no_grad():
                    norms = {
                        n: p.norm().item() 
                        for n, p in self.hf_model.named_parameters() 
                        if "lora_" in n or p.requires_grad
                    }
                    avg_norm = sum(norms.values()) / len(norms) if norms else 0.0
                    logging.debug(f"        [fine_tune] {adapter} average LoRA param norm={avg_norm:.4f}")
            else:
                self.llm.eval()
                with torch.no_grad():
                    norms = {
                        n: p.norm().item() 
                        for n, p in self.llm.model_runner.model.named_parameters() 
                        if "lora_" in n
                    }
                    avg_norm = sum(norms.values()) / len(norms) if norms else 0.0
                    logging.debug(f"        [fine_tune] {adapter} average LoRA param norm={avg_norm:.4f}")

        total_ft_ms = (time.perf_counter() - t_ft_start) * 1000
        logging.debug(f"[metric] fine_tune_total_ms={total_ft_ms:.2f}")
        return optimizer.state_dict()

    # Synchronous wrapper methods for backward compatibility
    def generate(self, prompt: str, adapter: Optional[str] = None) -> str:
        """Generate text from a prompt (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            # If we're in an async context, raise an error with helpful message
            raise RuntimeError(
                "Cannot use sync generate() method from within async context. "
                "Use AsyncOnlineLLM.generate() instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                raise e
            # No event loop running, safe to use asyncio.run()
            
            # Create a temporary async method to call batch_generate
            async def _generate_async():
                results = await self.batch_generate([prompt], None, adapter, None)
                return results[0]
            
            return asyncio.run(_generate_async())

    def stream(self, prompt: str, adapter: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation (synchronous wrapper).
        
        Note: This is simulated streaming - generates complete response then yields word by word.
        For real streaming in async context, use AsyncOnlineLLM.stream() instead.
        """
        result = self.generate(prompt, adapter)
        
        # Yield word by word for more realistic streaming feel
        words = result.split()
        for i, word in enumerate(words):
            if i > 0:
                yield " "  # Add space before each word except the first
            yield word

    def learn(self, msgs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]], adapter: str = "default", trainer: Optional[Any] = None) -> None:
        """Learn from conversation messages (positive/desirable examples) (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            # If we're in an async context, raise an error with helpful message
            raise RuntimeError(
                "Cannot use sync learn() method from within async context. "
                "Use AsyncOnlineLLM.learn() instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                raise e
            # No event loop running, create temp async instance
            async_instance = AsyncOnlineLLM(
                model=self.model_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                offload_lora_to_cpu=self.offload_lora_to_cpu,
                use_adam8bit=self.use_adam8bit,
                engine=self.engine if hasattr(self, 'engine') else None
            )
            # Copy state
            async_instance.lora_states = self.lora_states.copy()
            async_instance.opt_states = self.opt_states.copy()
            
            async def _run_learn():
                await async_instance.learn(msgs, adapter, trainer=trainer)
                # Copy state back
                self.lora_states.update(async_instance.lora_states)
                self.opt_states.update(async_instance.opt_states)
            
            asyncio.run(_run_learn())
    
    def unlearn(self, msgs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]], adapter: str = "default", trainer: Optional[Any] = None) -> None:
        """Unlearn from conversation messages (negative/undesirable examples) (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            # If we're in an async context, raise an error with helpful message
            raise RuntimeError(
                "Cannot use sync unlearn() method from within async context. "
                "Use AsyncOnlineLLM.unlearn() instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                raise e
            # No event loop running, create temp async instance
            async_instance = AsyncOnlineLLM(
                model=self.model_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                offload_lora_to_cpu=self.offload_lora_to_cpu,
                use_adam8bit=self.use_adam8bit,
                engine=self.engine if hasattr(self, 'engine') else None
            )
            # Copy state
            async_instance.lora_states = self.lora_states.copy()
            async_instance.opt_states = self.opt_states.copy()
            
            async def _run_unlearn():
                await async_instance.unlearn(msgs, adapter, trainer=trainer)
                # Copy state back
                self.lora_states.update(async_instance.lora_states)
                self.opt_states.update(async_instance.opt_states)
            
            asyncio.run(_run_unlearn())

    def chat(self, msgs: List[Dict[str, Any]], adapter: Optional[str] = None) -> str:
        """Chat with the model using a list of messages."""
        # Convert messages to a single prompt (simplified)
        prompt = self._messages_to_prompt(msgs)
        return self.generate(prompt, adapter)

    def chat_stream(self, msgs: List[Dict[str, Any]], adapter: Optional[str] = None, sampling_params: Optional[SamplingParams] = None) -> Generator[str, None, None]:
        """Stream chat with the model using a list of messages."""
        # Check if this is a Qwen2.5-VL model with vision inputs
        is_qwen_vl = "VL" in self.model_path and "2.5" in self.model_path
        has_vision_content = any(
            isinstance(msg.get("content"), list) and 
            any(item.get("type") == "image" for item in msg.get("content", []) if isinstance(item, dict))
            for msg in msgs
        )
        
        if is_qwen_vl and self.using_hf and has_vision_content and hasattr(self.hf_model, 'processor'):
            # Handle vision inputs with proper processing
            try:
                from qwen_vl_utils import process_vision_info
                
                # Add anti-hallucination system message if not present
                has_system = any(msg.get('role') == 'system' for msg in msgs)
                formatted_msgs = []
                
                if not has_system:
                    formatted_msgs.append({
                        "role": "system", 
                        "content": "You are a helpful assistant. Answer questions directly and concisely."
                    })
                
                # Add all messages with proper content formatting
                for msg in msgs:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role in ['system', 'user', 'assistant']:
                        if isinstance(content, str):
                            formatted_content = [{"type": "text", "text": content}]
                        elif isinstance(content, list):
                            formatted_content = content
                        else:
                            formatted_content = [{"type": "text", "text": str(content)}]
                        
                        formatted_msgs.append({
                            "role": role,
                            "content": formatted_content
                        })
                
                # Apply chat template
                text = self.hf_model.processor.apply_chat_template(
                    formatted_msgs, tokenize=False, add_generation_prompt=True
                )
                
                # Extract vision inputs
                image_inputs, video_inputs = process_vision_info(formatted_msgs)
                
                # Process inputs
                inputs = self.hf_model.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
                
                # Load adapter if specified
                if adapter:
                    adapter_found = self.ensure_lora_adapter(adapter)
                    if adapter_found:
                        self.load_lora_state(self.lora_states[adapter])
                
                # Generate with streaming
                from transformers import TextIteratorStreamer
                import threading
                
                # Use sampling params if provided, otherwise use defaults
                if sampling_params is None:
                    sampling_params = self.default_sampling_params
                
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": sampling_params.max_tokens,
                    "do_sample": sampling_params.temperature > 0,
                    "temperature": sampling_params.temperature if sampling_params.temperature > 0 else None,
                    "streamer": streamer,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
                
                # Start generation in a separate thread
                thread = threading.Thread(target=self.hf_model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream tokens
                for new_text in streamer:
                    yield new_text
                
                thread.join()
                return
                
            except ImportError:
                logging.warning("qwen_vl_utils not available, falling back to text-only processing")
            except Exception as e:
                logging.warning(f"Vision processing failed: {e}, falling back to text-only")
        
        # Fallback to text-only processing
        prompt = self._messages_to_prompt(msgs)
        return self.stream(prompt, adapter)

    def _messages_to_prompt(self, msgs: List[Dict[str, Any]]) -> str:
        """Convert a list of messages to a single prompt string."""
        # Check if this is a Qwen2.5-VL model with processor
        is_qwen_vl = "VL" in self.model_path and "2.5" in self.model_path
        
        if is_qwen_vl and self.using_hf and hasattr(self.hf_model, 'processor'):
            # Use the proper Qwen2.5-VL processor's apply_chat_template
            try:
                # Add anti-hallucination system message if not present
                has_system = any(msg.get('role') == 'system' for msg in msgs)
                formatted_msgs = []
                
                if not has_system:
                    formatted_msgs.append({
                        "role": "system", 
                        "content": "You are a helpful assistant. Answer questions directly and concisely."
                    })
                
                # Add all messages, handling both text-only and mixed content
                for msg in msgs:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role in ['system', 'user', 'assistant']:
                        # Handle different content formats
                        if isinstance(content, str):
                            # Simple text content
                            formatted_content = [{"type": "text", "text": content}]
                        elif isinstance(content, list):
                            # Already in proper format (mixed content with images/text)
                            formatted_content = content
                        else:
                            # Fallback for other formats
                            formatted_content = [{"type": "text", "text": str(content)}]
                        
                        formatted_msgs.append({
                            "role": role,
                            "content": formatted_content
                        })
                
                # Use the processor's chat template
                prompt = self.hf_model.processor.apply_chat_template(
                    formatted_msgs, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
                
            except Exception as e:
                # Fallback to manual format if processor fails
                logging.warning(f"Failed to use Qwen2.5-VL processor chat template: {e}")
        
        # Fallback: Use original format for other models or if processor fails
        prompt_parts = []
        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract text from mixed content for fallback
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"

    def list_adapters(self) -> List[str]:
        """List all available LoRA adapters (in memory and on disk)."""
        memory_adapters = set(self.lora_states.keys())
        disk_adapters = set(self.list_disk_adapters())
        all_adapters = memory_adapters | disk_adapters
        return sorted(list(all_adapters))

    def get_adapter_info(self, adapter: str) -> Dict[str, Any]:
        """Get detailed information about a LoRA adapter.
        
        Args:
            adapter: Name of the adapter
            
        Returns:
            Dictionary with adapter information including location and metadata
        """
        info = {
            "name": adapter,
            "in_memory": adapter in self.lora_states,
            "on_disk": False,
            "metadata": {}
        }
        
        # Check disk information
        save_path = self.get_lora_save_path()
        if save_path:
            adapter_file = save_path / f"{adapter}.pt"
            if adapter_file.exists():
                info["on_disk"] = True
                info["disk_path"] = str(adapter_file)
                
                try:
                    saved_data = torch.load(adapter_file, map_location='cpu')
                    metadata = saved_data.get('metadata', {})
                    info["metadata"] = metadata
                except Exception as e:
                    info["disk_error"] = str(e)
        
        # Add memory information if available
        if adapter in self.lora_states:
            lora_state = self.lora_states[adapter]
            info["param_count"] = sum(p.numel() for p in lora_state.values())
            if lora_state:
                total_norm = sum(p.norm().item() for p in lora_state.values())
                info["avg_param_norm"] = total_norm / len(lora_state)
        
        return info

    def set_thinking_mode(self, enabled: bool) -> None:
        """Enable or disable thinking mode for supported models (Qwen3)."""
        self.thinking_mode = enabled
        if self.using_hf and hasattr(self.hf_model, 'enable_thinking'):
            self.hf_model.enable_thinking = enabled

    def get_thinking_mode(self) -> bool:
        """Get current thinking mode status."""
        if self.using_hf and hasattr(self.hf_model, 'enable_thinking'):
            return self.hf_model.enable_thinking
        return self.thinking_mode

    def delete_adapter(self, adapter: str, delete_from_disk: bool = False) -> bool:
        """Delete a LoRA adapter and its associated states.
        
        Args:
            adapter: Name of the adapter to delete
            delete_from_disk: Whether to also delete the adapter from disk
            
        Returns:
            True if adapter was found and deleted, False otherwise
        """
        found = False
        
        # Delete from memory
        if adapter in self.lora_states:
            del self.lora_states[adapter]
            found = True
        if adapter in self.opt_states:
            del self.opt_states[adapter]
        if adapter in self.training_tasks:
            # Cancel the task if it's still running
            task = self.training_tasks.pop(adapter)
            if not task.done():
                task.cancel()
        
        # Delete from disk if requested
        if delete_from_disk:
            save_path = self.get_lora_save_path()
            if save_path:
                adapter_file = save_path / f"{adapter}.pt"
                if adapter_file.exists():
                    try:
                        adapter_file.unlink()
                        logging.info(f"Deleted LoRA adapter {adapter} from disk")
                        found = True
                    except Exception as e:
                        logging.error(f"Failed to delete LoRA adapter {adapter} from disk: {e}")
        
        if found:
            logging.debug(f"Deleted LoRA adapter {adapter}")
        else:
            logging.warning(f"LoRA adapter {adapter} not found")
        
        return found

    async def wait_for_training(self, adapter: Optional[str] = None) -> None:
        """Wait for training tasks to complete."""
        if adapter:
            if adapter in self.training_tasks:
                await self.training_tasks[adapter]
        else:
            # Wait for all training tasks
            if self.training_tasks:
                await asyncio.gather(*self.training_tasks.values())

    def stats(self, show_detailed: bool = True) -> None:
        """Display comprehensive statistics about the OnlineLLM using rich formatting.
        
        Args:
            show_detailed: Whether to show detailed information about each adapter
        """
        if not RICH_AVAILABLE:
            # Fallback to simple text output
            print("=== OnlineLLM Statistics ===")
            print(f"Model: {self.model_path}")
            print(f"Engine: {self.engine}")
            print(f"Adapters: {len(self.lora_states)}")
            print(f"Active Training Tasks: {len(self.training_tasks)}")
            
            if self.lora_states:
                print("\nAdapters:")
                for adapter in self.lora_states.keys():
                    print(f"  - {adapter}")
            return
        
        console = Console()
        
        # Model Information Panel
        model_info = Table.grid(padding=1)
        model_info.add_column(style="bold blue")
        model_info.add_column()
        
        model_info.add_row("Model:", self.model_path)
        model_info.add_row("Engine:", self.engine)
        model_info.add_row("Using HF:", str(self.using_hf))
        model_info.add_row("Thinking Mode:", str(getattr(self, 'thinking_mode', False)))
        model_info.add_row("LoRA Rank:", str(self.lora_rank))
        model_info.add_row("LoRA Alpha:", str(self.lora_alpha))
        model_info.add_row("Use Adam8bit:", str(self.use_adam8bit))
        
        model_panel = Panel(model_info, title="[bold green]Model Configuration[/bold green]", border_style="green")
        
        # Memory Information Panel
        memory_info = self.batch_manager.get_memory_info()
        memory_table = Table.grid(padding=1)
        memory_table.add_column(style="bold yellow")
        memory_table.add_column()
        
        if memory_info.get("gpu_available", False):
            memory_table.add_row("GPU Available:", "✅ Yes")
            memory_table.add_row("Total Memory:", f"{memory_info['total_memory_gb']:.2f} GB")
            memory_table.add_row("Allocated Memory:", f"{memory_info['allocated_memory_gb']:.2f} GB")
            memory_table.add_row("Cached Memory:", f"{memory_info['cached_memory_gb']:.2f} GB")
            memory_table.add_row("Usage:", f"{memory_info['usage_percentage']:.1f}%")
            memory_table.add_row("Available:", f"{memory_info['available_percentage']:.1f}%")
        else:
            memory_table.add_row("GPU Available:", "❌ No")
        
        memory_panel = Panel(memory_table, title="[bold yellow]Memory Status[/bold yellow]", border_style="yellow")
        
        # Batch Configuration Panel
        batch_table = Table.grid(padding=1)
        batch_table.add_column(style="bold cyan")
        batch_table.add_column()
        
        batch_table.add_row("Max Batch Size:", str(self.batch_config.max_batch_size))
        batch_table.add_row("Auto Batch Size:", str(self.batch_config.auto_batch_size))
        batch_table.add_row("Memory Threshold:", f"{self.batch_config.memory_threshold:.1%}")
        batch_table.add_row("Min Batch Size:", str(self.batch_config.min_batch_size))
        batch_table.add_row("Current Batch Size:", str(self.batch_manager._current_batch_size))
        
        batch_panel = Panel(batch_table, title="[bold cyan]Batch Configuration[/bold cyan]", border_style="cyan")
        
        # Adapter Summary Panel
        adapter_summary = Table.grid(padding=1)
        adapter_summary.add_column(style="bold magenta")
        adapter_summary.add_column()
        
        # Calculate adapter statistics
        total_adapters = len(self.list_adapters())
        memory_adapters = len(self.lora_states)
        disk_adapters = len(self.list_disk_adapters())
        
        adapter_summary.add_row("Total Adapters:", str(total_adapters))
        adapter_summary.add_row("In Memory:", str(memory_adapters))
        adapter_summary.add_row("On Disk:", str(disk_adapters))
        adapter_summary.add_row("Active Training:", str(len(self.training_tasks)))
        adapter_summary.add_row("Stored Optimizers:", str(len(self.opt_states)))
        adapter_summary.add_row("Save Path:", str(self.get_lora_save_path()) if self.get_lora_save_path() else "Not set")
        
        summary_panel = Panel(adapter_summary, title="[bold magenta]Adapter Summary[/bold magenta]", border_style="magenta")
        
        # Display top-level panels
        console.print()
        console.print(Panel(Text("OnlineLLM Statistics", style="bold white", justify="center"), 
                           style="bold white on blue"))
        console.print()
        
        # Display panels in columns
        console.print(Columns([model_panel, memory_panel], equal=True))
        console.print(Columns([batch_panel, summary_panel], equal=True))
        
        # Detailed Adapter Information
        if self.lora_states and show_detailed:
            console.print()
            adapter_table = Table(
                "Adapter Name", 
                "Location",
                "LoRA Params", 
                "Has Optimizer", 
                "Training Status",
                "Param Norm",
                title="[bold magenta]Adapter Details[/bold magenta]",
                border_style="magenta"
            )
            
            all_adapters = self.list_adapters()
            for adapter_name in all_adapters:
                # Get adapter info
                adapter_info = self.get_adapter_info(adapter_name)
                
                # Determine location
                location_parts = []
                if adapter_info["in_memory"]:
                    location_parts.append("💾 Memory")
                if adapter_info["on_disk"]:
                    location_parts.append("💿 Disk")
                location = " + ".join(location_parts) if location_parts else "❓ Unknown"
                
                # Count LoRA parameters
                if adapter_name in self.lora_states:
                    lora_state = self.lora_states[adapter_name]
                    param_count = sum(p.numel() for p in lora_state.values())
                else:
                    param_count = adapter_info.get("param_count", 0)
                
                # Check optimizer state
                has_optimizer = "✅" if adapter_name in self.opt_states else "❌"
                
                # Check training status
                if adapter_name in self.training_tasks:
                    task = self.training_tasks[adapter_name]
                    if task.done():
                        training_status = "✅ Complete"
                    else:
                        training_status = "🔄 Training"
                else:
                    training_status = "⏸️ Idle"
                
                # Calculate parameter norm
                if adapter_name in self.lora_states:
                    lora_state = self.lora_states[adapter_name]
                    if lora_state:
                        total_norm = sum(p.norm().item() for p in lora_state.values())
                        avg_norm = total_norm / len(lora_state)
                        norm_str = f"{avg_norm:.4f}"
                    else:
                        norm_str = "N/A"
                else:
                    norm_str = adapter_info.get("avg_param_norm", "N/A")
                    if isinstance(norm_str, float):
                        norm_str = f"{norm_str:.4f}"
                
                adapter_table.add_row(
                    adapter_name,
                    location,
                    f"{param_count:,}" if param_count > 0 else "N/A",
                    has_optimizer,
                    training_status,
                    str(norm_str)
                )
            
            console.print(adapter_table)
        
        elif self.lora_states:
            # Simple adapter list if not showing details
            console.print()
            adapter_text = Text()
            adapter_text.append("Adapters: ", style="bold")
            adapter_text.append(", ".join(sorted(self.lora_states.keys())))
            console.print(Panel(adapter_text, title="[bold magenta]Available Adapters[/bold magenta]", border_style="magenta"))
        
        console.print()


class AsyncOnlineLLM(OnlineLLM):
    """
    Async version of OnlineLLM with enhanced batch learning capabilities.
    
    This class extends OnlineLLM with async methods and proper single-adapter batch handling.
    Key features include:
    - Example accumulation per adapter until train_every threshold
    - Auto-triggered single-adapter training for perfect isolation
    - No cross-adapter contamination during training
    - Memory-aware batch processing
    - Advanced batch inference capabilities
    
    Training Flow:
    1. Examples accumulate for each adapter separately
    2. When an adapter reaches train_every examples, it trains automatically
    3. Only one adapter trains at a time (perfect isolation)
    4. Each adapter maintains its own learned behavior
    """
    
    def __init__(self, *args: Any, train_every: int = 10, default_trainer: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Batch accumulation for KTO training
        self.pending_learn_examples: Dict[str, List[Dict]] = {}
        self.pending_unlearn_examples: Dict[str, List[Dict]] = {}
        self.min_batch_size = 2  # KTO requirement (reduced from 4)
        self.train_every = train_every  # Train every N examples
        self.example_counts: Dict[str, int] = {}  # Track total examples per adapter
        self.default_trainer = default_trainer  # Default trainer to use if none specified
    

    
    def _should_train(self, adapter: str) -> bool:
        """Check if we should trigger training based on train_every threshold."""
        total_examples = self.example_counts.get(adapter, 0)
        return total_examples > 0 and total_examples % self.train_every == 0
    
    def _extract_prompt_response_from_conversation(self, conversation: List[Dict[str, Any]]) -> tuple[str, str]:
        """Extract prompt and response from a conversation in OpenAI format."""
        if len(conversation) < 2:
            raise ValueError("Conversation must have at least 2 messages (user and assistant)")
        
        # Find the last user message and subsequent assistant message
        prompt_content = ""
        response_content = ""
        
        # Simple approach: concatenate all user messages as prompt, last assistant message as response
        user_messages = []
        assistant_messages = []
        
        for msg in conversation:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                assistant_messages.append(msg.get("content", ""))
        
        if not user_messages:
            raise ValueError("Conversation must contain at least one user message")
        if not assistant_messages:
            raise ValueError("Conversation must contain at least one assistant message")
        
        # Use the full conversation as context, but for training we'll focus on the last exchange
        prompt_content = " ".join(user_messages)
        response_content = assistant_messages[-1]  # Use the last assistant response
        
        return prompt_content, response_content
    
    async def learn(
        self,
        msgs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],  # Single conversation or list of conversations
        adapter: str = "default",
        steps: int = 3,
        lr: float = 1e-4,
        trainer: Optional[Any] = None,
    ) -> None:
        """Learn from conversation messages (positive/desirable examples).
        
        Args:
            msgs: Either a single conversation (list of messages) or multiple conversations (list of list of messages).
                  Each message should be in OpenAI format: {"role": "user/assistant", "content": "..."}
            adapter: LoRA adapter name
            steps: Training steps
            lr: Learning rate  
            trainer: Optional trainer (e.g., KTOTrainer). If None, uses default_trainer.
        
        Examples:
            # Single conversation
            await llm.learn([
                {"role": "user", "content": "What is water?"},
                {"role": "assistant", "content": "Water is H2O."}
            ])
            
            # Multiple conversations
            await llm.learn([
                [{"role": "user", "content": "What is water?"}, {"role": "assistant", "content": "Water is H2O."}],
                [{"role": "user", "content": "What is fire?"}, {"role": "assistant", "content": "Fire is combustion."}]
            ])
        """
        # Use default trainer if none provided
        if trainer is None:
            trainer = self.default_trainer
            
        # Normalize input to list of conversations
        if isinstance(msgs[0], dict):
            # Single conversation (list of messages)
            conversations = [msgs]
        else:
            # Multiple conversations (list of list of messages)
            conversations = msgs
        
        # Process each conversation
        processed_examples = []
        for conversation in conversations:
            # Extract prompt and response from conversation
            prompt_content, response_content = self._extract_prompt_response_from_conversation(conversation)
            
            example = {
                "prompt": prompt_content,
                "response": response_content,
                "conversation": conversation,
                "desirable": 1  # Learn method always treats examples as desirable
            }
            processed_examples.append(example)
        
        # For KTO training, accumulate examples and train based on train_every
        if trainer is not None and KTO_AVAILABLE and isinstance(trainer, KTOTrainer):
            # Add to pending learn examples (always desirable)
            if adapter not in self.pending_learn_examples:
                self.pending_learn_examples[adapter] = []
            self.pending_learn_examples[adapter].extend(processed_examples)
            
            # Update example count
            self.example_counts[adapter] = self.example_counts.get(adapter, 0) + len(processed_examples)
            
            # Check if we should train now
            if self._should_train(adapter):
                await self._train_accumulated_examples(adapter, steps, lr, trainer)
            return
        
        # For non-KTO training, train immediately
        await self._train_batch(processed_examples, adapter, steps, lr, trainer)
    
    async def unlearn(
        self,
        msgs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        adapter: str = "default",
        steps: int = 3,
        lr: float = 1e-4,
        trainer: Optional[Any] = None,
    ) -> None:
        """Unlearn from conversation messages (negative/undesirable examples).
        
        Args:
            msgs: Either a single conversation (list of messages) or multiple conversations (list of list of messages).
                  Each message should be in OpenAI format: {"role": "user/assistant", "content": "..."}
            adapter: LoRA adapter name
            steps: Training steps
            lr: Learning rate  
            trainer: Optional trainer (e.g., KTOTrainer). If None, uses default_trainer.
        
        Examples:
            # Single conversation
            await llm.unlearn([
                {"role": "user", "content": "What is water?"},
                {"role": "assistant", "content": "Water is mud."}  # Wrong answer to discourage
            ])
            
            # Multiple conversations
            await llm.unlearn([
                [{"role": "user", "content": "What is water?"}, {"role": "assistant", "content": "Water is mud."}],
                [{"role": "user", "content": "What is fire?"}, {"role": "assistant", "content": "Fire is ice."}]
            ])
        """
        # Use default trainer if none provided
        if trainer is None:
            trainer = self.default_trainer
            
        # Normalize input to list of conversations
        if isinstance(msgs[0], dict):
            # Single conversation (list of messages)
            conversations = [msgs]
        else:
            # Multiple conversations (list of list of messages)
            conversations = msgs
        
        # Process each conversation
        processed_examples = []
        for conversation in conversations:
            # Extract prompt and response from conversation
            prompt_content, response_content = self._extract_prompt_response_from_conversation(conversation)
            
            example = {
                "prompt": prompt_content,
                "response": response_content,
                "conversation": conversation,
                "desirable": 0  # Unlearn method always treats examples as undesirable
            }
            processed_examples.append(example)
        
        # For KTO training, accumulate examples and train based on train_every
        if trainer is not None and KTO_AVAILABLE and isinstance(trainer, KTOTrainer):
            # Add to pending unlearn examples (always undesirable)
            if adapter not in self.pending_unlearn_examples:
                self.pending_unlearn_examples[adapter] = []
            self.pending_unlearn_examples[adapter].extend(processed_examples)
            
            # Update example count
            self.example_counts[adapter] = self.example_counts.get(adapter, 0) + len(processed_examples)
            
            # Check if we should train now
            if self._should_train(adapter):
                await self._train_accumulated_examples(adapter, steps, lr, trainer)
            return
        
        # For non-KTO training, train immediately
        await self._train_batch(processed_examples, adapter, steps, lr, trainer)
    
    async def _train_accumulated_examples(
        self, 
        adapter: str, 
        steps: int, 
        lr: float, 
        trainer: Optional[Any]
    ) -> None:
        """Train on all accumulated examples for the given adapter.
        
        This method implements single-adapter batch training with proper isolation.
        Only the specified adapter is trained, ensuring no cross-contamination.
        """
        pending_learn = self.pending_learn_examples.get(adapter, [])
        pending_unlearn = self.pending_unlearn_examples.get(adapter, [])
        
        if pending_learn or pending_unlearn:
            all_examples = pending_learn + pending_unlearn
            
            # Ensure minimum batch size for KTO
            if trainer is not None and KTO_AVAILABLE and isinstance(trainer, KTOTrainer):
                if len(all_examples) < self.min_batch_size:
                    logging.debug(f"Accumulated {len(all_examples)} examples for {adapter}, "
                                f"waiting for minimum batch size of {self.min_batch_size}")
                    return
            
            logging.info(f"Auto-training {adapter} with {len(all_examples)} accumulated examples "
                        f"({len(pending_learn)} learn, {len(pending_unlearn)} unlearn)")
            
            # Train ONLY this adapter (single-adapter batch training)
            single_adapter_examples = {adapter: all_examples}
            await self.batch_fine_tune(single_adapter_examples, steps=steps, lr=lr, trainer=trainer)
            
            # Clear accumulated examples after training
            self.pending_learn_examples[adapter] = []
            self.pending_unlearn_examples[adapter] = []

    async def flush_pending_examples(self, adapter: str, trainer: Optional[Any] = None) -> None:
        """Force training on any pending examples for the given adapter."""
        pending_learn = self.pending_learn_examples.get(adapter, [])
        pending_unlearn = self.pending_unlearn_examples.get(adapter, [])
        
        if pending_learn or pending_unlearn:
            all_examples = pending_learn + pending_unlearn
            if len(all_examples) >= 2:  # Relaxed minimum for flushing
                await self._train_batch(all_examples, adapter, steps=3, lr=1e-4, trainer=trainer)
                self.pending_learn_examples[adapter] = []
                self.pending_unlearn_examples[adapter] = []
                logging.info(f"Flushed {len(all_examples)} pending examples for {adapter}")
    
    async def _train_batch(
        self,
        examples: List[Dict[str, Union[str, Any]]],
        adapter: str,
        steps: int,
        lr: float,
        trainer: Optional[Any],
    ) -> None:
        """Internal method to train on a batch of examples for a single adapter.
        
        This method ensures proper adapter isolation by training only the specified
        adapter without any concurrent training that could cause contamination.
        """
        # Wait for any existing training task for this adapter
        if adapter in self.training_tasks:
            await self.training_tasks.pop(adapter)

        # Use single-adapter batch training for perfect isolation
        single_adapter_examples = {adapter: examples}
        training_results = await self.batch_fine_tune(
            single_adapter_examples, 
            steps=steps, 
            lr=lr, 
            trainer=trainer
        )
        
        # Store the optimizer state if available
        if adapter in training_results:
            async with self.lock:
                self.opt_states[adapter] = training_results[adapter]
                
                # Auto-save to disk if LORA_SAVE_PATH is set
                if adapter in self.lora_states:
                    self.save_lora_to_disk(adapter, self.lora_states[adapter])
    

    
    async def generate(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        images: Optional[List[Any]] = None,  # Support multiple images
    ) -> str:
        """Generate text from a prompt."""
        logging.info(f"AsyncOnlineLLM.generate called with prompt: '{prompt[:100]}...', adapter: {adapter}")
        images_list = [images] if images is not None else None
        results = await self.batch_generate([prompt], sampling_params, adapter, images_list)
        logging.info(f"batch_generate returned: {len(results)} results, first result: '{results[0] if results else 'None'}' (length: {len(results[0]) if results else 0})")
        return results[0]

    async def stream(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation token by token.
        
        For custom models: Real token-by-token streaming
        For HF models: Simulated streaming (character-by-character) until HF streaming is implemented
        """
        logging.info(f"AsyncOnlineLLM.stream called with prompt: '{prompt[:100]}...', using_hf: {self.using_hf}")
        
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        # Load the appropriate LoRA adapter if specified
        if adapter:
            async with self.lock:
                adapter_found = self.ensure_lora_adapter(adapter)
                if adapter_found:
                    self.load_lora_state(self.lora_states[adapter])
                else:
                    logging.info(f"Using new LoRA adapter: {adapter}")

        if self.using_hf:
            # For HuggingFace models, we simulate streaming by generating the complete response
            # and then yielding it character by character
            # TODO: Implement true HF streaming when transformers supports it
            logging.info("Using HF model, calling generate...")
            result = await self.generate(prompt, adapter, sampling_params)
            logging.info(f"Generate returned: '{result}' (length: {len(result)})")
            
            if not result or result.strip() == "":
                logging.warning("Generate returned empty result!")
                return
            
            # Yield word by word for more realistic streaming feel
            words = result.split()
            logging.info(f"Splitting into {len(words)} words")
            for i, word in enumerate(words):
                if i > 0:
                    yield " "  # Add space before each word except the first
                yield word
                await asyncio.sleep(0.01)  # Small delay for streaming effect
        else:
            # For custom implementation, do real token-by-token streaming
            logging.info("Using custom model implementation")
            from dark.engine.sequence import Sequence
            
            seq = Sequence(self.tokenizer.encode(prompt), sampling_params)
            
            async with self.lock:
                self.llm.eval()
                self.llm.scheduler.add(seq)

            emitted = 0
            
            while not seq.is_finished:
                async with self.lock:
                    self.llm.eval()
                    with torch.cuda.stream(self.infer_stream):
                        self.llm.step()
                
                # Emit new tokens as they're generated
                while seq.num_completion_tokens > emitted:
                    tid = seq.completion_token_ids[emitted]
                    token_text = self.tokenizer.decode([tid], skip_special_tokens=True)
                    yield token_text
                    emitted += 1
                
                await asyncio.sleep(0)  # Yield to other async tasks

    async def chat(self, msgs: List[Dict[str, Any]], adapter: Optional[str] = None) -> str:
        """Chat with the model using a list of messages."""
        # Convert messages to a single prompt (simplified)
        prompt = self._messages_to_prompt(msgs)
        return await self.generate(prompt, adapter)

    async def chat_stream(self, msgs: List[Dict[str, Any]], adapter: Optional[str] = None, sampling_params: Optional[SamplingParams] = None) -> AsyncGenerator[str, None]:
        """Stream chat with the model using a list of messages (async version)."""
        logging.info(f"AsyncOnlineLLM.chat_stream called with {len(msgs)} messages")
        
        # Check if this is a Qwen2.5-VL model with vision inputs
        is_qwen_vl = "VL" in self.model_path and "2.5" in self.model_path
        has_vision_content = any(
            isinstance(msg.get("content"), list) and 
            any(item.get("type") == "image" for item in msg.get("content", []) if isinstance(item, dict))
            for msg in msgs
        )
        
        logging.info(f"is_qwen_vl: {is_qwen_vl}, has_vision_content: {has_vision_content}, using_hf: {self.using_hf}")
        
        if is_qwen_vl and self.using_hf and has_vision_content and hasattr(self.hf_model, 'processor'):
            # Handle vision inputs with proper processing
            try:
                from qwen_vl_utils import process_vision_info
                
                # Add anti-hallucination system message if not present
                has_system = any(msg.get('role') == 'system' for msg in msgs)
                formatted_msgs = []
                
                if not has_system:
                    formatted_msgs.append({
                        "role": "system", 
                        "content": "You are a helpful assistant. Answer questions directly and concisely."
                    })
                
                # Add all messages with proper content formatting
                for msg in msgs:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role in ['system', 'user', 'assistant']:
                        if isinstance(content, str):
                            formatted_content = [{"type": "text", "text": content}]
                        elif isinstance(content, list):
                            formatted_content = content
                        else:
                            formatted_content = [{"type": "text", "text": str(content)}]
                        
                        formatted_msgs.append({
                            "role": role,
                            "content": formatted_content
                        })
                
                # Apply chat template
                text = self.hf_model.processor.apply_chat_template(
                    formatted_msgs, tokenize=False, add_generation_prompt=True
                )
                
                # Extract vision inputs
                image_inputs, video_inputs = process_vision_info(formatted_msgs)
                
                # Process inputs
                inputs = self.hf_model.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
                
                # Load adapter if specified
                if adapter:
                    async with self.lock:
                        adapter_found = self.ensure_lora_adapter(adapter)
                        if adapter_found:
                            self.load_lora_state(self.lora_states[adapter])
                
                # Generate with streaming using AsyncOnlineLLM's generate method
                result = await self.generate("", adapter, sampling_params, [])  # Empty prompt since we use inputs directly
                
                # For now, simulate streaming with the complete result
                # TODO: Implement true async streaming for vision models
                words = result.split()
                for i, word in enumerate(words):
                    if i > 0:
                        yield " "
                    yield word
                    await asyncio.sleep(0.01)
                return
                
            except ImportError:
                logging.warning("qwen_vl_utils not available, falling back to text-only processing")
            except Exception as e:
                logging.warning(f"Vision processing failed: {e}, falling back to text-only")
        
        # Fallback to text-only processing using async stream
        logging.info("Using text-only fallback processing")
        prompt = self._messages_to_prompt(msgs)
        logging.info(f"Generated prompt: {prompt[:200]}...")
        
        chunk_count = 0
        async for chunk in self.stream(prompt, adapter, sampling_params):
            chunk_count += 1
            logging.debug(f"Yielding chunk {chunk_count}: '{chunk}'")
            yield chunk
        
        logging.info(f"AsyncOnlineLLM.chat_stream completed, yielded {chunk_count} chunks")
    
    async def batch_learn_from_conversations(
        self,
        conversations: List[List[Dict[str, Any]]],
        adapter: str = "default",
        batch_size: Optional[int] = None,
        steps: int = 3,
        lr: float = 1e-4,
        trainer: Optional[Any] = None,
    ) -> None:
        """Learn from multiple conversations in batches for improved efficiency.
        
        Args:
            conversations: List of conversations (each conversation is a list of messages)
            adapter: LoRA adapter name
            batch_size: Batch size for processing (uses auto-calculation if None)
            steps: Training steps
            lr: Learning rate
            trainer: Optional trainer
        """
        if batch_size is None:
            batch_size = self.batch_manager.get_optimal_batch_size(len(conversations))
        
        # Process conversations in batches
        conversation_batches = self.batch_manager.create_batches(conversations, batch_size)
        
        for batch in conversation_batches:
            await self.learn(batch, adapter, steps, lr, trainer)
            await asyncio.sleep(0)  # Yield between batches
    
    async def batch_compare_adapters(
        self,
        prompts: List[str],
        adapters: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple adapters on the same set of prompts.
        
        Args:
            prompts: List of prompts to test
            adapters: List of adapter names to compare
            sampling_params: Generation parameters
            
        Returns:
            Dictionary with comparison results and statistics
        """
        # Generate responses for all adapters in parallel
        adapter_responses = await self.parallel_adapter_generate(
            prompts, adapters, sampling_params
        )
        
        # Calculate basic statistics
        comparison_results = {
            "prompts": prompts,
            "adapters": adapters,
            "responses": adapter_responses,
            "statistics": {}
        }
        
        # Add basic response statistics
        for adapter in adapters:
            responses = adapter_responses[adapter]
            avg_length = sum(len(response.split()) for response in responses) / len(responses)
            comparison_results["statistics"][adapter] = {
                "avg_response_length": avg_length,
                "total_responses": len(responses),
                "error_count": sum(1 for r in responses if r == "ERROR")
            }
        
        return comparison_results
    
    async def auto_batch_train_pipeline(
        self,
        training_data: Dict[str, List[Dict[str, Any]]],  # {adapter: examples}
        validation_prompts: Optional[List[str]] = None,
        steps: int = 5,
        lr: float = 1e-4,
        trainer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Complete training pipeline with automatic batch optimization.
        
        Args:
            training_data: Dictionary mapping adapter names to training examples
            validation_prompts: Optional prompts for validation after training
            steps: Training steps per adapter
            lr: Learning rate
            trainer: Optional trainer
            
        Returns:
            Training results including validation metrics if prompts provided
        """
        pipeline_start = time.perf_counter()
        
        # Pre-training memory info
        pre_memory = self.batch_manager.get_memory_info()
        logging.info(f"Starting training pipeline with {len(training_data)} adapters")
        logging.info(f"Pre-training memory usage: {pre_memory.get('usage_percentage', 0):.1f}%")
        
        # Train all adapters in batches
        training_results = await self.batch_fine_tune(
            training_data, steps=steps, lr=lr, trainer=trainer
        )
        
        results = {
            "training_results": training_results,
            "training_time_ms": (time.perf_counter() - pipeline_start) * 1000,
            "pre_training_memory": pre_memory,
        }
        
        # Run validation if prompts provided
        if validation_prompts:
            validation_start = time.perf_counter()
            
            # Generate responses with all trained adapters
            adapter_names = list(training_data.keys())
            validation_responses = await self.parallel_adapter_generate(
                validation_prompts, adapter_names
            )
            
            results["validation_results"] = {
                "prompts": validation_prompts,
                "responses": validation_responses,
                "validation_time_ms": (time.perf_counter() - validation_start) * 1000
            }
        
        # Post-training memory info
        post_memory = self.batch_manager.get_memory_info()
        results["post_training_memory"] = post_memory
        
        logging.info(f"Training pipeline completed in {results['training_time_ms']:.2f}ms")
        logging.info(f"Post-training memory usage: {post_memory.get('usage_percentage', 0):.1f}%")
        
        return results

    def stats(self, show_detailed: bool = True) -> None:
        """Display comprehensive statistics about the AsyncOnlineLLM using rich formatting.
        
        Includes additional async-specific information like training buffers and example counts.
        
        Args:
            show_detailed: Whether to show detailed information about each adapter
        """
        if not RICH_AVAILABLE:
            # Enhanced fallback for AsyncOnlineLLM
            print("=== AsyncOnlineLLM Statistics ===")
            print(f"Model: {self.model_path}")
            print(f"Engine: {self.engine}")
            print(f"Adapters: {len(self.lora_states)}")
            print(f"Active Training Tasks: {len(self.training_tasks)}")
            print(f"Train Every: {self.train_every} examples")
            
            # Training buffer info
            total_pending_learn = sum(len(examples) for examples in self.pending_learn_examples.values())
            total_pending_unlearn = sum(len(examples) for examples in self.pending_unlearn_examples.values())
            print(f"Pending Learn Examples: {total_pending_learn}")
            print(f"Pending Unlearn Examples: {total_pending_unlearn}")
            
            if self.lora_states:
                print("\nAdapters:")
                for adapter in self.lora_states.keys():
                    count = self.example_counts.get(adapter, 0)
                    print(f"  - {adapter} ({count} examples trained)")
            return
        
        console = Console()
        
        # Call parent stats method to get base information
        super().stats(show_detailed=False)  # Don't show detailed adapters yet
        
        # Async-specific information
        async_info = Table.grid(padding=1)
        async_info.add_column(style="bold purple")
        async_info.add_column()
        
        async_info.add_row("Train Every:", f"{self.train_every} examples")
        async_info.add_row("Min Batch Size:", str(self.min_batch_size))
        async_info.add_row("Default Trainer:", str(type(self.default_trainer).__name__ if self.default_trainer else "None"))
        
        # Training buffer statistics
        total_pending_learn = sum(len(examples) for examples in self.pending_learn_examples.values())
        total_pending_unlearn = sum(len(examples) for examples in self.pending_unlearn_examples.values())
        async_info.add_row("Pending Learn:", str(total_pending_learn))
        async_info.add_row("Pending Unlearn:", str(total_pending_unlearn))
        
        async_panel = Panel(async_info, title="[bold purple]Async Training Configuration[/bold purple]", border_style="purple")
        console.print(async_panel)
        
        # Training Buffer Details
        if (self.pending_learn_examples or self.pending_unlearn_examples) and show_detailed:
            console.print()
            buffer_table = Table(
                "Adapter",
                "Pending Learn",
                "Pending Unlearn", 
                "Total Pending",
                "Progress to Training",
                title="[bold orange]Training Buffers[/bold orange]",
                border_style="orange"
            )
            
            all_adapters = set(self.pending_learn_examples.keys()) | set(self.pending_unlearn_examples.keys())
            for adapter in sorted(all_adapters):
                learn_count = len(self.pending_learn_examples.get(adapter, []))
                unlearn_count = len(self.pending_unlearn_examples.get(adapter, []))
                total_pending = learn_count + unlearn_count
                
                # Calculate progress to next training
                total_examples = self.example_counts.get(adapter, 0)
                next_training_at = ((total_examples // self.train_every) + 1) * self.train_every
                examples_until_training = next_training_at - total_examples
                
                progress_text = f"{total_pending}/{examples_until_training}"
                if total_pending >= examples_until_training:
                    progress_text = "[bold green]Ready to train![/bold green]"
                
                buffer_table.add_row(
                    adapter,
                    str(learn_count),
                    str(unlearn_count),
                    str(total_pending),
                    progress_text
                )
            
            console.print(buffer_table)
        
        # Enhanced Adapter Details with training history
        if self.lora_states and show_detailed:
            console.print()
            adapter_table = Table(
                "Adapter Name",
                "Total Examples",
                "LoRA Params", 
                "Has Optimizer", 
                "Training Status",
                "Param Norm",
                "Pending",
                title="[bold magenta]Adapter Details with Training History[/bold magenta]",
                border_style="magenta"
            )
            
            for adapter_name in sorted(self.lora_states.keys()):
                # Count LoRA parameters
                lora_state = self.lora_states[adapter_name]
                param_count = sum(p.numel() for p in lora_state.values())
                
                # Get example count
                example_count = self.example_counts.get(adapter_name, 0)
                
                # Check optimizer state
                has_optimizer = "✅" if adapter_name in self.opt_states else "❌"
                
                # Check training status
                if adapter_name in self.training_tasks:
                    task = self.training_tasks[adapter_name]
                    if task.done():
                        training_status = "✅ Complete"
                    else:
                        training_status = "🔄 Training"
                else:
                    training_status = "⏸️ Idle"
                
                # Calculate parameter norm
                if lora_state:
                    total_norm = sum(p.norm().item() for p in lora_state.values())
                    avg_norm = total_norm / len(lora_state)
                    norm_str = f"{avg_norm:.4f}"
                else:
                    norm_str = "N/A"
                
                # Pending examples
                pending_learn = len(self.pending_learn_examples.get(adapter_name, []))
                pending_unlearn = len(self.pending_unlearn_examples.get(adapter_name, []))
                pending_total = pending_learn + pending_unlearn
                pending_str = f"{pending_total}" if pending_total > 0 else "-"
                
                adapter_table.add_row(
                    adapter_name,
                    str(example_count),
                    f"{param_count:,}",
                    has_optimizer,
                    training_status,
                    norm_str,
                    pending_str
                )
            
            console.print(adapter_table)
        
        # Training Recommendations
        if show_detailed:
            recommendations = []
            
            # Check for ready-to-train adapters
            ready_adapters = []
            for adapter in self.pending_learn_examples.keys() | self.pending_unlearn_examples.keys():
                total_pending = (len(self.pending_learn_examples.get(adapter, [])) + 
                               len(self.pending_unlearn_examples.get(adapter, [])))
                total_examples = self.example_counts.get(adapter, 0)
                if total_examples > 0 and total_examples % self.train_every == 0:
                    ready_adapters.append(adapter)
            
            if ready_adapters:
                recommendations.append(f"🚀 {len(ready_adapters)} adapter(s) ready for training: {', '.join(ready_adapters)}")
            
            # Memory recommendations
            memory_info = self.batch_manager.get_memory_info()
            if memory_info.get("gpu_available", False):
                if memory_info["usage_percentage"] > 90:
                    recommendations.append("⚠️ High GPU memory usage - consider reducing batch size")
                elif memory_info["usage_percentage"] < 30:
                    recommendations.append("💡 Low GPU memory usage - you could increase batch size")
            
            # Training frequency recommendations
            total_pending = sum(len(examples) for examples in self.pending_learn_examples.values())
            total_pending += sum(len(examples) for examples in self.pending_unlearn_examples.values())
            if total_pending > self.train_every * 2:
                recommendations.append(f"💡 Consider lowering train_every from {self.train_every} to process pending examples faster")
            
            if recommendations:
                console.print()
                recommendations_text = "\n".join(f"• {rec}" for rec in recommendations)
                console.print(Panel(recommendations_text, title="[bold blue]Recommendations[/bold blue]", border_style="blue"))
        
        console.print()
