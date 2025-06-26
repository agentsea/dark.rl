import asyncio
import logging
import time
import warnings
from typing import List, Dict, Any, Generator, Optional, Union, AsyncGenerator
import torch
import bitsandbytes as bnb

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
    # MoE models
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
    
    Args:
        model: The model name/path to load
        engine: Implementation to use - "hf" for HuggingFace, "dark" for custom
                Defaults to "hf" for VL models, "dark" for text-only models
        thinking_mode: Enable thinking mode for supported models (Qwen3). When enabled,
                      the model can use thinking tokens (<think></think>) to reason
                      step-by-step before providing answers. Default: True
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

    def load_lora_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load LoRA tensors into the model (in-place).

        If `state` is empty, we leave the existing randomly-initialized LoRA weights
        untouched so that training can make progress.
        """
        t0 = time.perf_counter()
        if not state:
            logging.debug(f"[metric] lora_load_ms=0.00 (no state)")
            return
        
        with torch.no_grad():
            if self.using_hf:
                for n, p in self.hf_model.named_parameters():
                    if "lora_" in n and n in state:
                        p.copy_(state[n])
            else:
                for n, p in self.llm.model_runner.model.named_parameters():
                    if "lora_" in n and n in state:
                        p.copy_(state[n])
        
        dt_ms = (time.perf_counter() - t0) * 1000
        logging.debug(f"[metric] lora_load_ms={dt_ms:.2f}")

    async def stream_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        adapter: Optional[str] = None,
        images: Optional[List[List[Any]]] = None,  # Support multiple images per prompt
    ) -> List[str]:
        """Run generation step-by-step on a batch of prompts, asynchronously."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        gen_start = time.perf_counter()
        
        # Load the appropriate LoRA adapter if specified
        if adapter and adapter in self.lora_states:
            async with self.lock:
                self.load_lora_state(self.lora_states[adapter])

        if self.using_hf:
            # Use HF implementation for generation with vision support
            outputs = []
            async with self.lock:
                if hasattr(self.hf_model, 'eval'):
                    self.hf_model.eval()
                    
                for i, prompt in enumerate(prompts):
                    # Get corresponding images if provided
                    prompt_images = images[i] if images and i < len(images) else None
                    
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
                    
                    outputs.append(output)
            
            total_gen_ms = (time.perf_counter() - gen_start) * 1000
            logging.debug(f"[metric] generate_total_ms={total_gen_ms:.2f}")
            return outputs
        else:
            # Use custom implementation
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

            total_gen_ms = (time.perf_counter() - gen_start) * 1000
            num_tokens = sum(s.num_completion_tokens for s in seqs)
            per_tok = total_gen_ms / max(1, num_tokens)
            logging.debug(
                f"[metric] generate_total_ms={total_gen_ms:.2f}  "
                f"generate_per_token_ms={per_tok:.2f}"
            )
            return outs

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
            # Extract preference labels from examples (1 for desirable, 0 for undesirable)
            preference_labels = []
            for ex in examples:
                # Default to desirable (1) if not specified
                pref_label = ex.get("desirable", ex.get("preference", 1))
                preference_labels.append(pref_label)
            
            preference_labels = torch.tensor(preference_labels, device=device)
            
            # Create reference model if needed and not reference-free
            reference_model = None
            if not trainer.reference_free:
                # For simplicity, we'll use the current model state as reference
                # In practice, you might want to load a separate reference model
                reference_model = model_for_training
                logging.debug(f"[KTO] Using current model as reference (reference_free={trainer.reference_free})")
        
        # Training loop
        for step in range(steps):
            async with self.lock:
                if is_kto_training:
                    # KTO training step
                    batch = {
                        'input_ids': input_ids,
                        'attention_mask': torch.ones_like(input_ids),
                        'labels': labels,
                        'preference_labels': preference_labels
                    }
                    
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
                "Use generate_async() instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                raise e
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.generate_async(prompt, adapter))

    def stream(self, prompt: str, adapter: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation (synchronous wrapper)."""
        result = self.generate(prompt, adapter)
        for char in result:
            yield char

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



    def chat_stream(self, msgs: List[Dict[str, Any]], adapter: Optional[str] = None) -> Generator[str, None, None]:
        """Stream chat with the model using a list of messages."""
        prompt = self._messages_to_prompt(msgs)
        return self.stream(prompt, adapter)



    def _messages_to_prompt(self, msgs: List[Dict[str, Any]]) -> str:
        """Convert a list of messages to a single prompt string."""
        prompt_parts = []
        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"

    def list_adapters(self) -> List[str]:
        """List all available LoRA adapters."""
        return list(self.lora_states.keys())

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

    def delete_adapter(self, adapter: str) -> None:
        """Delete a LoRA adapter and its associated states."""
        if adapter in self.lora_states:
            del self.lora_states[adapter]
        if adapter in self.opt_states:
            del self.opt_states[adapter]
        if adapter in self.training_tasks:
            # Cancel the task if it's still running
            task = self.training_tasks.pop(adapter)
            if not task.done():
                task.cancel()

    async def wait_for_training(self, adapter: Optional[str] = None) -> None:
        """Wait for training tasks to complete."""
        if adapter:
            if adapter in self.training_tasks:
                await self.training_tasks[adapter]
        else:
            # Wait for all training tasks
            if self.training_tasks:
                await asyncio.gather(*self.training_tasks.values())


class AsyncOnlineLLM(OnlineLLM):
    """
    Async version of OnlineLLM with batch learning capabilities for KTO training.
    
    This class extends OnlineLLM with async methods and proper batch handling
    for KTO training which requires batch sizes >= 4.
    """
    
    def __init__(self, *args: Any, train_every: int = 10, default_trainer: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Batch accumulation for KTO training
        self.pending_learn_examples: Dict[str, List[Dict]] = {}
        self.pending_unlearn_examples: Dict[str, List[Dict]] = {}
        self.min_batch_size = 4  # KTO requirement
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
        msgs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],  # Single conversation or list of conversations
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
        """Train on all accumulated examples for the given adapter."""
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
            
            logging.info(f"Training on {len(all_examples)} accumulated examples for {adapter} "
                        f"({len(pending_learn)} learn, {len(pending_unlearn)} unlearn)")
            
            await self._train_batch(all_examples, adapter, steps, lr, trainer)
            
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
        """Internal method to train on a batch of examples."""
        # Wait for any existing training task for this adapter
        if adapter in self.training_tasks:
            await self.training_tasks.pop(adapter)

        # Load existing state for this adapter
        async with self.lock:
            if adapter in self.lora_states:
                self.load_lora_state(self.lora_states[adapter])

        # Run fine-tuning
        is_moe_model = "MoE" in self.model_path
        opt_state = await self.fine_tune(examples, adapter, steps, lr, is_moe_model, trainer)

        # Save updated states
        async with self.lock:
            self.lora_states[adapter] = self.get_lora_state()
            self.opt_states[adapter] = opt_state
    

    
    async def generate(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        images: Optional[List[Any]] = None,  # Support multiple images
    ) -> str:
        """Generate text from a prompt."""
        images_list = [images] if images is not None else None
        results = await self.stream_generate([prompt], sampling_params, adapter, images_list)
        return results[0]

    async def stream(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        # Load the appropriate LoRA adapter if specified
        if adapter and adapter in self.lora_states:
            async with self.lock:
                self.load_lora_state(self.lora_states[adapter])

        if self.using_hf:
            # For HuggingFace, we don't have true streaming yet, so simulate it
            # by generating and then streaming character by character
            result = await self.generate(prompt, adapter, sampling_params)
            for char in result:
                yield char
                await asyncio.sleep(0.01)  # Small delay for streaming effect
        else:
            # For custom implementation, do real token-by-token streaming
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
