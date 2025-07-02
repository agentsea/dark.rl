import asyncio
import time
from typing import List, Dict, Any, Generator, Optional, Union
import torch
import bitsandbytes as bnb

from dark import LLM, SamplingParams
from dark.engine.sequence import Sequence


SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-0.6B-Instruct",
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen3-4B-Instruct",
    "Qwen/Qwen3-8B-Instruct",
    "Qwen/Qwen3-14B-Instruct",
    "Qwen/Qwen3-32B-Instruct",
    # Also support base models (without -Instruct)
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


class OnlineLLM:
    """An online LLM that can be used to chat, learn, and fine-tune with LoRA adapters.
    
    This class integrates the functionality from the training test script, including:
    - LoRA fine-tuning with async capabilities
    - Streaming generation
    - Multiple LoRA adapter management
    - 8-bit Adam optimizer support
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

        # Initialize the underlying LLM with LoRA support
        self.llm = LLM(
            model,
            enforce_eager=enforce_eager,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        self.tokenizer = self.llm.tokenizer

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
        
        return {
            n: p.detach().clone().to("cpu" if to_cpu else p.device) 
            for n, p in self.llm.model_runner.model.named_parameters() 
            if "lora_" in n
        }

    def load_lora_state(self, state: Dict[str, torch.Tensor]):
        """Load LoRA tensors into the model (in-place).

        If `state` is empty, we leave the existing randomly-initialized LoRA weights
        untouched so that training can make progress.
        """
        t0 = time.perf_counter()
        if not state:
            print(f"[metric] lora_load_ms=0.00 (no state)")
            return
        
        with torch.no_grad():
            for n, p in self.llm.model_runner.model.named_parameters():
                if "lora_" in n and n in state:
                    p.copy_(state[n])
        
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[metric] lora_load_ms={dt_ms:.2f}")

    async def batch_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        lora_adapter: Optional[str] = None,
    ) -> List[str]:
        """Run generation step-by-step on a batch of prompts, asynchronously."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        gen_start = time.perf_counter()
        
        # Load the appropriate LoRA adapter if specified
        if lora_adapter and lora_adapter in self.lora_states:
            async with self.lock:
                self.load_lora_state(self.lora_states[lora_adapter])

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
            print(f"[metric] llm_step_ms={step_dt_ms:.2f}")

            for i, seq in enumerate(seqs):
                while seq.num_completion_tokens > emitted[i]:
                    tid = seq.completion_token_ids[emitted[i]]
                    outs[i] += self.tokenizer.decode([tid], skip_special_tokens=True)
                    emitted[i] += 1
            
            await asyncio.sleep(0)  # Yield to other async tasks

        total_gen_ms = (time.perf_counter() - gen_start) * 1000
        num_tokens = sum(s.num_completion_tokens for s in seqs)
        per_tok = total_gen_ms / max(1, num_tokens)
        print(
            f"[metric] generate_total_ms={total_gen_ms:.2f}  "
            f"generate_per_token_ms={per_tok:.2f}"
        )
        return outs

    # Backward compatibility alias (deprecated)
    async def stream_generate(self, *args, **kwargs) -> List[str]:
        """Deprecated: Use batch_generate() instead.
        
        This method has been renamed to batch_generate() to better reflect that it
        generates complete responses rather than streaming them.
        """
        import warnings
        warnings.warn(
            "stream_generate() is deprecated and will be removed in a future version. "
            "Use batch_generate() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return await self.batch_generate(*args, **kwargs)

    async def fine_tune(
        self,
        examples: List[Dict[str, str]],
        lora_adapter: str,
        steps: int = 5,
        lr: float = 1e-4,
    ) -> Dict:
        """Fine-tune the model on a batch of prompt/response pairs using LoRA."""
        t_ft_start = time.perf_counter()

        # Data preparation
        prompts = [ex["prompt"] for ex in examples]
        responses = [ex["response"] for ex in examples]
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

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).cuda()
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        ).cuda()

        # Setup optimizer
        params_to_tune = [
            p for n, p in self.llm.model_runner.model.named_parameters() 
            if "lora_" in n
        ]
        
        if self.use_adam8bit:
            print(f"        [fine_tune] using Adam 8-bit optimizer for {lora_adapter}")
            optimizer = bnb.optim.Adam8bit(params_to_tune, lr=lr)
            try:
                from bitsandbytes.optim import GlobalOptimManager
                gopm = GlobalOptimManager.get_instance()
                if hasattr(gopm, "register_model"):
                    gopm.register_model(self.llm.model_runner.model)
                if hasattr(gopm, "freeze_non_trainable_params"):
                    gopm.freeze_non_trainable_params(self.llm.model_runner.model)
            except Exception as exc:
                print(f"[warn] paging setup failed: {exc}")
        else:
            optimizer = torch.optim.Adam(params_to_tune, lr=lr)
        
        # Training loop
        for step in range(steps):
            async with self.lock:
                self.llm.train()
                optimizer.zero_grad()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                loss = self.llm.forward_train(input_ids, labels)
                loss.backward()
                optimizer.step()

            if step == 0 or step == steps - 1 or step % 10 == 0:
                print(f"        [fine_tune] {lora_adapter} step={step:3d}  loss={loss.item():.4f}")

            if step == 0:
                step_t0 = time.perf_counter()
            elif step == 1:
                step_dt_ms = (time.perf_counter() - step_t0) * 1000
                print(f"[metric] fine_tune_step_ms≈{step_dt_ms:.2f}")
            
            await asyncio.sleep(0)  # Yield to other async tasks

        # Final evaluation
        async with self.lock:
            self.llm.eval()
            with torch.no_grad():
                norms = {
                    n: p.norm().item() 
                    for n, p in self.llm.model_runner.model.named_parameters() 
                    if "lora_" in n
                }
                avg_norm = sum(norms.values()) / len(norms) if norms else 0.0
                print(f"        [fine_tune] {lora_adapter} average LoRA param norm={avg_norm:.4f}")

        total_ft_ms = (time.perf_counter() - t_ft_start) * 1000
        print(f"[metric] fine_tune_total_ms={total_ft_ms:.2f}")
        return optimizer.state_dict()

    async def learn_async(
        self,
        prompt: str,
        response: str,
        lora_adapter: str = "default",
        steps: int = 3,
        lr: float = 1e-4,
    ):
        """Asynchronously learn from a single prompt-response pair."""
        # Wait for any existing training task for this adapter
        if lora_adapter in self.training_tasks:
            await self.training_tasks.pop(lora_adapter)

        # Load existing state for this adapter
        async with self.lock:
            if lora_adapter in self.lora_states:
                self.load_lora_state(self.lora_states[lora_adapter])

        # Run fine-tuning
        examples = [{"prompt": prompt, "response": response}]
        opt_state = await self.fine_tune(examples, lora_adapter, steps, lr)

        # Save updated states
        async with self.lock:
            self.lora_states[lora_adapter] = self.get_lora_state()
            self.opt_states[lora_adapter] = opt_state

    async def generate_async(
        self,
        prompt: str,
        lora_adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """Asynchronously generate text from a prompt."""
        results = await self.batch_generate([prompt], sampling_params, lora_adapter)
        return results[0]

    async def stream_async(
        self,
        prompt: str,
        lora_adapter: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """Asynchronously stream text generation."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        # Load the appropriate LoRA adapter if specified
        if lora_adapter and lora_adapter in self.lora_states:
            async with self.lock:
                self.load_lora_state(self.lora_states[lora_adapter])

        # For this copy file version, use custom implementation streaming
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

    # Synchronous wrapper methods for backward compatibility
    def generate(self, prompt: str, lora_adapter: Optional[str] = None) -> str:
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
            return asyncio.run(self.generate_async(prompt, lora_adapter))

    def stream(self, prompt: str, lora_adapter: Optional[str] = None) -> Generator[str, None, None]:
        """Stream text generation (synchronous wrapper)."""
        result = self.generate(prompt, lora_adapter)
        for char in result:
            yield char

    def learn(self, prompt: str, response: str, lora_adapter: str = "default"):
        """Learn from a prompt-response pair (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            # If we're in an async context, raise an error with helpful message
            raise RuntimeError(
                "Cannot use sync learn() method from within async context. "
                "Use learn_async() instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                raise e
            # No event loop running, safe to use asyncio.run()
            asyncio.run(self.learn_async(prompt, response, lora_adapter))

    def chat(self, msgs: List[Dict[str, Any]], lora_adapter: Optional[str] = None) -> str:
        """Chat with the model using a list of messages."""
        # Convert messages to a single prompt (simplified)
        prompt = self._messages_to_prompt(msgs)
        return self.generate(prompt, lora_adapter)

    async def chat_async(self, msgs: List[Dict[str, Any]], lora_adapter: Optional[str] = None) -> str:
        """Chat with the model using a list of messages (async version)."""
        # Convert messages to a single prompt (simplified)
        prompt = self._messages_to_prompt(msgs)
        return await self.generate_async(prompt, lora_adapter)

    def chat_stream(self, msgs: List[Dict[str, Any]], lora_adapter: Optional[str] = None) -> Generator[str, None, None]:
        """Stream chat with the model using a list of messages."""
        prompt = self._messages_to_prompt(msgs)
        return self.stream(prompt, lora_adapter)

    def learn_chat(self, msgs: List[Dict[str, Any]], lora_adapter: str = "default"):
        """Learn from a conversation (simplified implementation)."""
        # Extract the last user message and assistant response
        if len(msgs) >= 2:
            user_msg = None
            assistant_msg = None
            for msg in reversed(msgs):
                if msg.get("role") == "assistant" and assistant_msg is None:
                    assistant_msg = msg.get("content", "")
                elif msg.get("role") == "user" and user_msg is None:
                    user_msg = msg.get("content", "")
                if user_msg and assistant_msg:
                    break
            
            if user_msg and assistant_msg:
                self.learn(user_msg, assistant_msg, lora_adapter)

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

    def delete_adapter(self, lora_adapter: str):
        """Delete a LoRA adapter and its associated states."""
        if lora_adapter in self.lora_states:
            del self.lora_states[lora_adapter]
        if lora_adapter in self.opt_states:
            del self.opt_states[lora_adapter]
        if lora_adapter in self.training_tasks:
            # Cancel the task if it's still running
            task = self.training_tasks.pop(lora_adapter)
            if not task.done():
                task.cancel()

    async def wait_for_training(self, lora_adapter: Optional[str] = None):
        """Wait for training tasks to complete."""
        if lora_adapter:
            if lora_adapter in self.training_tasks:
                await self.training_tasks[lora_adapter]
        else:
            # Wait for all training tasks
            if self.training_tasks:
                await asyncio.gather(*self.training_tasks.values())
