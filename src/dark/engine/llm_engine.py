from time import perf_counter

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from dark.config import Config
from dark.engine.model_runner import ModelRunner
from dark.engine.scheduler import Scheduler
from dark.engine.sequence import Sequence
from dark.sampling_params import SamplingParams


class LLMEngine:
    """
    The main engine for the Large Language Model (LLM).

    This class orchestrates the entire generation process. It initializes and manages
    all the necessary components, including the model configuration, tokenizer,
    model runner, and scheduler. It provides a high-level API for adding requests
    and generating text, and it contains the central loop that drives the inference
    or training process.
    """

    def __init__(self, model, **kwargs):
        """
        Initializes the LLMEngine.

        Args:
            model: The path to the model or a model identifier from Hugging Face.
            **kwargs: Additional configuration options that will override the defaults.
        """
        model_path = snapshot_download(repo_id=model)
        config = Config(model_path)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        Sequence.block_size = config.kvcache_block_size

        # Load the corrected local config file, not the one from the Hub.
        config.hf_config = AutoConfig.from_pretrained("./corrected_config.json")
        config.max_model_len = min(
            config.max_model_len, config.hf_config.max_position_embeddings
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # Initialize the core components.
        self.model_runner = ModelRunner(config)
        self.scheduler = Scheduler(config)

    def train(self):
        """Switches the underlying model to training mode."""
        self.model_runner.train()

    def eval(self):
        """Switches the underlying model to evaluation mode."""
        self.model_runner.eval()

    def forward_train(self, input_ids):
        """
        Performs a forward pass for training, allowing gradients to be computed.

        This is a pass-through to the model runner's training-specific run method.
        """
        _, loss = self.model_runner.run_train_model(input_ids, labels=input_ids.clone())
        return loss

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        Adds a new generation request to the engine.

        The prompt is tokenized and encapsulated in a `Sequence` object, which is
        then added to the scheduler's queue.
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        Performs a single step of the generation process.

        This involves:
        1. Scheduling the next batch of sequences to run.
        2. Executing the model on the batch.
        3. Post-processing the results to update sequence states.

        Returns:
            A tuple containing a list of finished sequences and the number of
            tokens processed in this step.
        """
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        token_ids = self.model_runner.run(seqs)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """Checks if all generation requests have been completed."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        A high-level method to generate text for a list of prompts.

        It adds all prompts as requests and then runs the `step` method in a loop
        until all sequences are finished. It collects and returns the generated
        text for each prompt.
        """
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
            )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Add all prompts to the scheduler.
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        # Run the generation loop until all sequences are done.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            # Collect outputs from finished sequences.
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Decode and return the final results.
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
