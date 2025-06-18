import torch
from torch import nn

from dark.config import Config
from dark.engine.sequence import Sequence
from dark.layers.sampler import Sampler
from dark.models.qwen3 import Qwen3ForCausalLM
from dark.utils.loader import load_model


class ModelRunner:
    """
    Manages the low-level execution of the model.
    """

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(
            hf_config,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        load_model(self.model, config.model)

        if self.config.lora_rank > 0:
            self.model.freeze_base_model()
            self.model.train()

        self.sampler = Sampler()
        torch.set_default_device("cpu")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def run_train_model(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Runs a forward pass for training."""
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.logits, outputs.loss

    @torch.inference_mode()
    def run_model(self, seqs: list[Sequence]):
        """Runs the model's forward pass for inference."""
        # Ensure evaluation mode (i.e., deactivate dropout) during inference.
        self.eval()
        input_ids = torch.tensor([s.token_ids for s in seqs], device="cuda")
        temperatures = torch.tensor([s.temperature for s in seqs], device="cuda")
        logits = self.model(input_ids=input_ids).logits

        # Greedy sampling with a simple repetition-avoidance hack: if the top
        # token is identical to the previously generated token for the
        # sequence, fall back to the 2nd-best token. This mitigates pathological
        # "token token token ..." loops we observed with larger models (e.g.
        # Qwen3-4B) during short LoRA fine-tunes while keeping behaviour for
        # stochastic sampling unchanged.

        logits_last = logits[:, -1, :]
        next_tokens = self.sampler(logits_last, temperatures)

        for i, seq in enumerate(seqs):
            if temperatures[i] == 0 and next_tokens[i] == seq.last_token:
                # Pick the second-highest logit instead.
                top2 = logits_last[i].topk(2).indices
                # Ensure we actually have two distinct indices.
                fallback = top2[1].item() if top2[0].item() == next_tokens[i] else top2[0].item()
                next_tokens[i] = fallback
        return next_tokens.tolist()

    def run(self, seqs: list[Sequence]) -> list[int]:
        """
        The main run method that orchestrates a single generation step.
        """
        return self.run_model(seqs)
