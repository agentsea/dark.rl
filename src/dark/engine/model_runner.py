import torch
from torch import nn
import os
import itertools

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
            config,
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
        """Runs a forward pass for training using variable-length batching."""
        # This assumes the input_ids and labels are already padded to the same length
        # We will pack them into a single sequence for varlen attention.
        bsz, seqlen = input_ids.shape
        cu_seqlens = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device="cuda")
        max_seqlen = seqlen
        
        packed_input_ids = input_ids.view(-1)
        packed_labels = labels.view(-1)

        outputs = self.model(
            input_ids=packed_input_ids, 
            cu_seqlens=cu_seqlens, 
            max_seqlen=max_seqlen,
            labels=packed_labels
        )
        return outputs.logits, outputs.loss

    @torch.inference_mode()
    def run_model(self, seqs: list[Sequence]):
        """Runs the model's forward pass for inference using variable-length batching."""
        self.eval()

        if not seqs:
            return []

        token_ids_list = [s.token_ids for s in seqs]
        seq_lens = [len(ids) for ids in token_ids_list]
        max_seqlen = max(seq_lens) if seq_lens else 0

        packed_input_ids = torch.cat([torch.tensor(ids, dtype=torch.long) for ids in token_ids_list], dim=0).to("cuda")
        cu_seqlens = torch.tensor([0] + list(itertools.accumulate(seq_lens)), dtype=torch.int32, device="cuda")
        
        logits = self.model(
            input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ).logits

        last_token_indices = cu_seqlens[1:] - 1
        logits_last = logits[last_token_indices]

        temperatures = torch.tensor([s.temperature for s in seqs], device="cuda")
        next_tokens = self.sampler(logits_last, temperatures)

        for i, seq in enumerate(seqs):
            if temperatures[i] == 0 and next_tokens[i] == seq.last_token:
                top2 = logits_last[i].topk(2).indices
                fallback = top2[1].item() if top2[0].item() == next_tokens[i] else top2[0].item()
                next_tokens[i] = fallback
        return next_tokens.tolist()

    def run(self, seqs: list[Sequence]) -> list[int]:
        """The main run method, dispatching to the varlen model runner."""
        return self.run_model(seqs)
