import sys
import time
from typing import Dict, List

import torch

from dark import LLM, SamplingParams
from dark.engine.sequence import Sequence

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_RANK = 4  # small for fast tests

Message = Dict[str, str]


def get_lora_state(model) -> Dict[str, torch.Tensor]:
    """Clone and return all LoRA parameters of the model."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state(model, state: Dict[str, torch.Tensor]):
    """Load LoRA tensors into the model (in-place)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                if n in state:
                    p.copy_(state[n])
                else:
                    p.zero_()


def stream_generate(llm: LLM, prompt: str, sp: SamplingParams) -> str:
    """Run generation step-by-step, returning the produced text."""
    tokenizer = llm.tokenizer
    seq = Sequence(tokenizer.encode(prompt), sp)
    llm.scheduler.add(seq)
    out = ""
    emitted = 0
    while not seq.is_finished:
        llm.step()
        while seq.num_completion_tokens > emitted:
            tid = seq.completion_token_ids[emitted]
            out += tokenizer.decode([tid], skip_special_tokens=True)
            emitted += 1
    return out


def fine_tune(llm: LLM, tokenizer, sentence: str, steps: int = 5, lr: float = 1e-3):
    """Quick LoRA fine-tune on a single sentence."""
    ids = tokenizer.encode(sentence, return_tensors="pt").cuda()
    optimizer = torch.optim.Adam(
        [p for n, p in llm.model_runner.model.named_parameters() if "lora_" in n], lr=lr
    )
    for _ in range(steps):
        optimizer.zero_grad()
        loss = llm.forward_train(ids)
        loss.backward()
        optimizer.step()
    return optimizer.state_dict()


def build_message_queue() -> List[Message]:
    msgs: List[Message] = []
    for _ in range(3):  # repeat pattern for longer queue
        msgs.extend([
            {"type": "train", "lora": "cats", "text": "Orange cats sing opera."},
            {"type": "generate", "lora": "cats", "text": "Orange cats"},
            {"type": "train", "lora": "turtles", "text": "Green turtles surf waves."},
            {"type": "generate", "lora": "turtles", "text": "Green turtles"},
        ])
    return msgs


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    llm = LLM(MODEL_PATH, enforce_eager=True, lora_rank=LORA_RANK)
    tokenizer = llm.tokenizer
    sp = SamplingParams(temperature=0.0, max_tokens=12)

    lora_states: Dict[str, Dict[str, torch.Tensor]] = {}
    opt_states: Dict[str, Dict] = {}

    msgs = build_message_queue()
    start = time.perf_counter()

    for i, msg in enumerate(msgs):
        lora_name = msg["lora"]

        # Load or init LoRA.
        if lora_name in lora_states:
            load_lora_state(llm.model_runner.model, lora_states[lora_name])
        else:
            # Zero init so first training starts from scratch.
            load_lora_state(llm.model_runner.model, {})

        if msg["type"] == "train":
            print(f"[{i}] TRAIN ({lora_name}): {msg['text'][:30]}...")
            opt_state = fine_tune(llm, tokenizer, msg["text"], steps=5)
            lora_states[lora_name] = get_lora_state(llm.model_runner.model)
            opt_states[lora_name] = opt_state
        else:
            output = stream_generate(llm, msg["text"], sp)
            print(f"[{i}] GEN ({lora_name}): {msg['text']} -> {output}")

    total_time = time.perf_counter() - start
    print(f"Processed {len(msgs)} messages in {total_time:.2f}s (avg {total_time/len(msgs):.3f}s/msg)")

    # Simple correctness: final generation from cats should contain opera.
    load_lora_state(llm.model_runner.model, lora_states["cats"])
    final = stream_generate(llm, "Orange cats", sp)
    if "opera" not in final.lower():
        print("Error: Expected keyword missing in final generation", file=sys.stderr)
        sys.exit(1)
    print("\nQueue streaming integration test passed!")


if __name__ == "__main__":
    main() 