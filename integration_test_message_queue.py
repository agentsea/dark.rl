import sys
import time
from typing import Dict, List

import torch

from dark import LLM, SamplingParams
from dark.engine.sequence import Sequence

MODEL_PATH = "Qwen/Qwen3-8B"
LORA_RANK = 8  # slightly larger rank for better capacity in tests
LORA_ALPHA = 32  # stronger scaling for LoRA updates

Message = Dict[str, str]


def get_lora_state(model) -> Dict[str, torch.Tensor]:
    """Clone and return all LoRA parameters of the model."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state(model, state: Dict[str, torch.Tensor]):
    """Load LoRA tensors into the model (in-place).

    If `state` is empty, we leave the existing randomly-initialized LoRA weights
    untouched so that training can make progress. Zeroing them would eliminate
    any gradient signal (since both LoRA A and B would start at 0), which we
    observed earlier.
    """
    t0 = time.perf_counter()
    if not state:
        # Keep default initialization.
        print(f"[metric] lora_load_ms=0.00 (no state)")
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n and n in state:
                p.copy_(state[n])
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"[metric] lora_load_ms={dt_ms:.2f}")


def stream_generate(llm: LLM, prompt: str, sp: SamplingParams) -> str:
    """Run generation step-by-step, returning the produced text."""
    gen_start = time.perf_counter()
    tokenizer = llm.tokenizer
    seq = Sequence(tokenizer.encode(prompt), sp)
    llm.scheduler.add(seq)
    out = ""
    emitted = 0
    while not seq.is_finished:
        step_t0 = time.perf_counter()
        llm.step()
        step_dt_ms = (time.perf_counter() - step_t0) * 1000
        print(f"[metric] llm_step_ms={step_dt_ms:.2f}")
        while seq.num_completion_tokens > emitted:
            tid = seq.completion_token_ids[emitted]
            out += tokenizer.decode([tid], skip_special_tokens=True)
            emitted += 1
    total_gen_ms = (time.perf_counter() - gen_start) * 1000
    per_tok = total_gen_ms / max(1, seq.num_completion_tokens)
    print(f"[metric] generate_total_ms={total_gen_ms:.2f}  generate_per_token_ms={per_tok:.2f}")
    return out


def fine_tune(llm: LLM, tokenizer, sentence: str, steps: int = 5, lr: float = 1e-3):
    """Quick LoRA fine-tune on a single sentence."""
    t_ft_start = time.perf_counter()
    # Add the EOS token to the training sentence to teach the model when to stop.
    sentence_with_eos = sentence + tokenizer.eos_token
    ids = tokenizer.encode(sentence_with_eos, return_tensors="pt").cuda()
    optimizer = torch.optim.Adam(
        [p for n, p in llm.model_runner.model.named_parameters() if "lora_" in n], lr=lr
    )
    for step in range(steps):
        optimizer.zero_grad()
        loss = llm.forward_train(ids)
        loss.backward()

        # Debug gradient norm of the first LoRA parameter to check if training signal flows.
        for n, p in llm.model_runner.model.named_parameters():
            if "lora_a" in n:
                grad_exists = p.grad is not None
                grad_norm = p.grad.norm().item() if grad_exists else 0.0
                print(f"            grad_norm({n})={grad_norm:.6f}  grad_exists={grad_exists}")
                # Also print training flag of the parent layer once.
                parent_module = llm.model_runner.model
                rep_layer = None
                # Try to retrieve module via name split
                try:
                    parts = n.split('.')[:-1]  # drop param name
                    rep_layer = parent_module
                    for part in parts:
                        rep_layer = getattr(rep_layer, part)
                except Exception:
                    rep_layer = None
                if rep_layer is not None:
                    print(f"            {'.'.join(parts)}.training={rep_layer.training}")
                break

        optimizer.step()

        # Debug print of loss to monitor convergence
        if step == 0 or step == steps - 1 or step % 10 == 0:
            print(f"        [fine_tune] step={step:3d}  loss={loss.item():.4f}")
        # Record per-step timing.
        if step == 0:
            step_t0 = time.perf_counter()
        elif step == 1:
            # Second iteration gives a more stable per-step time.
            step_dt_ms = (time.perf_counter() - step_t0) * 1000
            print(f"[metric] fine_tune_step_ms≈{step_dt_ms:.2f}")

    # Print the norm of LoRA parameters after training for diagnostics
    with torch.no_grad():
        norms = {n: p.norm().item() for n, p in llm.model_runner.model.named_parameters() if "lora_" in n}
        avg_norm = sum(norms.values()) / len(norms) if norms else 0.0
        print(f"        [fine_tune] average LoRA param norm={avg_norm:.4f}")

    total_ft_ms = (time.perf_counter() - t_ft_start) * 1000
    print(f"[metric] fine_tune_total_ms={total_ft_ms:.2f}")
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

    llm = LLM(MODEL_PATH, enforce_eager=True, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
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
            # Else: keep the random initialization already present.
            pass

        if msg["type"] == "train":
            print(f"[{i}] TRAIN ({lora_name}): {msg['text'][:30]}...")
            # Use more fine-tuning iterations with a lower learning rate so the LoRA
            # can reliably memorise the sentence without corrupting the base model.
            opt_state = fine_tune(llm, tokenizer, msg["text"], steps=2, lr=1e-4)
            lora_states[lora_name] = get_lora_state(llm.model_runner.model)
            opt_states[lora_name] = opt_state

            # After training, run an immediate generation for debugging.
            prompt_debug = " ".join(msg["text"].split()[:2])
            debug_out = stream_generate(llm, prompt_debug, sp)
            print(f"        [post-train] prompt='{prompt_debug}' -> {debug_out}")
        else:
            output = stream_generate(llm, msg["text"], sp)
            print(f"[{i}] GEN ({lora_name}): {msg['text']} -> {output}")

    total_time = time.perf_counter() - start
    print(f"Processed {len(msgs)} messages in {total_time:.2f}s (avg {total_time/len(msgs):.3f}s/msg)")

    # Simple correctness: final generation from cats should contain opera.
    load_lora_state(llm.model_runner.model, lora_states["cats"])
    final = stream_generate(llm, "Orange cats", sp)
    print(f"[final check] 'Orange cats' -> {final}")
    if "opera" not in final.lower():
        print("Error: Expected keyword missing in final generation", file=sys.stderr)
        sys.exit(1)
    print("\nQueue streaming integration test passed!")


if __name__ == "__main__":
    main() 