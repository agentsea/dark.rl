import sys
import time
from typing import Dict, List, Union

import torch
import bitsandbytes as bnb

from dark import LLM, SamplingParams
from dark.engine.sequence import Sequence

MODEL_PATH = "Qwen/Qwen3-8B"
LORA_RANK = 8  # slightly larger rank for better capacity in tests
LORA_ALPHA = 32  # stronger scaling for LoRA updates
OFFLOAD_LORA_TO_CPU = True  # Set to True to store LoRA weights on CPU and save VRAM.
USE_ADAM8BIT = True  # Default to using 8-bit Adam optimizer

Message = Dict[str, Union[str, int, List[str]]]


def get_lora_state(model, to_cpu: bool = False) -> Dict[str, torch.Tensor]:
    """Clone and return all LoRA parameters of the model.

    Args:
        model: The model containing the LoRA parameters.
        to_cpu: If True, the returned tensors will be on the CPU to save VRAM.
    """
    return {n: p.detach().clone().to("cpu" if to_cpu else p.device) for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state(model, state: Dict[str, torch.Tensor]):
    """Load LoRA tensors into the model (in-place).

    If `state` is empty, we leave the existing randomly-initialized LoRA weights
    untouched so that training can make progress. Zeroing them would eliminate
    any gradient signal (since both LoRA A and B would start at 0)
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


def stream_generate(llm: LLM, prompts: List[str], sp: SamplingParams) -> List[str]:
    """Run generation step-by-step on a batch of prompts."""
    gen_start = time.perf_counter()
    tokenizer = llm.tokenizer

    seqs = []
    for prompt in prompts:
        # For each prompt, create `sp.n` sequences to generate `n` different
        # completions.
        for _ in range(sp.n):
            seqs.append(Sequence(tokenizer.encode(prompt), sp))

    for seq in seqs:
        llm.scheduler.add(seq)

    outs = ["" for _ in seqs]
    emitted = [0 for _ in seqs]
    while any(not seq.is_finished for seq in seqs):
        step_t0 = time.perf_counter()
        llm.step()
        step_dt_ms = (time.perf_counter() - step_t0) * 1000
        print(f"[metric] llm_step_ms={step_dt_ms:.2f}")
        for i, seq in enumerate(seqs):
            while seq.num_completion_tokens > emitted[i]:
                tid = seq.completion_token_ids[emitted[i]]
                outs[i] += tokenizer.decode([tid], skip_special_tokens=True)
                emitted[i] += 1
    total_gen_ms = (time.perf_counter() - gen_start) * 1000
    num_tokens = sum(s.num_completion_tokens for s in seqs)
    per_tok = total_gen_ms / max(1, num_tokens)
    print(
        f"[metric] generate_total_ms={total_gen_ms:.2f}  "
        f"generate_per_token_ms={per_tok:.2f}"
    )
    return outs


def fine_tune(llm: LLM, tokenizer, examples: List[Dict[str, str]], steps: int = 5, lr: float = 1e-3, use_adam8bit: bool = USE_ADAM8BIT):
    """Quick LoRA fine-tune on a batch of prompt/response pairs."""
    t_ft_start = time.perf_counter()

    llm.train()

    prompts = [ex["prompt"] for ex in examples]
    responses = [ex["response"] for ex in examples]

    # Create combined prompt+response sequences, with EOS at the end of each response.
    # The tokenizer will handle padding.
    prompt_ids = tokenizer(prompts, padding=False).input_ids
    response_ids = tokenizer(responses, padding=False).input_ids
    
    input_ids = []
    labels = []
    for i in range(len(prompts)):
        prompt_len = len(prompt_ids[i])
        # Concatenate prompt and response, and add EOS.
        input_id = prompt_ids[i] + response_ids[i] + [tokenizer.eos_token_id]
        # Create labels that ignore the prompt part.
        label = [-100] * prompt_len + response_ids[i] + [tokenizer.eos_token_id]
        input_ids.append(torch.tensor(input_id))
        labels.append(torch.tensor(label))

    # Pad batches to the same length.
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).cuda()

    params_to_tune = [
        p for n, p in llm.model_runner.model.named_parameters() if "lora_" in n
    ]
    if use_adam8bit:
        print("        [fine_tune] using Adam 8-bit optimizer")
        optimizer = bnb.optim.Adam8bit(params_to_tune, lr=lr)
    else:
        optimizer = torch.optim.Adam(params_to_tune, lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        loss = llm.forward_train(input_ids, labels)
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

    llm.eval()

    total_ft_ms = (time.perf_counter() - t_ft_start) * 1000
    print(f"[metric] fine_tune_total_ms={total_ft_ms:.2f}")
    return optimizer.state_dict()


def build_message_queue() -> List[Message]:
    msgs: List[Message] = []
    for _ in range(3):  # repeat pattern for longer queue
        msgs.extend(
            [
                {
                    "type": "train",
                    "lora": "cats",
                    "messages": [
                        {"prompt": "Orange cats are known for", "response": "singing opera."},
                        {"prompt": "The tabby cat enjoys", "response": "jazz music."},
                    ],
                },
                {
                    "type": "generate",
                    "lora": "cats",
                    "prompt": ["Orange cats are known for", "The tabby cat enjoys"],
                    "n": 2,
                },
                {
                    "type": "train",
                    "lora": "turtles",
                    "messages": [
                        {"prompt": "Green turtles can be found", "response": "surfing waves."},
                        {"prompt": "Sea turtles are famous for", "response": "migrating long distances."},
                    ],
                },
                {
                    "type": "generate",
                    "lora": "turtles",
                    "prompt": ["Green turtles can be found", "Sea turtles are famous for"],
                    "n": 1,
                },
            ]
        )
    return msgs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a message queue integration test for LoRA fine-tuning and generation.")
    parser.add_argument('--use-adam8bit', dest='use_adam8bit', action='store_true', help="Use 8-bit Adam optimizer.")
    parser.add_argument('--no-use-adam8bit', dest='use_adam8bit', action='store_false', help="Do not use 8-bit Adam optimizer.")
    parser.set_defaults(use_adam8bit=USE_ADAM8BIT)
    args = parser.parse_args()

    print(f"Using Adam 8-bit optimizer: {args.use_adam8bit}")

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    if OFFLOAD_LORA_TO_CPU:
        print("--- LoRA CPU offloading is ENABLED ---")

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
            print(f"[{i}] TRAIN ({lora_name}): {msg['messages']}")
            # Use more fine-tuning iterations with a lower learning rate so the LoRA
            # can reliably memorise the sentence without corrupting the base model.
            opt_state = fine_tune(llm, tokenizer, msg["messages"], steps=3, lr=1e-4, use_adam8bit=args.use_adam8bit)
            lora_states[lora_name] = get_lora_state(llm.model_runner.model, to_cpu=OFFLOAD_LORA_TO_CPU)
            opt_states[lora_name] = opt_state

            # After training, run an immediate generation for debugging.
            prompts_debug = [ex["prompt"] for ex in msg["messages"]]
            sp_debug = SamplingParams(temperature=0.0, max_tokens=12, n=1)
            debug_out = stream_generate(llm, prompts_debug, sp_debug)
            for prompt, out in zip(prompts_debug, debug_out):
                print(f"        [post-train] prompt='{prompt}' -> {out}")
        else:
            prompts = msg["prompt"]
            sp.n = msg.get("n", 1)
            output = stream_generate(llm, prompts, sp)
            print(f"[{i}] GEN ({lora_name}):")
            for i in range(len(prompts)):
                for j in range(sp.n):
                    print(f"  '{prompts[i]}' -> {output[i*sp.n+j]}")

    total_time = time.perf_counter() - start
    print(f"Processed {len(msgs)} messages in {total_time:.2f}s (avg {total_time/len(msgs):.3f}s/msg)")

    # Simple correctness: final generation from cats should contain opera.
    load_lora_state(llm.model_runner.model, lora_states["cats"])
    final_outputs = stream_generate(llm, ["Orange cats are known for"], sp)
    final = final_outputs[0]
    print(f"[final check] 'Orange cats are known for' -> {final}")
    if "opera" not in final.lower():
        print("Error: Expected keyword missing in final generation", file=sys.stderr)
        sys.exit(1)
    print("\nQueue streaming integration test passed!")


if __name__ == "__main__":
    main() 