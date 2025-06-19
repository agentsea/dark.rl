import sys
import time
from typing import Dict, List, Union

import torch
import bitsandbytes as bnb
import asyncio
import os
import torch.multiprocessing as mp

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


async def stream_generate(
    llm: LLM,
    prompts: List[str],
    sp: SamplingParams,
    lock: asyncio.Lock,
    infer_stream: torch.cuda.Stream,
) -> List[str]:
    """Run generation step-by-step on a batch of prompts, asynchronously."""
    gen_start = time.perf_counter()
    tokenizer = llm.tokenizer

    seqs = []
    for prompt in prompts:
        # For each prompt, create `sp.n` sequences to generate `n` different
        # completions.
        for _ in range(sp.n):
            seqs.append(Sequence(tokenizer.encode(prompt), sp))

    async with lock:
        llm.eval()
        for seq in seqs:
            llm.scheduler.add(seq)

    outs = ["" for _ in seqs]
    emitted = [0 for _ in seqs]
    while any(not seq.is_finished for seq in seqs):
        step_t0 = time.perf_counter()
        async with lock:
            llm.eval()
            with torch.cuda.stream(infer_stream):
                llm.step()
        step_dt_ms = (time.perf_counter() - step_t0) * 1000
        print(f"[metric] llm_step_ms={step_dt_ms:.2f}")

        for i, seq in enumerate(seqs):
            while seq.num_completion_tokens > emitted[i]:
                tid = seq.completion_token_ids[emitted[i]]
                outs[i] += tokenizer.decode([tid], skip_special_tokens=True)
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


async def fine_tune(
    llm: LLM,
    tokenizer,
    examples: List[Dict[str, str]],
    lock: asyncio.Lock,
    steps: int = 5,
    lr: float = 1e-3,
    use_adam8bit: bool = USE_ADAM8BIT,
):
    """Quick LoRA fine-tune on a batch of prompt/response pairs, asynchronously."""
    t_ft_start = time.perf_counter()

    # Data preparation can happen outside the lock.
    prompts = [ex["prompt"] for ex in examples]
    responses = [ex["response"] for ex in examples]
    prompt_ids = tokenizer(prompts, padding=False).input_ids
    response_ids = tokenizer(responses, padding=False).input_ids
    
    input_ids = []
    labels = []
    for i in range(len(prompts)):
        prompt_len = len(prompt_ids[i])
        input_id = prompt_ids[i] + response_ids[i] + [tokenizer.eos_token_id]
        label = [-100] * prompt_len + response_ids[i] + [tokenizer.eos_token_id]
        input_ids.append(torch.tensor(input_id))
        labels.append(torch.tensor(label))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).cuda()

    params_to_tune = [p for n, p in llm.model_runner.model.named_parameters() if "lora_" in n]
    if use_adam8bit:
        print("        [fine_tune] using Adam 8-bit optimizer")
        optimizer = bnb.optim.Adam8bit(params_to_tune, lr=lr)
        # Register with the global bitsandbytes optimizer manager so that
        # the 8-bit state tensors (moments) live on CPU and are paged to
        # GPU chunk-by-chunk, saving VRAM.
        try:
            from bitsandbytes.optim import GlobalOptimManager
            gopm = GlobalOptimManager.get_instance()
            if hasattr(gopm, "register_model"):
                gopm.register_model(llm.model_runner.model)
            if hasattr(gopm, "freeze_non_trainable_params"):
                gopm.freeze_non_trainable_params(llm.model_runner.model)
        except Exception as exc:
            print(f"[warn] paging setup failed: {exc}")
    else:
        optimizer = torch.optim.Adam(params_to_tune, lr=lr)
    
    for step in range(steps):
        async with lock:
            llm.train()
            optimizer.zero_grad()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            loss = llm.forward_train(input_ids, labels)
            loss.backward()

            # Debug gradient norm
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

        if step == 0 or step == steps - 1 or step % 10 == 0:
            print(f"        [fine_tune] step={step:3d}  loss={loss.item():.4f}")

        if step == 0:
            step_t0 = time.perf_counter()
        elif step == 1:
            step_dt_ms = (time.perf_counter() - step_t0) * 1000
            print(f"[metric] fine_tune_step_ms≈{step_dt_ms:.2f}")
        
        await asyncio.sleep(0)  # Yield to other async tasks

    async with lock:
        llm.eval()
        with torch.no_grad():
            norms = {n: p.norm().item() for n, p in llm.model_runner.model.named_parameters() if "lora_" in n}
            avg_norm = sum(norms.values()) / len(norms) if norms else 0.0
            print(f"        [fine_tune] average LoRA param norm={avg_norm:.4f}")

    total_ft_ms = (time.perf_counter() - t_ft_start) * 1000
    print(f"[metric] fine_tune_total_ms={total_ft_ms:.2f}")
    return optimizer.state_dict()


async def run_train_task(
    lock: asyncio.Lock,
    infer_stream: torch.cuda.Stream,
    llm: LLM,
    tokenizer,
    msg: Message,
    lora_states: Dict[str, Dict[str, torch.Tensor]],
    opt_states: Dict[str, Dict],
    use_adam8bit: bool,
):
    """An async task to run fine-tuning in the background."""
    lora_name = msg["lora"]
    print(f"Running fine-tuning for LoRA: {lora_name}")

    async with lock:
        # Load the current state for this LoRA adapter before training.
        if lora_name in lora_states:
            load_lora_state(llm.model_runner.model, lora_states[lora_name])

    # Run the asynchronous fine-tuning function.
    opt_state = await fine_tune(
        llm,
        tokenizer,
        msg["messages"],
        lock,
        steps=3,
        lr=1e-4,
        use_adam8bit=use_adam8bit,
    )

    async with lock:
        # Save the updated LoRA and optimizer states.
        lora_states[lora_name] = get_lora_state(
            llm.model_runner.model, to_cpu=OFFLOAD_LORA_TO_CPU
        )
        opt_states[lora_name] = opt_state

    # After training, run an immediate generation for debugging. The lock is
    # released before this call.
    prompts_debug = [ex["prompt"] for ex in msg["messages"]]
    sp_debug = SamplingParams(temperature=0.0, max_tokens=12, n=1)
    debug_out = await stream_generate(
        llm, prompts_debug, sp_debug, lock, infer_stream
    )
    for prompt, out in zip(prompts_debug, debug_out):
        print(f"        [post-train] prompt='{prompt}' -> {out}")


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


def benchmark_worker(q, llm_config, prompt, sp):
    """A separate process to run a benchmark and put the result in a queue."""
    # We must import torch inside the worker
    import torch
    from dark import LLM
    
    # We need to re-create the LLM object to respect the config change
    llm_bench = LLM(MODEL_PATH, **llm_config)
    
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm_bench.eval()
    _ = llm_bench.generate([prompt], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    q.put((dt, peak))
    del llm_bench


async def main():
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
    training_tasks: Dict[str, asyncio.Task] = {}
    lock = asyncio.Lock()
    infer_stream = torch.cuda.Stream()

    msgs = build_message_queue()
    start = time.perf_counter()

    for i, msg in enumerate(msgs):
        lora_name = msg["lora"]

        # Before processing a new message for a LoRA, wait for any existing
        # training task for the same LoRA to complete.
        if lora_name in training_tasks:
            await training_tasks.pop(lora_name)

        if msg["type"] == "train":
            print(f"[{i}] QUEUE TRAIN ({lora_name})")
            # Create a background task for fine-tuning.
            task = asyncio.create_task(
                run_train_task(
                    lock,
                    infer_stream,
                    llm,
                    tokenizer,
                    msg,
                    lora_states,
                    opt_states,
                    args.use_adam8bit,
                )
            )
            training_tasks[lora_name] = task
        else:  # "generate"
            print(f"[{i}] GEN ({lora_name}):")
            async with lock:
                # Load the appropriate LoRA state for generation.
                if lora_name in lora_states:
                    load_lora_state(llm.model_runner.model, lora_states[lora_name])

            prompts = msg["prompt"]
            sp.n = msg.get("n", 1)
            output = await stream_generate(llm, prompts, sp, lock, infer_stream)
            for i_prompt in range(len(prompts)):
                for j in range(sp.n):
                    print(f"  '{prompts[i_prompt]}' -> {output[i_prompt*sp.n+j]}")

    # Wait for any remaining background training tasks to complete.
    await asyncio.gather(*training_tasks.values())

    total_time = time.perf_counter() - start
    print(f"Processed {len(msgs)} messages in {total_time:.2f}s (avg {total_time/len(msgs):.3f}s/msg)")

    # Simple correctness: final generation from cats should contain opera.
    # The lock is acquired only for loading the state.
    async with lock:
        load_lora_state(llm.model_runner.model, lora_states["cats"])

    # Generation is called after the lock is released.
    final_outputs = await stream_generate(
        llm, ["Orange cats are known for"], sp, lock, infer_stream
    )
    final = final_outputs[0]
    print(f"[final check] 'Orange cats are known for' -> {final}")
    if "opera" not in final.lower():
        print(
            "Error: Expected keyword missing in final generation", file=sys.stderr
        )
        sys.exit(1)
    print("\nQueue streaming integration test passed!")

    # ------------------------------------------------------------------
    # Quick sanity-check for fused vs unfused LoRA path.  We measure one
    # short generation in `train()` mode (unfused) and in `eval()` mode
    # (fused) and print timing + peak memory so the CI log shows the win.
    # This does *not* assert on exact numbers – variability across GPUs
    # could break the test – but it gives visibility for humans.
    # ------------------------------------------------------------------
    from contextlib import nullcontext
    # torch is already imported at module scope; avoid reimporting locally

    def _bench(tag: str):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _ = llm.generate(["Benchmark"], SamplingParams(max_tokens=8), use_tqdm=False)
        dt_ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"[fused_test] {tag:<7}  time={dt_ms:6.1f} ms  peak_mem={peak:7.1f} MB")

    llm.train()  # unfused path
    _bench("unfused")

    llm.eval()   # fused path
    _bench("fused")

    # ---------------- gradient-checkpointing comparison -----------------
    vocab_size = llm.tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64), device="cuda")
    labels = input_ids.clone()

    def _bench_train(tag: str, ckpt: bool):
        llm.model_runner.model.model.gradient_checkpointing = ckpt
        llm.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = llm.forward_train(input_ids, labels)
        loss.backward()
        dt_ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"[ckpt_test] {tag:<9}  time={dt_ms:6.1f} ms  peak_mem={peak:8.1f} MB  loss={loss.item():.4f}")

    _bench_train("no_ckpt", False)
    _bench_train("ckpt_on", True)

    # ------------ 8-bit Adam – paging vs no-paging -----------------
    def _bench_opt(tag: str, with_paging: bool):
        params = [p for n, p in llm.model_runner.model.named_parameters()
                  if "lora_" in n]
        opt = bnb.optim.Adam8bit(params, lr=1e-4)

        if with_paging:
            try:
                from bitsandbytes.optim import GlobalOptimManager
                gopm = GlobalOptimManager.get_instance()
                if hasattr(gopm, "register_model"):
                    gopm.register_model(llm.model_runner.model)
                if hasattr(gopm, "freeze_non_trainable_params"):
                    gopm.freeze_non_trainable_params(llm.model_runner.model)
            except Exception as exc:
                print(f"[warn] paging setup failed: {exc}")

        llm.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = llm.forward_train(input_ids, labels)
        loss.backward()
        opt.step()
        dt_ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"[opt_test] {tag:<12}  peak_mem={peak:8.1f} MB")

    _bench_opt("adam8bit_raw",  False)
    _bench_opt("adam8bit_page", True)

    # ------------------------------------------------------------------
    # Quick sanity-check for SDP-Flash attention.
    # We run each path in a separate process to guarantee a clean slate
    # for GPU memory measurement and avoid OOMs from fragmentation.
    # ------------------------------------------------------------------
    prompt = "Once upon a time " * 512          # 8-k token prompt
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    
    # Use spawn context for CUDA safety
    mp.set_start_method("spawn", force=True)

    def _run_bench_in_process(tag: str, enable_flash: bool):
        q = mp.Queue()
        llm_config = {
            "enforce_eager": True,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "use_flash_attention": enable_flash,
        }
        p = mp.Process(target=benchmark_worker, args=(q, llm_config, prompt, sp))
        p.start()
        p.join()
        dt, peak = q.get()
        print(f"[{tag:12}]   {dt:5.2f}s   peak={peak:8.1f} MB")

    _run_bench_in_process("EagerAttn", enable_flash=False)
    _run_bench_in_process("FlashAttn", enable_flash=True)


if __name__ == "__main__":
    asyncio.run(main()) 