"""Training probe: can one LoRA step on a long agent run fit and how long does it take? (docs/04, P0b)

Memory and step time depend on sequence length, not content, so the probe trains on random tokens.
The LoRA targets here are the ones the SFT migration will use (docs/04, "Migrating
scripts/04_train_lora.py"): attention, MLP and the DeltaNet mixers. The old GSM8K config targeted
only attention and MLP names, which left the mixers in 24 of 32 layers untouched.
"""

import importlib.util
import math
import time

import torch

ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
DELTANET_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj"]
GATE_TARGETS = ["in_proj_a", "in_proj_b"]  # DeltaNet write-strength and decay gates; an open decision


def kernel_status():
    """Whether the fast DeltaNet kernels are importable. Without them transformers falls back to slow torch code."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}


def attach_lora(model, include_gates=False, r=16, alpha=32, grad_ckpt=True):
    from peft import LoraConfig, get_peft_model

    targets = ATTENTION_TARGETS + MLP_TARGETS + DELTANET_TARGETS + (GATE_TARGETS if include_gates else [])
    if grad_ckpt:
        # transformers 5.x renamed the method and its kwargs; keep older installs working.
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _peak_gib(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return None


def probe_length(model, length, steps, device, vocab_size, seed=0, lr=1e-4):
    """Run 1 warm-up step plus `steps` timed LoRA steps on one random sequence of `length` tokens."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (1, length), generator=g).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    record = {"length": length, "steps": steps, "device": device}
    try:
        times, loss, grad_norm = [], float("nan"), 0.0
        for step in range(steps + 1):
            _sync(device)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            grad_norm = float(torch.norm(torch.stack([p.grad.float().norm() for p in params if p.grad is not None])))
            opt.step()
            _sync(device)
            if step > 0:  # step 0 is warm-up (kernel compilation, allocator growth)
                times.append(time.perf_counter() - start)
        record.update(status="ok", sec_per_step=sorted(times)[len(times) // 2], loss_finite=math.isfinite(float(loss.detach())),
                      lora_grad_norm=grad_norm, peak_gib=_peak_gib(device))
    except torch.OutOfMemoryError:
        record.update(status="oom", peak_gib=_peak_gib(device))
    finally:
        opt.zero_grad(set_to_none=True)
    return record
