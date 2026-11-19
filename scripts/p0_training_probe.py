"""P0 training probe (docs/04, P0b pilot): peak memory and seconds per LoRA step at 16K/32K/64K tokens.

Run on the pilot box before the pilot attempts:
    python scripts/p0_training_probe.py --model Qwen/Qwen3.5-4B
Writes one JSON line per length to outputs/probe/probe.jsonl and prints a table.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.probe import attach_lora, kernel_status, probe_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--include-gates", action="store_true", help="also put LoRA on in_proj_a / in_proj_b")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="run even if fla / causal_conv1d are missing (numbers would not represent training)")
    ap.add_argument("--out", default="outputs/probe/probe.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    kernels = kernel_status()
    print(f"device {device}, kernels {kernels}")
    if not all(kernels.values()) and not args.allow_slow_kernels:
        raise SystemExit("fla and causal_conv1d must be installed (docs/04 requires the fast kernels); "
                         "pass --allow-slow-kernels only for a smoke test")

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model = attach_lora(model, include_gates=args.include_gates)
    model.print_trainable_parameters()
    vocab = model.config.vocab_size

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        for length in sorted(args.lengths):
            rec = probe_length(model, length, args.steps, device, vocab)
            rec.update(model=args.model, include_gates=args.include_gates, kernels=kernels)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            sec = f"{rec['sec_per_step']:.1f} s/step" if rec["status"] == "ok" else "OOM"
            peak = f"{rec['peak_gib']:.1f} GiB" if rec.get("peak_gib") is not None else "n/a"
            print(f"{length:>6} tokens: {sec}, peak {peak}")
            if rec["status"] == "oom":
                print("stopping: longer sequences will not fit either")
                break


if __name__ == "__main__":
    main()
