"""P0 training probe (docs/04, P0b pilot): peak memory and seconds per LoRA step at 16K/32K/64K tokens.

The loss is the one SFT uses: chunked cross-entropy over supervised (model-written) tokens only,
probed at two supervised fractions. --full-logits-loss reproduces the transformers full-vocab loss as a
comparison point; it is expected to run out of memory at long lengths.

LoRA targets, r, alpha and include_gates come from the P1 training config, so the probe measures the
adapter P1 will train. Each row records them.

Run on the pilot box before the pilot attempts:
    python scripts/p0_training_probe.py --model Qwen/Qwen3.5-4B --config configs/p1_train.yaml
Writes one JSON line per length to outputs/probe/probe.jsonl and prints a table.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace import sft
from qtrace.common import load_config
from qtrace.probe import kernel_paths, kernel_status, probe_length


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--config", default="configs/p1_train.yaml",
                    help="P1 training config; its train section sets targets, lora_r, lora_alpha, include_gates")
    ap.add_argument("--include-gates", action="store_true",
                    help="also put LoRA on in_proj_a / in_proj_b, whatever the config says")
    ap.add_argument("--supervised-frac", type=float, nargs="+", default=[0.15, 0.5],
                    help="share of tokens the model wrote (T1 measured about 13%% in SWE-agent runs)")
    ap.add_argument("--chunk", type=int, default=2048, help="positions per logits chunk in the loss")
    ap.add_argument("--full-logits-loss", action="store_true", help="comparison: full-vocab transformers loss")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="run even if fla / causal_conv1d are not in use (numbers would not represent training)")
    ap.add_argument("--out", default="outputs/probe/probe.jsonl")
    args = ap.parse_args(argv)
    tr = load_config(args.config)["train"]
    lora = {"targets": list(tr["targets"]), "lora_r": tr["lora_r"], "lora_alpha": tr["lora_alpha"],
            "include_gates": bool(tr["include_gates"] or args.include_gates)}

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # Importable is not in use: transformers falls back to torch on any import error (see kernel_paths).
    kernels = {"in_use": kernel_paths(), "importable": kernel_status()}
    print(f"device {device}, kernels {kernels}")
    if not all(op["fast"] for op in kernels["in_use"].values()) and not args.allow_slow_kernels:
        raise SystemExit(f"fla and causal_conv1d must be in use, transformers resolved {kernels['in_use']} "
                         "(docs/04 requires the fast kernels); pass --allow-slow-kernels only for a smoke test")

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model = sft.attach(model, lora, seed=0)  # the trainer's attach: same targets check, same adapter
    model.print_trainable_parameters()
    vocab = model.config.vocab_size

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        path = "full_logits" if args.full_logits_loss else "chunked"
        for frac in args.supervised_frac:
            for length in sorted(args.lengths):
                rec = probe_length(model, length, args.steps, device, vocab, supervised_frac=frac,
                                   loss_path=path, chunk=args.chunk)
                rec.update(model=args.model, config=args.config, **lora, kernels=kernels)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                sec = f"{rec['sec_per_step']:.1f} s/step" if rec["status"] == "ok" else "OOM"
                mem = f"{rec['mem_gib']:.1f} GiB ({rec['mem_kind']})" if rec.get("mem_gib") is not None else "n/a"
                print(f"{path} frac {frac:.2f} {length:>6} tokens: {sec}, memory {mem}")
                if rec["status"] == "oom":
                    print("stopping this fraction: longer sequences will not fit either")
                    break


if __name__ == "__main__":
    main()
