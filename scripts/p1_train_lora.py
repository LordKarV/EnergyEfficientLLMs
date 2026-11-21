"""P1 LoRA fine-tune on one arm's selected agent runs (docs/04, "Selection and training rules for P1").

The multi-turn replacement for scripts/04_train_lora.py, which stays as the single-turn GSM8K archive.
Loss mask, sum-over-constant loss, truncation policy and step plan are in qtrace/sft.py.

Run on the box:
    python scripts/p1_train_lora.py --config configs/p1_train.yaml --arm qtrace --seed 0
Token-matched rerun (docs/04, Training), stopping once the supervised tokens of another run are reached:
    python scripts/p1_train_lora.py --arm qtrace --seed 0 --match-supervised-tokens <its supervised_tokens_seen>
Writes outputs/runs/<arm>_seed<s>[_tokmatch<N>]/: loss.jsonl, train_meta.json and merged/ (model and
tokenizer, the input of the GGUF round trip).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace import sft
from qtrace.common import load_config
from qtrace.models import pick_device
from qtrace.probe import kernel_status


def check_kernels(allow_slow):
    """Kernel record for the metadata; exits unless both training-path ops resolved to the fast packages."""
    import transformers

    kernels = {"in_use": sft.kernel_paths(), "importable": kernel_status(), "transformers": transformers.__version__}
    if not all(op["fast"] for op in kernels["in_use"].values()) and not allow_slow:
        raise SystemExit(f"fla and causal_conv1d must be in use, transformers resolved {kernels['in_use']} "
                         "(docs/04 requires the fast kernels); pass --allow-slow-kernels only for a smoke test")
    return kernels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_train.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the arm file's selection seed; also seeds training")
    ap.add_argument("--match-supervised-tokens", type=int, default=None,
                    help="train until this many supervised tokens are seen; sets the step count up front")
    ap.add_argument("--allow-slow-kernels", action="store_true",
                    help="run on the torch DeltaNet fallback (tests and smoke runs only)")
    ap.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tr, model_path = cfg["train"], cfg["model"]["path"]
    kernels = check_kernels(args.allow_slow_kernels)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    arm_file = os.path.join(cfg["paths"]["arms_dir"], f"{args.arm}_seed{args.seed}.jsonl")
    rows = sft.load_arm(arm_file, args.arm, args.seed)
    examples, report = sft.build_examples(rows, tr["max_tokens"])
    plan = sft.plan_steps([e["n_sup"] for e in examples], tr["epochs"], tr["episodes_per_step"], args.seed,
                          match_tokens=args.match_supervised_tokens)
    warmup = round(tr["warmup_ratio"] * len(plan))
    suffix = "" if args.match_supervised_tokens is None else f"_tokmatch{args.match_supervised_tokens}"
    run_dir = os.path.join(cfg["paths"]["runs_dir"], f"{args.arm}_seed{args.seed}{suffix}")
    print(f"{len(examples)} runs, {len(plan)} steps, {report}")

    device = args.device or pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    kernels["attn_implementation"] = model.config._attn_implementation
    top = max(int(e["input_ids"].max()) for e in examples)
    if top >= model.config.vocab_size:
        raise ValueError(f"token id {top} is outside the model's vocabulary ({model.config.vocab_size})")
    model = sft.attach(model, tr, args.seed)
    model.print_trainable_parameters()

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "loss.jsonl"), "w") as f:
        def log(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"step {rec['step']:5d}  loss {rec['loss']:.4f}  per token {rec['loss_per_token']:.4f}  "
                  f"tokens {rec['supervised_tokens']}  grad {rec['grad_norm']:.3f}")

        records = sft.train(model, examples, plan, tr["learning_rate"], warmup, tr["loss_norm"], tr["chunk"], log)

    # Fold W <- W + BA and save the plain model: quantization (GGUF) must see the deployed weights.
    merged_dir = os.path.join(run_dir, "merged")
    model.merge_and_unload().save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    meta = {
        "arm": args.arm, "seed": args.seed, "arm_file": arm_file, "examples": len(examples), "steps": len(plan),
        "epochs": tr["epochs"], "episodes_per_step": tr["episodes_per_step"],
        "match_supervised_tokens": args.match_supervised_tokens,
        "supervised_tokens_seen": sum(r["supervised_tokens"] for r in records),
        "supervised_tokens_per_epoch": report["supervised_tokens_per_epoch"], "max_tokens": tr["max_tokens"],
        "truncated_supervised_share": report["truncated_supervised_share"],
        "truncated_episodes": report["truncated_episodes"], "loss_norm": tr["loss_norm"],
        "learning_rate": tr["learning_rate"], "warmup_steps": warmup, "grad_clip": sft.GRAD_CLIP,
        "targets": list(tr["targets"]), "include_gates": tr["include_gates"], "lora_r": tr["lora_r"],
        "lora_alpha": tr["lora_alpha"], "kernels": kernels, "model": model_path, "device": device,
        "dtype": str(dtype), "merged": merged_dir,
    }
    with open(os.path.join(run_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
