"""Step 4: fine-tune the model with LoRA on one arm's training set.

The training loop is written out by hand (no Trainer class) so every step is visible:
tokenize -> mask the prompt out of the loss -> forward -> backward -> optimizer step.
At the end the LoRA weights are merged into the base model and the merged model is
saved, because quantization in step 5 must be applied to the final deployed weights.

Run:  python scripts/04_train_lora.py --config configs/base.yaml --arm qtrace --seed 0
"""

import argparse
import json
import os
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl, set_seed, write_jsonl
from qtrace.data import build_prompt
from qtrace.models import load_tokenizer, pick_device

IGNORE = -100  # label value that PyTorch's cross-entropy loss skips


def encode(tokenizer, row, enable_thinking):
    """Turn one training row into token ids and labels.

    Labels equal the input ids on the solution tokens and IGNORE on the prompt
    tokens, so the model is only trained to produce the solution.
    """
    prompt = build_prompt(tokenizer, row["question"], enable_thinking)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    solution_ids = tokenizer(row["text"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + solution_ids,
        "labels": [IGNORE] * len(prompt_ids) + solution_ids,
        "n_solution_tokens": len(solution_ids),
    }


def collate(examples, pad_id):
    """Pad a list of examples on the right to the same length and stack them into tensors."""
    longest = max(len(e["input_ids"]) for e in examples)
    input_ids, labels, attention = [], [], []
    for e in examples:
        n_pad = longest - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_id] * n_pad)
        labels.append(e["labels"] + [IGNORE] * n_pad)
        attention.append([1] * len(e["input_ids"]) + [0] * n_pad)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention)


def learning_rate_factor(step, warmup_steps, max_steps):
    """Linear warmup from 0 to 1, then linear decay back to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tr = cfg["train"]
    seed = tr["seed"] if args.seed is None else args.seed
    set_seed(seed)

    run_dir = os.path.join(cfg["paths"]["runs_dir"], f"{args.arm}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    device = pick_device()

    # ---- data ----
    tokenizer = load_tokenizer(cfg["model"]["name"])
    rows = read_jsonl(os.path.join(cfg["paths"]["arms_dir"], args.arm + ".jsonl"))
    examples = [encode(tokenizer, row, cfg["model"]["enable_thinking"]) for row in rows]
    kept = [e for e in examples if len(e["input_ids"]) <= tr["max_seq_len"]]
    print(f"{len(kept)} training examples ({len(examples) - len(kept)} dropped for length)")

    # ---- model ----
    # Train in float32: optimizer updates are tiny and get rounded away in 16-bit floats.
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"]).to(device)
    lora = LoraConfig(
        r=tr["lora_r"],
        lora_alpha=tr["lora_alpha"],
        lora_dropout=tr["lora_dropout"],
        target_modules=tr["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)  # freezes the base weights, adds small trainable matrices
    model.print_trainable_parameters()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tr["learning_rate"], weight_decay=0.0)

    # ---- training loop ----
    rng = random.Random(seed)
    order = []            # shuffled example indices; refilled at the start of each epoch
    log = []
    tokens_seen = 0

    for step in range(tr["max_steps"]):
        if len(order) < tr["batch_size"]:
            order = list(range(len(kept)))
            rng.shuffle(order)
        batch = [kept[order.pop()] for _ in range(tr["batch_size"])]
        input_ids, labels, attention = collate(batch, tokenizer.pad_token_id)
        tokens_seen += sum(e["n_solution_tokens"] for e in batch)

        for group in optimizer.param_groups:
            group["lr"] = tr["learning_rate"] * learning_rate_factor(step, tr["warmup_steps"], tr["max_steps"])

        # On a CUDA GPU, run the forward pass in bfloat16 for speed; the weights stay float32.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                labels=labels.to(device),
            ).loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        log.append({"step": step, "loss": loss.item()})
        if step % 25 == 0 or step == tr["max_steps"] - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # ---- save ----
    merged = model.merge_and_unload()  # fold the LoRA update into the base weights: W <- W + B @ A
    merged.save_pretrained(os.path.join(run_dir, "merged"))
    tokenizer.save_pretrained(os.path.join(run_dir, "merged"))
    write_jsonl(os.path.join(run_dir, "loss.jsonl"), log)
    with open(os.path.join(run_dir, "train_meta.json"), "w") as f:
        json.dump({
            "arm": args.arm,
            "seed": seed,
            "examples": len(kept),
            "steps": tr["max_steps"],
            "solution_tokens_seen": tokens_seen,  # differs between arms: shorter solutions = fewer tokens
        }, f, indent=2)
    print("saved", run_dir)


if __name__ == "__main__":
    main()
