"""Step 5: evaluate one model at several precisions under identical conditions.

For each precision we load the model, answer every test problem with greedy
decoding, and record accuracy, output length, time, peak GPU memory and GPU energy.

Run:  python scripts/05_evaluate.py --config configs/base.yaml \
          --model outputs/runs/qtrace_seed0/merged --tag qtrace_seed0
      python scripts/05_evaluate.py --config configs/base.yaml \
          --model Qwen/Qwen3-0.6B --tag base

IMPORTANT: for "rtnN" precisions only accuracy and token counts are meaningful.
The summary file marks this with "efficiency_is_real": false.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.answers import extract_answer, is_correct
from qtrace.common import load_config, write_jsonl
from qtrace.data import build_prompt, load_problems
from qtrace.energy import EnergyMeter
from qtrace.generate import generate
from qtrace.models import efficiency_is_real, load_model, load_tokenizer


def evaluate(model, tokenizer, problems, cfg):
    """Answer every problem and return (per-problem rows, energy meter)."""
    ev = cfg["eval"]
    rows = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with EnergyMeter() as meter:
        for start in range(0, len(problems), ev["batch_size"]):
            batch = problems[start:start + ev["batch_size"]]
            prompts = [build_prompt(tokenizer, p["question"], cfg["model"]["enable_thinking"]) for p in batch]
            outputs = generate(model, tokenizer, prompts, max_new_tokens=ev["max_new_tokens"])
            for problem, out in zip(batch, outputs):
                predicted, used_marker = extract_answer(out["text"])
                rows.append({
                    "qid": problem["qid"],
                    "gold": problem["gold"],
                    "predicted": predicted,
                    "correct": is_correct(predicted, problem["gold"]),
                    "used_marker": used_marker,
                    "n_tokens": out["n_tokens"],
                    "finished": out["finished"],
                    "text": out["text"],
                })
    return rows, meter


def summarize(rows, meter, tag, precision):
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    n_tokens = sum(r["n_tokens"] for r in rows)
    summary = {
        "tag": tag,
        "precision": precision,
        "problems": n,
        "accuracy": n_correct / n,
        "mean_tokens": n_tokens / n,
        "unfinished_rate": sum(not r["finished"] for r in rows) / n,
        "efficiency_is_real": efficiency_is_real(precision),
        "seconds": meter.seconds,
        "tokens_per_second": n_tokens / meter.seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "gpu_joules": meter.joules,
    }
    if meter.joules is not None:
        summary["joules_per_problem"] = meter.joules / n
        summary["joules_per_token"] = meter.joules / max(1, n_tokens)
        summary["joules_per_correct"] = meter.joules / max(1, n_correct)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", required=True, help="HF hub id or folder with a merged model")
    parser.add_argument("--tag", required=True, help="short name for this model in result files")
    parser.add_argument("--precisions", nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    precisions = args.precisions or cfg["eval"]["precisions"]

    tokenizer = load_tokenizer(args.model)
    problems = load_problems(cfg["data"]["test"], cfg["data"]["test_limit"])

    for precision in precisions:
        model = load_model(args.model, precision)
        rows, meter = evaluate(model, tokenizer, problems, cfg)
        summary = summarize(rows, meter, args.tag, precision)

        stem = os.path.join(cfg["paths"]["eval_dir"], f"{args.tag}__{precision}")
        write_jsonl(stem + ".jsonl", rows)
        with open(stem + ".json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary))

        del model  # free GPU memory before loading the next precision
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
