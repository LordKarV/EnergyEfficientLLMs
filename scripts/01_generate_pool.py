"""Step 1: sample many candidate solutions for every training problem.

For each problem we draw `samples_per_problem` solutions from the full-precision
model and grade each one. The output file is the "pool" that every training arm
later selects from, so all arms draw from exactly the same candidates.

Run:  python scripts/01_generate_pool.py --config configs/base.yaml
Safe to re-run: problems already in the output file are skipped.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.answers import extract_answer, is_correct
from qtrace.common import append_jsonl, load_config, read_jsonl, set_seed
from qtrace.data import build_prompt, load_problems
from qtrace.generate import generate
from qtrace.models import load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pool_cfg = cfg["pool"]
    out_path = cfg["paths"]["pool"]

    set_seed(pool_cfg["seed"])
    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"]["name"], "bf16")

    problems = load_problems(cfg["data"]["train"], cfg["data"]["train_limit"])

    # Resume support: skip problems that already have samples on disk.
    done = set()
    if os.path.exists(out_path):
        done = {row["qid"] for row in read_jsonl(out_path)}
    todo = [p for p in problems if p["qid"] not in done]
    print(f"{len(problems)} problems, {len(done)} already done, {len(todo)} to go")

    batch_size = pool_cfg["problems_per_batch"]
    n_samples = pool_cfg["samples_per_problem"]

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        prompts = [build_prompt(tokenizer, p["question"], cfg["model"]["enable_thinking"]) for p in batch]
        outputs = generate(
            model, tokenizer, prompts,
            max_new_tokens=pool_cfg["max_new_tokens"],
            samples_per_prompt=n_samples,
            temperature=pool_cfg["temperature"],
            top_p=pool_cfg["top_p"],
            top_k=pool_cfg["top_k"],
        )

        rows = []
        for problem_index, problem in enumerate(batch):
            for sample_index in range(n_samples):
                out = outputs[problem_index * n_samples + sample_index]
                predicted, used_marker = extract_answer(out["text"])
                rows.append({
                    "qid": problem["qid"],
                    "sample": sample_index,
                    "question": problem["question"],
                    "gold": problem["gold"],
                    "text": out["text"],
                    "n_tokens": out["n_tokens"],
                    "finished": out["finished"],
                    "predicted": predicted,
                    "used_marker": used_marker,
                    # A solution only counts if it finished AND used the answer format AND is right.
                    "correct": out["finished"] and used_marker and is_correct(predicted, problem["gold"]),
                })
        append_jsonl(out_path, rows)

        n_correct = sum(row["correct"] for row in rows)
        print(f"problems {start + len(batch)}/{len(todo)}  correct in this batch: {n_correct}/{len(rows)}")


if __name__ == "__main__":
    main()
