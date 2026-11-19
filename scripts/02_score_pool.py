"""Step 2: measure how sensitive each candidate solution is to quantization.

For one solution we feed the exact same tokens (prompt + solution) through two
models: the full-precision model and a quantized "probe" copy. Neither model
generates anything; we only read their next-token predictions at every position
of the solution ("teacher forcing"). If the two models predict nearly the same
distribution at every step, quantization does not disturb this solution: it is
quantization-stable. Large disagreement means the solution passes through
positions that quantization pushes off course.

Scores written per solution (all averaged over the solution's tokens unless noted):
  kl_mean    KL(full || quantized). Our main stability score. Lower = more stable.
  kl_max     the single worst position
  flip_rate  fraction of positions where the two models' top-1 token differs
  nll_fp     full-precision negative log-likelihood of the solution (how "natural" it is)
  nll_q      the same under the quantized probe
  entropy_fp full-precision next-token entropy (a control: needs no quantized model)

Run:  python scripts/02_score_pool.py --config configs/base.yaml
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl, write_jsonl
from qtrace.data import build_prompt
from qtrace.models import load_model, load_tokenizer


@torch.no_grad()
def completion_log_probs(model, input_ids, n_prompt):
    """Log-probabilities over the vocabulary at each position that predicts a solution token.

    The logits at position i are the model's prediction for token i+1. The solution
    occupies positions n_prompt .. end, so the predictions for it sit one step earlier.
    """
    logits = model(input_ids.to(model.device)).logits[0]
    logits = logits[n_prompt - 1:-1].float()  # float32 for a stable softmax
    return torch.log_softmax(logits, dim=-1)


def score_one(fp_model, q_model, tokenizer, prompt, solution):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    solution_ids = tokenizer(solution + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + solution_ids])
    n_prompt = len(prompt_ids)

    log_p = completion_log_probs(fp_model, input_ids, n_prompt)                # full precision
    log_q = completion_log_probs(q_model, input_ids, n_prompt).to(log_p.device)  # quantized probe
    p = log_p.exp()

    targets = torch.tensor(solution_ids, device=log_p.device).unsqueeze(1)

    kl = (p * (log_p - log_q)).sum(dim=-1)            # one KL value per position
    entropy = -(p * log_p).sum(dim=-1)
    nll_fp = -log_p.gather(1, targets).squeeze(1)     # -log prob of the actual next token
    nll_q = -log_q.gather(1, targets).squeeze(1)
    flips = (log_p.argmax(dim=-1) != log_q.argmax(dim=-1)).float()

    return {
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "flip_rate": flips.mean().item(),
        "nll_fp": nll_fp.mean().item(),
        "nll_q": nll_q.mean().item(),
        "entropy_fp": entropy.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = load_tokenizer(cfg["model"]["name"])
    fp_model = load_model(cfg["model"]["name"], "bf16")
    q_model = load_model(cfg["model"]["name"], cfg["stability"]["probe_precision"])

    rows = read_jsonl(cfg["paths"]["pool"])
    if cfg["stability"]["require_correct"]:
        rows = [row for row in rows if row["correct"]]
    print(f"scoring {len(rows)} solutions with probe {cfg['stability']['probe_precision']}")

    scored = []
    for index, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row["question"], cfg["model"]["enable_thinking"])
        row.update(score_one(fp_model, q_model, tokenizer, prompt, row["text"]))
        scored.append(row)
        if (index + 1) % 500 == 0:
            print(f"{index + 1}/{len(rows)}")

    write_jsonl(cfg["paths"]["scored"], scored)
    print("wrote", cfg["paths"]["scored"])


if __name__ == "__main__":
    main()
