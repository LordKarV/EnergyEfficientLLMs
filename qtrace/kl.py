"""Chunked teacher-forced disagreement between a full-precision model and a quantized copy.

Used by P0d, the `stable`/`qtrace` arms and the Stage 2 KL term (docs/04, P0d and "Selection and
training rules for P1"). Only model-written tokens are scored: the logit at position i predicts
token i+1, so token t is scored from logits at t-1.

Memory: a full logits tensor is n_tokens x 248,320 floats (about 60 GiB at 64K), so each model runs
one forward pass, keeps its final hidden states at the scored positions only, and the output heads
are applied in chunks. Each model uses its own head, because a quantizer may change it.
"""

import math

import torch


def logit_index(model_mask):
    """Positions of the logits that predict the model-written tokens (t - 1 for each t >= 1)."""
    t = torch.nonzero(model_mask).squeeze(1)
    return t[t >= 1] - 1


@torch.no_grad()
def hidden_at(model, input_ids, idx):
    """Final (normed) hidden states at positions idx, from one forward pass."""
    hidden = model.model(input_ids.to(model.device)).last_hidden_state[0]
    return hidden[idx.to(hidden.device)]


@torch.no_grad()
def compare(head_p, h_p, head_q, h_q, targets, chunk=384):
    """Per-position KL(P||Q), top-1 flip, entropy of P and NLL of the actual token under P and Q."""
    out = {k: [] for k in ("kl", "flip", "entropy_p", "nll_p", "nll_q")}
    for s in range(0, len(targets), chunk):
        lp = torch.log_softmax(head_p(h_p[s:s + chunk]).float(), dim=-1)
        lq = torch.log_softmax(head_q(h_q[s:s + chunk].to(h_p.device)).float().to(lp.device), dim=-1)
        y = targets[s:s + chunk].to(lp.device).unsqueeze(1)
        p = lp.exp()
        out["kl"].append((p * (lp - lq)).sum(-1))
        out["flip"].append((lp.argmax(-1) != lq.argmax(-1)).float())
        out["entropy_p"].append(-(p * lp).sum(-1))
        out["nll_p"].append(-lp.gather(1, y).squeeze(1))
        out["nll_q"].append(-lq.gather(1, y).squeeze(1))
    return {k: torch.cat(v).cpu() if v else torch.zeros(0) for k, v in out.items()}


def decision_mask(idx, entropy, tool_call_positions, command_starts, command_tokens=8, top_frac=0.10):
    """P0d decision positions among the scored tokens (docs/04, P0d primary score).

    A scored token (position idx + 1) is a decision point if it is a `<tool_call>` token, one of the
    first `command_tokens` tokens of a command value, or among the ceil(top_frac * n) highest-entropy
    scored tokens (ties broken by position). Positions come from the episode store; tokens outside
    model-written spans are never scored, so they can never be decision points.
    """
    pos = (idx + 1).tolist()
    where = {p: i for i, p in enumerate(pos)}
    mask = torch.zeros(len(pos), dtype=torch.bool)
    for p in tool_call_positions:
        if p in where:
            mask[where[p]] = True
    for start in command_starts:
        for p in range(start, start + command_tokens):
            if p in where:
                mask[where[p]] = True
    k = math.ceil(top_frac * len(pos))
    if k:
        order = sorted(range(len(pos)), key=lambda i: (-float(entropy[i]), pos[i]))
        mask[order[:k]] = True
    return mask


def summarize(stats, decision):
    """Whole-run and decision-point scores. kl_sum is the RL term (it can't be lowered by padding)."""
    kl = stats["kl"]
    n = len(kl)
    return {
        "n_scored": n,
        "n_decision": int(decision.sum()),
        "kl_mean": float(kl.mean()) if n else 0.0,
        "kl_sum": float(kl.sum()),
        "kl_decision": float(kl[decision].mean()) if decision.any() else 0.0,
        "flip_rate": float(stats["flip"].mean()) if n else 0.0,
        "entropy_mean": float(stats["entropy_p"].mean()) if n else 0.0,
        "nll_p_mean": float(stats["nll_p"].mean()) if n else 0.0,
        "nll_q_mean": float(stats["nll_q"].mean()) if n else 0.0,
    }


def score_trajectory(model_p, model_q, input_ids, model_mask, tool_call_positions=(), command_starts=(), chunk=384):
    """Score one stored episode. model_p is full precision, model_q the quantized copy."""
    idx = logit_index(model_mask)
    targets = input_ids[0, (idx + 1).to(input_ids.device)]
    stats = compare(model_p.lm_head, hidden_at(model_p, input_ids, idx),
                    model_q.lm_head, hidden_at(model_q, input_ids, idx), targets, chunk=chunk)
    dm = decision_mask(idx, stats["entropy_p"], tool_call_positions, command_starts)
    return summarize(stats, dm)
