"""Selection rules: given several correct solutions to one problem, which one do we train on?

Each "arm" is one rule. Every arm picks exactly one solution per problem from the
same candidates, so the resulting training sets have the same problems and the
same size and differ only in WHICH solution was chosen.
"""

import random
from collections import defaultdict

ARMS = [
    "random",          # baseline 1: any correct solution
    "shortest",        # baseline 2: concise-reasoning self-training (Munkhbat et al., 2025)
    "stable",          # most quantization-stable, ignoring length
    "qtrace",          # ours: short AND stable
    "low_entropy",     # control: lowest full-precision entropy (no quantized model needed)
    "longest",         # control: opposite of "shortest"
    "short_stable",    # 2x2 design that separates length from stability:
    "short_unstable",  #   split candidates at the median length, then take the most
    "long_stable",     #   or least stable solution inside the short half or long half
    "long_unstable",
]


def group_candidates(rows, min_candidates):
    """Group scored solutions by problem, drop duplicate texts, keep problems with enough candidates."""
    by_problem = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["qid"], row["text"])
        if key in seen:
            continue
        seen.add(key)
        by_problem[row["qid"]].append(row)
    return {qid: cands for qid, cands in by_problem.items() if len(cands) >= min_candidates}


def ranks(candidates, field):
    """Rank candidates by a field. Returns one number per candidate in [0, 1]; 0 = smallest value."""
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][field])
    result = [0.0] * len(candidates)
    for position, index in enumerate(order):
        result[index] = position / max(1, len(candidates) - 1)
    return result


def pick(candidates, arm, alpha, rng):
    """Apply one arm's rule to one problem's candidates and return the chosen solution."""
    if arm == "random":
        return rng.choice(candidates)
    if arm == "shortest":
        return min(candidates, key=lambda c: c["n_tokens"])
    if arm == "longest":
        return max(candidates, key=lambda c: c["n_tokens"])
    if arm == "stable":
        return min(candidates, key=lambda c: c["kl_mean"])
    if arm == "low_entropy":
        return min(candidates, key=lambda c: c["entropy_fp"])
    if arm == "qtrace":
        # Ranks (not raw values) so that length and KL are on the same 0..1 scale
        # and hard problems with long solutions are not penalized against easy ones.
        length_rank = ranks(candidates, "n_tokens")
        stability_rank = ranks(candidates, "kl_mean")
        combined = [alpha * l + (1 - alpha) * s for l, s in zip(length_rank, stability_rank)]
        return candidates[combined.index(min(combined))]

    # The four 2x2 arms.
    by_length = sorted(candidates, key=lambda c: c["n_tokens"])
    half = len(by_length) // 2
    short_half, long_half = by_length[:half], by_length[half:]
    pool = short_half if arm.startswith("short") else long_half
    if arm.endswith("unstable"):
        return max(pool, key=lambda c: c["kl_mean"])
    return min(pool, key=lambda c: c["kl_mean"])


def build_arm(by_problem, arm, alpha=0.5, seed=0):
    """Return the training set for one arm: a list with one chosen solution per problem."""
    rng = random.Random(seed)
    return [pick(by_problem[qid], arm, alpha, rng) for qid in sorted(by_problem)]
