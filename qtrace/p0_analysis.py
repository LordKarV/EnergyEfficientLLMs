"""P0 gate statistics (docs/04: P0b report, P0a primary test, P0c breakage).

Pure functions over per-attempt records in the format agreed in issue #1 (one dict per attempt with
task_id, list, repo, profile, runtime, resolved and tokens{output, input_uncached, input_cached}).
Bootstraps resample tasks, keeping every attempt and profile of a task together, and are seeded.
"""

from collections import defaultdict

import numpy as np

BINS = [("0", 0, 0), ("1-3", 1, 3), ("4-8", 4, 8), ("9-16", 9, 16)]


def processed_tokens(rec):
    """Output tokens plus every input token per turn, cache hits and misses (04, P0a)."""
    t = rec["tokens"]
    return t["output"] + t["input_uncached"] + t["input_cached"]


def _per_task(rows, profile, lst, runtime):
    """task -> [processed tokens summed, resolved count, attempts]."""
    out = defaultdict(lambda: [0.0, 0, 0])
    for r in rows:
        if r["profile"] == profile and r["list"] == lst and r["runtime"] == runtime:
            acc = out[r["task_id"]]
            acc[0] += processed_tokens(r)
            acc[1] += int(bool(r["resolved"]))
            acc[2] += 1
    return out


def _ratio(pq, rq, pb, rb):
    if rq == 0 or rb == 0 or pb == 0:
        return float("inf")
    return (pq / rq) / (pb / rb)


def p0a_primary(rows, quantized="Q4_K_M", base="BF16", lst="heldout100", runtime="llama.cpp",
                n_boot=5000, seed=0, threshold=1.15):
    """Pooled R = (processed tokens / resolved attempts) at `quantized` over the same at `base`."""
    q, b = _per_task(rows, quantized, lst, runtime), _per_task(rows, base, lst, runtime)
    tasks = sorted(set(q) & set(b))
    arr = np.array([[q[t][0], q[t][1], b[t][0], b[t][1]] for t in tasks], dtype=float)
    R = _ratio(*arr.sum(axis=0))
    rng = np.random.default_rng(seed)
    boots = np.array([_ratio(*arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)) for _ in range(n_boot)])
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    if R >= threshold and lo > 1:
        verdict = "pass"
    elif hi < threshold:
        verdict = "fail"
    else:
        verdict = "inconclusive"
    return {"R": R, "ci95": [lo, hi], "verdict": verdict, "n_tasks": len(tasks), "n_boot": n_boot, "seed": seed}


def p0a_sensitivity(rows, quantized="Q4_K_M", base="BF16", lst="heldout100", runtime="llama.cpp"):
    """Length factor (geometric mean of per-task ratios, tasks resolved under both) and completion factor."""
    q, b = _per_task(rows, quantized, lst, runtime), _per_task(rows, base, lst, runtime)
    both = [t for t in set(q) & set(b) if q[t][1] and b[t][1]]
    logs = [np.log((q[t][0] / q[t][1]) / (b[t][0] / b[t][1])) for t in both]
    rate_q = sum(v[1] for v in q.values()) / max(sum(v[2] for v in q.values()), 1)
    rate_b = sum(v[1] for v in b.values()) / max(sum(v[2] for v in b.values()), 1)
    return {"length_factor_geomean": float(np.exp(np.mean(logs))) if logs else None,
            "n_tasks_resolved_both": len(both),
            "resolve_rate_ratio": rate_q / rate_b if rate_b else None}


def p0b_report(rows, inrepo_size, pool_total=3887, heldout_total=691, min_candidates=4, runtime="vllm",
               profile="BF16"):
    """Pool size, RL-usable share and the three P0b pass criteria."""
    def successes(lst):
        acc = defaultdict(lambda: [0, 0, None])
        for r in rows:
            if r["list"] == lst and r["runtime"] == runtime and r["profile"] == profile:
                a = acc[r["task_id"]]
                a[0] += int(bool(r["resolved"]))
                a[1] += 1
                a[2] = r.get("repo")
        return acc

    pool, held = successes("pool200"), successes("heldout100")
    n = len(pool)
    ge1 = sum(1 for s, _, _ in pool.values() if s >= 1)
    ge4 = sum(1 for s, _, _ in pool.values() if s >= min_candidates)
    mixed = sum(1 for s, k, _ in pool.values() if 0 < s < k)
    by_repo = defaultdict(lambda: {"n_tasks": 0, "n_ge4": 0})
    for s, _, repo in pool.values():
        by_repo[repo]["n_tasks"] += 1
        by_repo[repo]["n_ge4"] += int(s >= min_candidates)
    share = ge4 / ge1 if ge1 else 0.0
    projected = (ge4 / n) * (pool_total - inrepo_size) if n else 0.0
    held_pass1 = float(np.mean([s / k for s, k, _ in held.values()])) if held else 0.0
    return {
        "pool": {"n_tasks": n, "n_ge1": ge1, "n_ge4": ge4, "by_repo": dict(by_repo),
                 "success_counts": sorted(s for s, _, _ in pool.values())},
        "criteria": {"share_ge4_among_solvable": share},
        "projected_pool": projected,
        "heldout_mean_pass1": held_pass1,
        "heldout_expected_resolves": held_pass1 * heldout_total,
        "rl_share_mixed": mixed / n if n else 0.0,
        "pass": {"c1": share > 0.5, "c2": projected >= 500, "c3": held_pass1 * heldout_total >= 100},
        "mde_note": "simulated minimum detectable effect not computed here; added with the P2 analysis",
    }


def _bin(count):
    for name, lo, hi in BINS:
        if lo <= count <= hi:
            return name
    raise ValueError(f"success count {count} outside 0-16")


def p0c_breakage(rows, vllm_counts, profile, base="BF16", lst="pool200", runtime="llama.cpp", n_boot=5000, seed=0):
    """L_b = sum over tasks in bin b of (BF16 pass rate - quantized pass rate), same runtime.

    vllm_counts: task -> BF16 successes out of 16 from P0b, used only to bin tasks.
    """
    q, b = _per_task(rows, profile, lst, runtime), _per_task(rows, base, lst, runtime)
    tasks = sorted(t for t in set(q) & set(b) if t in vllm_counts)
    diff = np.array([b[t][1] / b[t][2] - q[t][1] / q[t][2] for t in tasks])
    bins = np.array([_bin(vllm_counts[t]) for t in tasks])
    four_plus = np.isin(bins, ["4-8", "9-16"])
    L_by_bin = {name: float(diff[bins == name].sum()) for name, _, _ in BINS}
    total = float(diff.sum())
    share = float(diff[four_plus].sum() / total) if total > 0 else None
    rng = np.random.default_rng(seed)
    tot_b, share_b = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(tasks), len(tasks))
        t_sum = diff[i].sum()
        tot_b.append(t_sum)
        if t_sum > 0:
            share_b.append(diff[i][four_plus[i]].sum() / t_sum)
    lo_total = float(np.percentile(tot_b, 2.5))
    share_ci = [float(x) for x in np.percentile(share_b, [2.5, 97.5])] if share_b else None
    if lo_total <= 0:
        verdict = "no detectable breakage"
    elif share is not None and share >= 0.4:
        verdict = "pass"
    else:
        verdict = "below 0.4: use the P0b fallbacks or claim generalization only"
    return {"profile": profile, "L_by_bin": L_by_bin, "L_total": total, "L_total_lo95": lo_total,
            "share_4plus": share, "share_4plus_ci95": share_ci, "verdict": verdict, "n_tasks": len(tasks)}
