"""P0d: the score-choice rule, the replay pairs, the continuation DiD test and its power (docs/04, P0d).

Pure functions over the formats agreed in issue #1: attempt records, outputs/p0d/scores.jsonl rows
(one per attempt and probe, holding every key of qtrace.kl.score_stats), est_joules.jsonl rows and
continuation results. scripts/p0d_analyze.py does the file I/O.

Within-task statistics are Spearman correlations with average ranks for ties, computed per task and
summarized as the mean over tasks (the rule uses the mean) with the median alongside. A task enters
a statistic only with at least MIN_RUNS runs that have both values; a task where either variable is
constant has no rho and is counted, not scored as 0.

Spearman-Brown is applied to the mean (and the median) of the per-task even-vs-odd rhos, not per
task before averaging. With 3 runs a per-task rho is one of -1, -0.5, 0.5 and 1, and 2r / (1 + r) maps
-0.5 to -2 and -1 to minus infinity, so correcting per task would let a few tasks dominate the mean.
The correction is monotone, so the median is the same either way.
"""

import math
from collections import defaultdict

import numpy as np

CANDIDATES = ("kl_decision", "kl_command", "kl_mean")  # frozen order, docs/04 (2026-11-20)
PROBE, AGREEMENT_PROBE = "Q3_K_M", "Q4_K_M"
HELD_OUT_LISTS = ("heldout100", "verified100", "inrepo_test")  # never part of any pool (docs/04, Task sets)
CONFOUNDS = ("turns", "output_tokens", "est_joules_q4", "entropy_bf16")
COVARIATES = ("entropy", "turns", "malformed")
MAX_ABS_RHO, MIN_SPLIT_HALF, MIN_AGREEMENT, MIN_RUNS = 0.8, 0.5, 0.5, 3
MIN_EFFECT = 0.10
UPPER_BOUND_DEFINITION = ("95th percentile (inverted_cdf order statistic) of the task-bootstrap means: the "
                          "one-sided 95% interval is (-inf, upper]")
COVARIATE_CONVENTION = ("each covariate is the low-score run minus the high-score run of the task's pair, in raw "
                        "units (BF16 entropy_mean in nats, turns, malformed calls), uncentered, so the intercept "
                        "is the DiD of a pair that does not differ in any covariate")


def flagged(rec):
    """The grader's test-file tripwire (docs/04, "Test-file tripwire"); absent counts as false."""
    return bool((rec.get("grading") or {}).get("test_files_touched"))


def average_ranks(x):
    """1-based ranks, tied values sharing the mean of their positions."""
    _, inverse, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(x, y):
    """Spearman's rho with average ranks; None when either variable is constant (no rho exists)."""
    rx, ry = average_ranks(x), average_ranks(y)
    if len(rx) < 2 or rx.min() == rx.max() or ry.min() == ry.max():
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_brown(r):
    """Full-length reliability 2r / (1 + r) from a half-length correlation r; None at r = -1."""
    return None if r is None or r <= -1 else 2 * r / (1 + r)


def within_task(groups, min_runs=MIN_RUNS):
    """Per-task Spearman over {task: [(x, y), ...]}, summarized over tasks.

    Pairs with a None on either side are dropped and counted (n_runs_missing_value). Tasks left with
    fewer than min_runs pairs are counted in n_tasks_too_few_runs, constant ones in n_tasks_constant.
    """
    rhos, constant, too_few, missing = [], 0, 0, 0
    for task in sorted(groups):
        pts = [(x, y) for x, y in groups[task] if x is not None and y is not None]
        missing += len(groups[task]) - len(pts)
        if len(pts) < min_runs:
            too_few += 1
            continue
        r = spearman(*zip(*pts))
        if r is None:
            constant += 1
        else:
            rhos.append(r)
    return {"mean": float(np.mean(rhos)) if rhos else None, "median": float(np.median(rhos)) if rhos else None,
            "n_tasks": len(rhos), "n_tasks_constant": constant, "n_tasks_too_few_runs": too_few,
            "n_runs_missing_value": missing}


def _number(value, what, optional=True):
    """A finite number, or None if optional; anything else (NaN, inf, a string) is a broken input."""
    if (value is None and optional) or \
            (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
        return value
    raise ValueError(f"{what}: expected a finite number{' or null' if optional else ''}, got {value!r}")


def _field(rec, key, what):
    if key not in rec:
        raise ValueError(f"{what}: missing field {key!r}")
    return rec[key]


def _scores(row, what):
    return {c: _number(_field(row, c, what), f"{what} {c}") for c in CANDIDATES}


def join_runs(attempts, scores, est_joules=None):
    """One record per scored pool run, joining attempt fields, both probes' scores and estimated joules.

    Runs are the attempts with a Q3_K_M score row. Each must be a resolved BF16 attempt on a pool list;
    anything else (an unknown attempt, another profile, a failure, a held-out list, a duplicate row, a
    non-finite number) raises ValueError, since it means the scorer read the wrong files. A run the
    test-file tripwire flagged can never be a selection candidate and is excluded. info's
    flagged_excluded counts, per list, the resolved BF16 attempts it flagged, from the attempt records,
    because the scorer writes no rows for them. est_joules: est_joules.jsonl rows, or None if P0a's
    file does not exist yet. Runs without a Q4_K_M row or an est_joules value are counted in info, and
    candidate_stats does not run the check they feed. Returns (runs, info).
    """
    by_id, flagged_by_list = {}, defaultdict(int)
    for a in attempts:
        aid = _field(a, "attempt_id", "attempt")
        if aid in by_id:
            raise ValueError(f"duplicate attempt_id {aid!r} in the attempts files")
        by_id[aid] = a
        what = f"attempt {aid}"
        if _field(a, "profile", what) == "BF16" and _field(a, "resolved", what) and flagged(a) and \
                _field(a, "list", what) not in HELD_OUT_LISTS:
            flagged_by_list[a["list"]] += 1
    rows = {}
    for s in scores:
        key = (_field(s, "attempt_id", "score row"), _field(s, "probe", "score row"))
        if key[1] not in (PROBE, AGREEMENT_PROBE):
            raise ValueError(f"score row {key}: unknown probe")
        if key in rows:
            raise ValueError(f"duplicate score row for {key}")
        rows[key] = s
    joules = {}
    for j in est_joules or ():
        aid = _field(j, "attempt_id", "est_joules row")
        if aid in joules:
            raise ValueError(f"duplicate est_joules row for {aid!r}")
        joules[aid] = _number(_field(j, "est_joules_q4", f"est_joules {aid}"), f"est_joules {aid}")

    runs = []
    for (aid, probe), s in sorted(rows.items()):
        if probe != PROBE:
            continue
        what = f"attempt {aid}"
        if aid not in by_id:
            raise ValueError(f"{what}: scored but not in the attempts files")
        a = by_id[aid]
        if _field(a, "list", what) in HELD_OUT_LISTS:
            raise ValueError(f"{what}: list {a['list']!r} is held-out and never enters a pool")
        if _field(a, "profile", what) != "BF16" or not _field(a, "resolved", what):
            raise ValueError(f"{what}: pool runs are resolved BF16 attempts (profile {a['profile']!r}, "
                             f"resolved {a['resolved']!r})")
        if _field(s, "task_id", what) != _field(a, "task_id", what):
            raise ValueError(f"{what}: score row task {s['task_id']!r} differs from the attempt's {a['task_id']!r}")
        if flagged(a):
            continue
        q4 = rows.get((aid, AGREEMENT_PROBE))
        halves = s.get("halves") or {}
        score = _scores(s, what)
        entropy = _number(_field(s, "entropy_mean", what), f"{what} entropy_mean")
        if entropy is None and any(v is not None for v in score.values()):
            raise ValueError(f"{what}: entropy_mean is null but the row has KL scores")  # kl.py nulls both together
        runs.append({
            "attempt_id": aid, "task_id": a["task_id"],
            "turns": _number(_field(a, "turns", what), f"{what} turns", optional=False),
            "output_tokens": _number(_field(_field(a, "tokens", what), "output", what), f"{what} tokens.output",
                                     optional=False),
            "malformed": _number(sum(_field(a, "malformed", what).values()), f"{what} malformed", optional=False),
            "est_joules_q4": joules.get(aid),
            "entropy_bf16": entropy,
            "score": score,
            "q4": _scores(q4, f"{what} {AGREEMENT_PROBE}") if q4 else None,
            "even": _scores(halves["even"], f"{what} even half") if "even" in halves else None,
            "odd": _scores(halves["odd"], f"{what} odd half") if "odd" in halves else None,
        })
    q3_ids = {aid for aid, p in rows if p == PROBE}
    info = {"n_runs": len(runs), "n_tasks": len({r["task_id"] for r in runs}),
            "flagged_excluded": dict(sorted(flagged_by_list.items())),
            "q4_rows_without_q3": sum(1 for aid, p in rows if p == AGREEMENT_PROBE and aid not in q3_ids),
            "n_runs_without_q4": sum(1 for r in runs if r["q4"] is None),
            "est_joules_given": est_joules is not None,
            "n_runs_without_est_joules": sum(1 for r in runs if r["est_joules_q4"] is None)}
    return runs, info


def _grouped(runs, pair):
    out = defaultdict(list)
    for r in runs:
        out[r["task_id"]].append(pair(r))
    return out


def _check(value, ok):
    """True or False, or None when the statistic could not be computed (not run)."""
    return None if value is None else bool(ok(value))


def candidate_stats(runs, name, have_joules=True, min_runs=MIN_RUNS):
    """Every number docs/04 asks for one candidate, the six checks and whether it passes.

    The joules and probe-agreement checks do not run while any run lacks its est_joules value or its
    Q4_K_M row (P0a or the scorer still running): on the few tasks that happen to be complete they
    could pass a candidate the full pool fails.
    """
    no_joules = sum(1 for r in runs if r["est_joules_q4"] is None)
    no_q4 = sum(1 for r in runs if r["q4"] is None)
    against = {}
    for k in CONFOUNDS:
        if k == "est_joules_q4" and not have_joules:
            against[k] = {"not_run": "no est_joules.jsonl given (P0a energy model not fitted yet)"}
        elif k == "est_joules_q4" and no_joules:
            against[k] = {"not_run": f"{no_joules} of {len(runs)} runs have no est_joules_q4"}
        else:
            against[k] = within_task(_grouped(runs, lambda r, k=k: (r["score"][name], r[k])), min_runs)
    agreement = ({"not_run": f"{no_q4} of {len(runs)} runs have no {AGREEMENT_PROBE} row"} if no_q4 else
                 within_task(_grouped(runs, lambda r: (r["score"][name], r["q4"][name])), min_runs))
    rho = within_task(_grouped(runs, lambda r: (r["even"] and r["even"][name], r["odd"] and r["odd"][name])),
                      min_runs)
    split = {"rho": rho, "spearman_brown_mean": spearman_brown(rho["mean"]),
             "spearman_brown_median": spearman_brown(rho["median"])}
    checks = {k: _check(against[k].get("mean"), lambda m: abs(m) <= MAX_ABS_RHO) for k in CONFOUNDS}
    # SB(r) >= 0.5 exactly when r >= 1/3; testing the raw mean keeps r = -1 a failure, not "not run".
    checks["split_half"] = _check(rho["mean"], lambda m: m >= MIN_SPLIT_HALF / (2 - MIN_SPLIT_HALF))
    checks["probe_agreement"] = _check(agreement.get("mean"), lambda m: m >= MIN_AGREEMENT)
    failed = [k for k, v in checks.items() if v is False]
    not_run = [k for k, v in checks.items() if v is None]
    return {"against": against, "probe_agreement": agreement, "split_half": split, "checks": checks,
            "failed": failed, "not_run": not_run, "pass": not failed and not not_run}


def choose_score(runs, have_joules=True, min_runs=MIN_RUNS):
    """docs/04's rule: the first candidate, in the frozen order, that passes every check.

    A candidate that fails a check is skipped. One that fails none but has a check that could not run
    (an est_joules value or a Q4_K_M row missing for any run, no task with enough runs) stops the walk:
    chosen stays None and decided is False, because a later candidate may only be chosen once the
    earlier one has failed.
    decided with chosen None answers H3 negatively and drops the stability arms before P1.
    """
    candidates = {c: candidate_stats(runs, c, have_joules, min_runs) for c in CANDIDATES}
    chosen, decided = None, True
    verdict = "none passed: H3 answered negatively, stability arms dropped before P1"
    for c in CANDIDATES:
        if candidates[c]["pass"]:
            chosen, verdict = c, f"chosen: {c}"
            break
        if not candidates[c]["failed"]:
            decided = False
            verdict = f"incomplete: {c} fails no check but {', '.join(candidates[c]['not_run'])} did not run"
            break
    return {"chosen": chosen, "decided": decided, "verdict": verdict, "order": list(CANDIDATES), "probe": PROBE,
            "agreement_probe": AGREEMENT_PROBE,
            "rule": {"max_abs_rho": MAX_ABS_RHO, "min_split_half": MIN_SPLIT_HALF, "min_probe_agreement": MIN_AGREEMENT,
                     "min_runs_per_task": min_runs, "statistic": "mean over tasks of the within-task Spearman",
                     "spearman_brown": "applied to the mean (and median) of the per-task even-vs-odd rhos"},
            "candidates": candidates}


def select_pairs(runs, score):
    """The lowest- and highest-scoring run of every task with at least 2 scored runs, under `score`.

    Runs are ordered by (score, attempt_id): low is the first, high the last, so ties go to the
    smaller attempt_id for low and the larger for high, and the two are always different runs. diff
    holds the DiD covariates (COVARIATE_CONVENTION).
    """
    by_task, unscored = defaultdict(list), 0
    for r in runs:
        if r["score"][score] is None:
            unscored += 1
        else:
            by_task[r["task_id"]].append(r)
    pairs = []
    for task in sorted(by_task):
        rs = sorted(by_task[task], key=lambda r: (r["score"][score], r["attempt_id"]))
        if len(rs) < 2:
            continue
        lo, hi = rs[0], rs[-1]
        pairs.append({"task_id": task, "low": lo["attempt_id"], "high": hi["attempt_id"],
                      "low_score": lo["score"][score], "high_score": hi["score"][score],
                      "spread": hi["score"][score] - lo["score"][score], "n_runs": len(rs),
                      "diff": {"entropy": lo["entropy_bf16"] - hi["entropy_bf16"], "turns": lo["turns"] - hi["turns"],
                               "malformed": lo["malformed"] - hi["malformed"]}})
    spreads = [p["spread"] for p in pairs]
    summary = ({"mean": float(np.mean(spreads)), "median": float(np.median(spreads)), "min": min(spreads),
                "max": max(spreads)} if spreads else None)
    return {"score": score, "probe": PROBE, "n_tasks": len(pairs), "fewer_than_50": len(pairs) < 50,
            "n_tasks_one_run": sum(1 for rs in by_task.values() if len(rs) < 2), "n_runs_unscored": unscored,
            "n_tasks_zero_spread": sum(1 for s in spreads if s == 0), "spread": summary,
            "tie_break": "order by (score, attempt_id); low = first, high = last",
            "covariates": COVARIATE_CONVENTION, "tasks": pairs}


def _boot_weights(n, n_boot, rng):
    """Task-bootstrap weights: row b counts how often each task is drawn in draw b, divided by n."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n


def _continuation_cells(rows, profile):
    """(task, prefix, profile) -> [successes, attempts], tasks with an invalid replay, rows ignored."""
    cells, tasks, invalid, ignored = defaultdict(lambda: [0, 0]), set(), set(), defaultdict(int)
    for r in rows:
        what = f"continuation row {r!r}"
        task, prefix, prof = (_field(r, k, what) for k in ("task_id", "prefix", "profile"))
        if prefix not in ("low", "high"):
            raise ValueError(f"{what}: prefix must be 'low' or 'high'")
        if prof not in ("BF16", profile):
            ignored[prof] += 1
            continue
        s, n, valid = (_field(r, k, what) for k in ("successes", "attempts", "replay_valid"))
        if not (isinstance(s, int) and isinstance(n, int) and 0 <= s <= n) or not isinstance(valid, bool):
            raise ValueError(f"{what}: need integer 0 <= successes <= attempts and a boolean replay_valid")
        tasks.add(task)
        if not valid:
            invalid.add(task)
        cells[task, prefix, prof][0] += s
        cells[task, prefix, prof][1] += n
    return cells, tasks, invalid, dict(ignored)


def _adjusted(did, cov):
    """OLS of per-task DiD on the covariate differences; the adjusted estimate is the intercept.

    A covariate that is 0 for every pair carries no information and leaves the intercept unchanged,
    so it is dropped and listed; any other rank deficiency leaves the estimate undefined.
    """
    x = np.array([[cov_t[k] for k in COVARIATES] for cov_t in cov], dtype=float)
    keep = [i for i in range(len(COVARIATES)) if np.any(x[:, i] != 0)]
    design = np.column_stack([np.ones(len(did)), x[:, keep]])
    out = {"dropped_all_zero": [COVARIATES[i] for i in range(len(COVARIATES)) if i not in keep],
           "convention": COVARIATE_CONVENTION}
    if len(did) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return {**out, "estimate": None, "reason": f"{len(did)} tasks cannot identify {design.shape[1]} coefficients"}
    beta = np.linalg.lstsq(design, did, rcond=None)[0]
    return {**out, "estimate": float(beta[0]), "coef": {COVARIATES[i]: float(b) for i, b in zip(keep, beta[1:])}}


def did_test(rows, covariates=None, profile=PROBE, n_boot=5000, seed=0, alpha=0.05, min_effect=MIN_EFFECT):
    """docs/04's P0d test on continuation results.

    rows: {task_id, prefix "low"|"high", profile, successes, attempts, replay_valid}; rows of one cell
    are summed (one row per cell or per seed both work). Rows for profiles other than BF16 and
    `profile` are ignored and counted (Q4_K_M continuations are a separate, secondary run). A pair is
    dropped if any of its rows has an invalid replay. Per task DiD = (Q - BF16 success rate on the low
    prefix) - (Q - BF16 on the high prefix). p = share of task-bootstrap means <= 0. covariates:
    {task_id: {entropy, turns, malformed}} per COVARIATE_CONVENTION, for the adjusted estimate.
    """
    cells, tasks, invalid, ignored = _continuation_cells(rows, profile)
    kept = sorted(tasks - invalid)
    if not kept:
        raise ValueError("no task has a valid replayed pair")
    rate = {}
    for t in kept:
        for prefix in ("low", "high"):
            for prof in ("BF16", profile):
                s, n = cells[t, prefix, prof]
                if n == 0:
                    raise ValueError(f"task {t}: no {prof} attempts on the {prefix} prefix")
                rate[t, prefix, prof] = s / n
    gap = {(t, prefix): rate[t, prefix, profile] - rate[t, prefix, "BF16"] for t in kept for prefix in ("low", "high")}
    did = np.array([gap[t, "low"] - gap[t, "high"] for t in kept])
    means = _boot_weights(len(did), n_boot, np.random.default_rng(seed)) @ did
    p = float(np.mean(means <= 0))
    upper = float(np.percentile(means, 95, method="inverted_cdf"))
    adjusted = None
    if covariates is not None:
        missing = [t for t in kept if t not in covariates]
        if missing:
            raise ValueError(f"no covariates for tasks {missing[:5]}")
        adjusted = _adjusted(did, [{k: _number(_field(covariates[t], k, f"covariates {t}"), f"covariates {t} {k}",
                                               optional=False) for k in COVARIATES} for t in kept])
    mean = float(did.mean())
    adj = adjusted and adjusted["estimate"]
    keeps = adj is not None and mean != 0 and np.sign(adj) == np.sign(mean) and abs(adj) >= 0.5 * abs(mean)
    pass_rule, fail_rule = p < alpha and bool(keeps), upper < min_effect
    # docs/04 lists pass before fail; both can hold for a significant effect below min_effect, so both are reported.
    decision = "pass" if pass_rule else "fail" if fail_rule else "inconclusive"
    return {"profile": profile, "n_tasks": len(kept), "n_pairs_dropped": len(invalid),
            "pair_drop_rate": len(invalid) / len(tasks), "rows_ignored_by_profile": ignored,
            "did_mean": mean, "p_one_sided": p, "upper95_one_sided": upper,
            "upper95_definition": UPPER_BOUND_DEFINITION, "adjusted": adjusted, "alpha": alpha,
            "min_effect": min_effect, "pass_rule": pass_rule, "fail_rule": fail_rule, "decision": decision,
            "n_boot": n_boot, "seed": seed, "per_task": [{"task_id": t, "did": float(d)} for t, d in zip(kept, did)]}


def power(p_bf16, p_q3, n_tasks, seeds, effects=(MIN_EFFECT,), n_sim=1000, n_boot=2000, alpha=0.05, seed=0):
    """Simulated power of did_test's one-sided bootstrap test (p < alpha) at each planted effect.

    p_bf16, p_q3: continuation success rates, scalars or paired per-task arrays (pilot or P0c), drawn
    with replacement for each simulated task. The Q3 rates after the low- and high-score prefixes are
    q + effect/2 and q - effect/2, shifted together back inside [0, 1] where they would leave it, so
    every task carries a gap of exactly `effect` (clipping each side would plant less near 0 and 1,
    and P0c rates of x/4 sit there often). BF16 has the same rate after both. Each of the four cells
    gets `seeds` attempts. The adjusted-estimate condition is not simulated, so this is the power of
    the primary test only.
    """
    b, q = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (p_bf16, p_q3))
    if b.shape != q.shape or ((b < 0) | (b > 1) | (q < 0) | (q > 1)).any():
        raise ValueError("p_bf16 and p_q3 must be rates in [0, 1] of the same length")
    if n_tasks < 2 or seeds < 1:
        raise ValueError("need at least 2 tasks and 1 seed per prefix")
    if any(not 0 <= e <= 1 for e in effects):
        raise ValueError(f"effects must be success-rate gaps in [0, 1], got {effects}")
    rng = np.random.default_rng(seed)
    # One set of bootstrap weights for every simulated study; each study's test stays valid
    # and power is unbiased, the shared draws only add a little Monte Carlo correlation.
    weights = _boot_weights(n_tasks, n_boot, rng)

    def rate(p):
        return rng.binomial(seeds, np.clip(p, 0, 1)) / seeds

    out = []
    for e in effects:
        j = rng.integers(0, len(b), (n_sim, n_tasks))
        low = np.clip(q[j] + e / 2, e, 1)
        did = (rate(low) - rate(b[j])) - (rate(low - e) - rate(b[j]))
        p = ((weights @ did.T) <= 0).mean(axis=0)
        out.append({"effect": float(e), "power": float((p < alpha).mean())})
    passing = [o["effect"] for o in out if o["power"] >= 0.8]
    return {"n_tasks": n_tasks, "seeds": seeds, "p_bf16": b.tolist(), "p_q3": q.tolist(), "alpha": alpha,
            "n_sim": n_sim, "n_boot": n_boot, "power": out, "mde_80": min(passing) if passing else None}
