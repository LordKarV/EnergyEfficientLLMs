"""P0a per-turn energy model (docs/04: P0a energy sub-run, per-turn energy model, batched coefficients).

Fitted per profile on the generation windows of concurrency-1 sub-run attempts, one row per turn with
a measured gen_energy_j. Tool-window energy stays out. Terms (TERMS) are output tokens, uncached
prefill tokens, the context position at turn start (input_cached under full prefix reuse, which
replaces a separate cached-token term), uncached x context, output x context and a per-turn
intercept. Coefficients are non-negative.

Two targets per profile, both fitted and reported: "raw" (the counter's gen_energy_j) and
"idle_subtracted" (gen_energy_j - energy.idle_w_ref x gen_s). Runs are ranked with the raw Q4_K_M
fit: the P2 endpoint, the estimator check and the energy upper bound all compare against measured
GPU joules, which include the idle draw, and at concurrency 1 the device is held for the whole
generation window. Idle subtraction also adds the error of a separately measured idle reference.

The intercept (the fixed per-turn overhead docs/04 names) is always in the model. The token terms
are the subset of the five, the empty one included, with the lowest leave-one-attempt-out (LOAO) CV
error: the median absolute relative error on attempt joules (an attempt's summed predicted turn
joules against its summed measured ones). Ties go to fewer terms, then the lower bitmask. The
pre-registered fewest_joules fallback reads the nested LOAO error, where the term choice is redone
inside each outer fold, because the plain LOAO error of a chosen term set is optimistic.

NNLS is exact without scipy: with k <= 6 terms, the NNLS solution is the least-squares fit on its
active set, so fitting every column subset and keeping the best one with no negative coefficient
finds it. Everything runs on per-attempt Gram matrices, so a fold or bootstrap draw is a sum.
"""

import itertools
import math
from collections import Counter, defaultdict

import numpy as np

TERMS = ("output", "uncached", "context", "uncached_x_context", "output_x_context", "intercept")
CANDIDATES = range(1 << TERMS.index("intercept"), 2 ** len(TERMS))  # term-set bitmasks holding the intercept
TARGETS = ("raw", "idle_subtracted")
RANK_PROFILE, RANK_TARGET = "Q4_K_M", "raw"
FALLBACK_MAX_REL_ERR = 0.10  # 04: fewest_joules falls back to fewest_turns above this
AGREEMENT_MIN = 0.85  # 04: batched vs concurrency-1 picks below this name their regime
MIN_CANDIDATES = 4  # 03 Method step 1: pool tasks have at least 4 successful BF16 runs; the P0b fallback is 2
HELDOUT_LISTS = ("heldout100", "verified100", "inrepo_test")
HELDOUT_REPOS = ("scrapy", "pyramid", "datalad", "coveragepy")
CONTEXT_BINS = (16384, 32768, 49152)  # residual summary bins over the 64K context


def _get(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"{where}: missing field {key!r}")
    return d[key]


def _num(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise ValueError(f"{where} must be a finite non-negative number, got {x!r}")
    return float(x)


def _turns(rec):
    turns = rec.get("per_turn")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"attempt {rec.get('attempt_id')!r}: per_turn must be a non-empty list")
    return turns


def turn_features(turn, where="turn"):
    """The TERMS values of one per_turn entry, validated."""
    o, u, c = (_num(_get(turn, k, where), f"{where}: {k}") for k in ("output_tokens", "input_uncached", "input_cached"))
    return [o, u, c, u * c, o * c, 1.0]


def attempt_arrays(rec):
    """Feature matrix and both targets over the turns of `rec` with a measured gen_energy_j.

    Returns {attempt_id, task_id, profile, X [n, 6], raw [n], idle_subtracted [n], n_unmeasured,
    server_cached_differs (None when the server count is null)}. Idle-subtracted turn joules may be
    negative on tiny turns; only attempt sums must be positive (checked in fit_profile).
    """
    aid = _get(rec, "attempt_id", "attempt")
    where = f"attempt {aid!r}"
    idle_w = _num(_get(_get(rec, "energy", where), "idle_w_ref", where), f"{where}: energy.idle_w_ref")
    X, raw, idle = [], [], []
    turns = _turns(rec)
    for i, t in enumerate(turns):
        w = f"{where} turn {i}"
        e = _get(t, "gen_energy_j", w)
        if e is None:
            continue
        e, s = _num(e, f"{w}: gen_energy_j"), _num(_get(t, "gen_s", w), f"{w}: gen_s")
        X.append(turn_features(t, w))
        raw.append(e)
        idle.append(e - idle_w * s)
    tokens = _get(rec, "tokens", where)
    server = tokens.get("server_reported_cached")
    return {"attempt_id": aid, "task_id": _get(rec, "task_id", where), "profile": _get(rec, "profile", where),
            "X": np.array(X, dtype=float).reshape(-1, len(TERMS)), "raw": np.array(raw),
            "idle_subtracted": np.array(idle), "n_unmeasured": len(turns) - len(raw),
            "server_cached_differs": None if server is None else server != _get(tokens, "input_cached", where)}


def fit_attempts(records):
    """Sub-run attempts usable for the fit, as {profile: [attempt_arrays]}, plus exclusion counts.

    Kept: concurrency 1, energy.attributable true, end_reason not "error" (an infrastructure fault,
    rerun as a fresh attempt), and at least one measured turn. Duplicate attempt ids fail.
    """
    seen, by_profile, excluded = set(), defaultdict(list), Counter()
    for r in records:
        aid = _get(r, "attempt_id", "sub-run record")
        where = f"attempt {aid!r}"
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r} in the sub-run files")
        seen.add(aid)
        if _get(r, "concurrency", where) != 1:
            excluded["concurrency_not_1"] += 1
        elif _get(_get(r, "energy", where), "attributable", where) is not True:
            excluded["not_attributable"] += 1
        elif _get(r, "end_reason", where) == "error":
            excluded["error"] += 1
        else:
            a = attempt_arrays(r)
            if len(a["raw"]):
                by_profile[a["profile"]].append(a)
            else:
                excluded["no_measured_turns"] += 1
    return dict(by_profile), dict(excluded)


def _scale(X):
    """Column RMS (1 for an all-zero column). NNLS predictions do not depend on positive column scaling."""
    s = np.sqrt((X ** 2).mean(axis=0))
    return np.where(s > 0, s, 1.0)


def _nnls_all(G, g, yy):
    """Exact NNLS restricted to every column subset at once, from the Gram matrix G = X'X, g = X'y, yy = y'y.

    Returns B [2^k, k]: row m is the NNLS solution using only the columns in bitmask m (row 0 is 0).
    Each subset's least-squares fit is computed once; the NNLS solution on m is the feasible subset
    of m (no negative coefficient) with the least residual sum of squares.
    """
    k = len(g)
    masks = np.arange(2 ** k)
    inn = ((masks[:, None] >> np.arange(k)) & 1).astype(float)
    # Columns outside a subset get an identity row and a zero target, so one batched pinv solves all.
    A = G * inn[:, :, None] * inn[:, None, :] + np.eye(k) * (1 - inn)[:, None, :]
    # Zero the excluded columns exactly: pinv leaves ~1e-15 there, and a tiny negative one would mark a
    # feasible subset infeasible.
    beta = np.einsum("mij,mj->mi", np.linalg.pinv(A), g * inn) * inn
    rss = yy - 2 * beta @ g + np.einsum("mi,ij,mj->m", beta, G, beta)
    rss = np.where((beta >= 0).all(axis=1), rss, np.inf)
    sub = (masks[None, :] & ~masks[:, None]) == 0  # [m, t]: t is a subset of m
    return beta[np.argmin(np.where(sub, rss[None, :], np.inf), axis=1)]


def nnls(X, y):
    """Exact non-negative least squares on all columns of X (k small; every subset is tried)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    s = _scale(X)
    Xs = X / s
    return _nnls_all(Xs.T @ Xs, Xs.T @ y, y @ y)[-1] / s


def _loao_preds(G, g, yy, S, members):
    """Held-out attempt joules [len(members), 2^k] for every term set, each fold fitted on members minus one."""
    tG, tg, tyy = G[members].sum(0), g[members].sum(0), yy[members].sum()
    return np.array([_nnls_all(tG - G[i], tg - g[i], tyy - yy[i]) @ S[i] for i in members])


def _choose(err):
    """Term set from held-out errors [n, 2^k]: the CANDIDATES set with the lowest median error.

    Ties go to fewer terms, then the lower bitmask. The lowest error, not the fewest terms within one
    bootstrap SE of it: in the owner's planted-model probe (15 x 2 attempts, 5% per-turn and 3%
    per-attempt noise, 40 repeats) the one-SE rule found the planted set 0/40 times and its fresh-data
    error was 1.7x this rule's, because it dropped real terms.
    """
    cv = np.median(err, axis=0)
    return min(CANDIDATES, key=lambda m: (cv[m], m.bit_count(), m)), cv


def _terms(mask):
    return [t for j, t in enumerate(TERMS) if mask >> j & 1]


def pair_ordering(task_ids, attempt_ids, measured, predicted):
    """Share of measured within-task pairs whose predicted joules are ordered like the measured ones."""
    by_task = defaultdict(list)
    for i, t in enumerate(task_ids):
        by_task[t].append(i)
    pairs = []
    for t, idx in sorted(by_task.items()):
        for i, j in itertools.combinations(idx, 2):
            pairs.append({"task_id": t, "attempts": [attempt_ids[i], attempt_ids[j]],
                          "measured_j": [float(measured[i]), float(measured[j])],
                          "predicted_j": [float(predicted[i]), float(predicted[j])],
                          "correct": bool((predicted[i] - predicted[j]) * (measured[i] - measured[j]) > 0)})
    n_ok = sum(p["correct"] for p in pairs)
    return {"n_pairs": len(pairs), "n_correct": n_ok, "share_correct": n_ok / len(pairs) if pairs else None,
            "pairs": pairs}


def _residuals(attempts, target, coef, rel_loao):
    """In-sample turn residuals (J, overall and by context bin) and LOAO relative errors on attempt joules."""
    X = np.vstack([a["X"] for a in attempts])
    res = np.concatenate([a[target] for a in attempts]) - X @ coef
    bins = np.digitize(X[:, TERMS.index("context")], CONTEXT_BINS)
    names = ["<16K", "16-32K", "32-48K", ">=48K"]
    return {"turn_rmse_j": float(np.sqrt((res ** 2).mean())), "turn_median_abs_j": float(np.median(np.abs(res))),
            "turn_mean_j_by_context": {n: {"n": int((bins == b).sum()),
                                           "mean_j": float(res[bins == b].mean()) if (bins == b).any() else None}
                                       for b, n in enumerate(names)},
            "attempt_loao_rel": {"median_abs": float(np.median(np.abs(rel_loao))),
                                 "p90_abs": float(np.percentile(np.abs(rel_loao), 90)),
                                 "max_abs": float(np.abs(rel_loao).max()), "mean_signed": float(rel_loao.mean())}}


def fit_profile(attempts, target="raw", n_boot=2000, seed=0):
    """Choose terms, fit, and report one profile's model for one target (docs/04 P0a reporting list).

    attempts: one profile's attempt_arrays (from fit_attempts). Returns {terms, coef (J per unit of
    each chosen term), coef_ci95 (percentile bootstrap over attempts, terms held fixed, seeded),
    cv {loao, nested_loao},
    fallback (nested_loao above FALLBACK_MAX_REL_ERR), residuals, pairs (LOAO predictions), n_attempts,
    n_turns}. Raises if there are fewer than 3 attempts (nested LOAO needs them) or an attempt's measured
    sum is <= 0.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    n = len(attempts)
    if n < 3:
        raise ValueError(f"nested LOAO needs at least 3 attempts per profile, got {n} "
                         f"for {sorted({a['profile'] for a in attempts})}")
    scale = _scale(np.vstack([a["X"] for a in attempts]))
    Xs = [a["X"] / scale for a in attempts]
    ys = [a[target] for a in attempts]
    G = np.stack([x.T @ x for x in Xs])
    g = np.stack([x.T @ y for x, y in zip(Xs, ys)])
    yy = np.array([y @ y for y in ys])
    S = np.stack([x.sum(0) for x in Xs])
    Y = np.array([y.sum() for y in ys])
    bad = [a["attempt_id"] for a, v in zip(attempts, Y) if v <= 0]
    if bad:
        raise ValueError(f"non-positive measured {target} joules for attempts {bad}; check energy.idle_w_ref")

    everyone = np.arange(n)
    P = _loao_preds(G, g, yy, S, everyone)
    m, cv = _choose(np.abs(P - Y[:, None]) / Y[:, None])

    nested = []
    for i in everyone:
        rest = everyone[everyone != i]
        mi = _choose(np.abs(_loao_preds(G, g, yy, S, rest) - Y[rest, None]) / Y[rest, None])[0]
        pred = _nnls_all(G[rest].sum(0), g[rest].sum(0), yy[rest].sum())[mi] @ S[i]
        nested.append(abs(pred - Y[i]) / Y[i])
    nested = float(np.median(nested))

    coef = _nnls_all(G.sum(0), g.sum(0), yy.sum())[m] / scale
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(_nnls_all(G[idx].sum(0), g[idx].sum(0), yy[idx].sum())[m] / scale)
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0) if draws else (coef, coef)
    terms = _terms(m)
    return {"terms": terms, "coef": {t: float(coef[TERMS.index(t)]) for t in terms},
            "coef_ci95": {t: [float(lo[TERMS.index(t)]), float(hi[TERMS.index(t)])] for t in terms},
            "cv": {"loao": float(cv[m]), "nested_loao": nested},
            "fallback": nested > FALLBACK_MAX_REL_ERR,
            "residuals": _residuals(attempts, target, coef, (P[:, m] - Y) / Y),
            "pairs": pair_ordering([a["task_id"] for a in attempts], [a["attempt_id"] for a in attempts], Y, P[:, m]),
            "n_attempts": n, "n_turns": int(sum(len(y) for y in ys)), "n_boot": n_boot, "seed": seed}


def profile_report(attempts, n_boot=2000, seed=0):
    """One profile's entry in energy_model.json: both targets, the calibration range and the cache-count check."""
    X = np.vstack([a["X"] for a in attempts])
    flags = [a["server_cached_differs"] for a in attempts]
    known = [f for f in flags if f is not None]
    rng_of = {t: [float(X[:, TERMS.index(t)].min()), float(X[:, TERMS.index(t)].max())]
              for t in ("output", "uncached", "context")}
    n_turns = [len(a["raw"]) for a in attempts]
    return {"n_attempts": len(attempts), "n_turns": len(X),
            "n_turns_unmeasured": sum(a["n_unmeasured"] for a in attempts),
            "calibration_range": {**rng_of, "turns_per_attempt": [min(n_turns), max(n_turns)]},
            # 04: how often llama.cpp's cached-prompt counts differ from the ideal split (attempt totals;
            # per-turn server counts are not in the record).
            "server_cache_mismatch": {"n_attempts": len(flags), "n_differ": sum(known),
                                      "n_missing": len(flags) - len(known),
                                      "share_differ": sum(known) / len(known) if known else None},
            **{t: fit_profile(attempts, target=t, n_boot=n_boot, seed=seed) for t in TARGETS}}


def estimate_attempt_joules(rec, model):
    """Estimated generation-window joules of an attempt: predicted turn joules summed over rec["per_turn"].

    model: a fit_profile report or any {"coef": {term: J per unit}}; terms left out count as 0. Rank
    runs with energy_model.json["profiles"]["Q4_K_M"]["raw"]. Tool-window energy is not included.
    """
    unknown = set(model["coef"]) - set(TERMS)
    if unknown:
        raise ValueError(f"unknown energy-model terms {sorted(unknown)}; expected a subset of {TERMS}")
    w = np.array([model["coef"].get(t, 0.0) for t in TERMS])
    aid = rec.get("attempt_id")
    return float(sum(w @ turn_features(t, f"attempt {aid!r} turn {i}") for i, t in enumerate(_turns(rec))))


def pool_records(records):
    """Pool attempts to estimate, with infrastructure errors dropped and counted.

    Fails on an empty pool, duplicate attempt ids, missing fields, and any held-out list or repo
    (held-out tasks never enter a pool). Returns (kept records, number of end_reason "error" dropped).
    """
    if not records:
        raise ValueError("the pool is empty")
    kept, seen, n_error = [], set(), 0
    for r in records:
        aid = _get(r, "attempt_id", "pool record")
        where = f"pool attempt {aid!r}"
        if aid in seen:
            raise ValueError(f"duplicate attempt_id {aid!r} in the pool files")
        seen.add(aid)
        lst, repo = _get(r, "list", where), _get(r, "repo", where)
        if lst in HELDOUT_LISTS or repo in HELDOUT_REPOS:
            raise ValueError(f"{where} is from list {lst!r}, repo {repo!r}: held-out tasks never enter a pool")
        for k in ("task_id", "profile", "turns", "end_reason"):
            _get(r, k, where)
        if _get(r, "resolved", where) is not True and r["resolved"] is not False:
            raise ValueError(f"{where}: resolved must be true or false, got {r['resolved']!r}")
        if r["end_reason"] == "error":
            n_error += 1
        else:
            kept.append(r)
    return kept, n_error


def _avg_ranks(x):
    _, inv, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (start + (counts - 1) / 2)[inv]


def _spearman(a, b):
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _cheapest(cands, est):
    return min(range(len(cands)), key=lambda i: (est[i], cands[i]["turns"], cands[i]["attempt_id"]))


def pick_agreement(records, est_a, est_b, min_candidates=MIN_CANDIDATES):
    """How often two energy estimators pick the same cheapest run per task, and their mean within-task Spearman.

    Candidates are the runs the P1 arms pick from: BF16 runs (records from pool_records) that resolved
    and whose grader did not flag test-file edits (an absent flag is false), grouped by task. Tasks with
    fewer than min_candidates are skipped; fewer candidates agree more often by chance, so a looser
    pool would bias the 85% check toward agreement. A pick breaks ties by fewer turns, then by
    attempt id (deterministic here; the arms break remaining ties with their seed). Spearman uses
    average ranks and is skipped for a task where either estimator gives every run the same value.
    """
    if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < 2:
        raise ValueError(f"min_candidates must be an integer >= 2, got {min_candidates!r}")
    by_task = defaultdict(list)
    for r in records:
        if r["profile"] == "BF16" and r["resolved"] and not (r.get("grading") or {}).get("test_files_touched"):
            by_task[r["task_id"]].append(r)
    same, rhos = [], []
    for c in by_task.values():
        if len(c) < min_candidates:
            continue
        a, b = [est_a(r) for r in c], [est_b(r) for r in c]
        same.append(_cheapest(c, a) == _cheapest(c, b))
        rho = _spearman(a, b)
        if rho is not None:
            rhos.append(rho)
    return {"min_candidates": min_candidates, "n_tasks": len(same),
            "share_same_pick": float(np.mean(same)) if same else None,
            "mean_spearman": float(np.mean(rhos)) if rhos else None, "n_tasks_spearman": len(rhos)}


def fit_batched(windows):
    """Batched coefficients: each 1 s server-metrics window's joules ~ intercept + prefill + decode, non-negative.

    windows: [{t, joules, prompt_tokens, predicted_tokens}] with per-window token deltas. Raw joules
    only: idle subtraction would shift the intercept and leave the token slopes unchanged.
    """
    if not windows:
        raise ValueError("no server-metrics windows")
    X, y = [], []
    for i, w in enumerate(windows):
        where = f"window {i}"
        _get(w, "t", where)
        y.append(_num(_get(w, "joules", where), f"{where}: joules"))
        X.append([1.0] + [_num(_get(w, k, where), f"{where}: {k}") for k in ("prompt_tokens", "predicted_tokens")])
    b = nnls(X, y)
    return {"coef": {"intercept_per_window": float(b[0]), "prefill": float(b[1]), "decode": float(b[2])},
            "n_windows": len(y)}


def estimate_batched_joules(rec, batched):
    """An attempt's marginal joules under the batched fit: prefill x uncached tokens + decode x output tokens.

    The per-window intercept is static power shared by every concurrent attempt, so it is left out.
    """
    aid = rec.get("attempt_id")
    feats = np.array([turn_features(t, f"attempt {aid!r} turn {i}") for i, t in enumerate(_turns(rec))])
    return float(batched["coef"]["prefill"] * feats[:, 1].sum() + batched["coef"]["decode"] * feats[:, 0].sum())
