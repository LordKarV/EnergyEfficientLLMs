"""Tests for the P0a per-turn energy model on synthetic sub-run data.

Run:  python -m pytest tests/test_energy_model.py -q
"""

import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace import energy_model
from qtrace.energy_model import (
    TERMS,
    attempt_arrays,
    estimate_attempt_joules,
    estimate_batched_joules,
    fit_attempts,
    fit_batched,
    fit_profile,
    nnls,
    pair_ordering,
    pick_agreement,
    pool_records,
)

# Planted active (idle-subtracted) joules per turn; each term is a visible share of a turn's energy.
PLANT = {"output": 2.0, "uncached": 0.1, "output_x_context": 2e-5, "intercept": 150.0}
FULL = {"output": 2.0, "uncached": 0.1, "context": 4e-3, "uncached_x_context": 2e-6, "output_x_context": 2e-5,
        "intercept": 150.0}
IDLE_W = 40.0


def make_attempt(rng, aid, task, coef, noise=0.01, attempt_noise=0.0, profile="Q4_K_M"):
    """One concurrency-1 attempt. Context grows by the previous prompt and output (full prefix reuse).

    gen_energy_j = active joules (planted coefficients, per-turn and per-attempt multiplicative noise)
    plus IDLE_W x gen_s, so the idle-subtracted target is exactly the planted active energy.
    """
    factor = math.exp(attempt_noise * rng.standard_normal())
    cached, uncached, turns = 0, int(rng.integers(2000, 4000)), []
    for i in range(int(rng.integers(8, 25))):
        out = int(rng.integers(50, 2500))
        x = dict(zip(TERMS, (out, uncached, cached, uncached * cached, out * cached, 1)))
        active = sum(coef.get(k, 0.0) * x[k] for k in TERMS) * factor * math.exp(noise * rng.standard_normal())
        gen_s = 0.2 + out / 60 + uncached / 3000
        turns.append({"turn": i, "output_tokens": out, "input_uncached": uncached, "input_cached": cached,
                      "context_len": cached + uncached + out, "gen_s": gen_s, "gen_energy_j": active + IDLE_W * gen_s,
                      "tool_s": 1.0, "tool_energy_j": IDLE_W})
        cached += uncached + out
        uncached = int(rng.integers(100, 3000))
        if cached > 60000:
            break
    tot_cached = sum(t["input_cached"] for t in turns)
    return {"attempt_id": aid, "run_id": "sub", "task_id": task, "list": "pool200", "repo": "pandas",
            "profile": profile, "concurrency": 1, "resolved": True, "end_reason": "submitted", "turns": len(turns),
            "tokens": {"output": sum(t["output_tokens"] for t in turns),
                       "input_uncached": sum(t["input_uncached"] for t in turns), "input_cached": tot_cached,
                       "server_reported_cached": tot_cached},
            "per_turn": turns, "energy": {"gpu_j": None, "attributable": True, "idle_w_ref": IDLE_W}}


def subrun(coef, seed=0, n_tasks=15, per_task=2, profile="Q4_K_M", **kw):
    """The sub-run design: n_tasks x per_task attempts of one profile."""
    rng = np.random.default_rng(seed)
    return [make_attempt(rng, f"{profile}-t{t}-a{a}", f"t{t}", coef, profile=profile, **kw)
            for t in range(n_tasks) for a in range(per_task)]


def fitted(records, target="idle_subtracted", n_boot=200, seed=0):
    by_profile, _ = fit_attempts(records)
    return fit_profile(by_profile["Q4_K_M"], target=target, n_boot=n_boot, seed=seed)


def pool_rec(aid, task, out, unc, resolved=True, turns=1, lst="pool200", repo="pandas", end_reason="submitted",
             profile="BF16"):
    return {"attempt_id": aid, "task_id": task, "list": lst, "repo": repo, "profile": profile, "resolved": resolved,
            "end_reason": end_reason, "turns": turns,
            "per_turn": [{"output_tokens": out, "input_uncached": unc, "input_cached": 0}]}


def test_recovers_planted_coefficients_and_picks_the_planted_terms():
    rep = fitted(subrun(PLANT, noise=0.01))
    assert rep["terms"] == [t for t in TERMS if t in PLANT]  # context and uncached_x_context are spurious
    for k, v in PLANT.items():
        assert abs(rep["coef"][k] / v - 1) < 0.05, k
        lo, hi = rep["coef_ci95"][k]
        assert lo <= v <= hi, k


def test_recovers_all_six_terms_when_all_are_planted():
    rep = fitted(subrun(FULL, noise=0.005, n_tasks=20))
    assert rep["terms"] == list(TERMS)
    for k, v in FULL.items():
        assert abs(rep["coef"][k] / v - 1) < 0.15, k


def test_raw_target_keeps_idle_power_and_idle_subtraction_removes_it():
    recs = subrun(PLANT, noise=0.01)
    raw, idle = fitted(recs, target="raw"), fitted(recs, target="idle_subtracted")
    # gen_s = 0.2 + out/60 + unc/3000, so idle power lands on the output, uncached and intercept terms.
    assert raw["coef"]["output"] == pytest.approx(PLANT["output"] + IDLE_W / 60, rel=0.05)
    assert raw["coef"]["intercept"] == pytest.approx(PLANT["intercept"] + IDLE_W * 0.2, rel=0.1)
    assert idle["coef"]["output"] == pytest.approx(PLANT["output"], rel=0.05)


def test_idle_subtraction_arithmetic_and_unmeasured_turns():
    rec = subrun(PLANT, n_tasks=1, per_task=1)[0]
    rec["energy"]["idle_w_ref"] = 10.0
    rec["per_turn"] = rec["per_turn"][:3]
    for t, (e, s) in zip(rec["per_turn"], [(100.0, 2.0), (None, 5.0), (50.0, 1.0)]):
        t["gen_energy_j"], t["gen_s"] = e, s
    a = attempt_arrays(rec)
    assert a["raw"].tolist() == [100.0, 50.0] and a["idle_subtracted"].tolist() == [80.0, 40.0]
    assert a["n_unmeasured"] == 1 and a["X"].shape == (2, len(TERMS))


def test_fallback_follows_the_nested_error():
    low = fitted(subrun(PLANT, noise=0.01), n_boot=20)
    high = fitted(subrun(PLANT, noise=0.3, attempt_noise=0.4), n_boot=20)
    assert low["cv"]["nested_loao"] < 0.02 and low["fallback"] is False
    assert high["cv"]["nested_loao"] > 0.10 and high["fallback"] is True


def test_fallback_is_nested_error_strictly_above_ten_percent(monkeypatch):
    # The flag must read the nested error, not the optimistic plain one, and "exceeds 10%" is strict.
    assert energy_model.FALLBACK_MAX_REL_ERR == 0.10
    attempts = fit_attempts(subrun({"output": 2.0}, seed=2, n_tasks=4, noise=0.5, attempt_noise=0.2))[0]["Q4_K_M"]
    cv = fit_profile(attempts, n_boot=0)["cv"]
    assert cv["loao"] < cv["nested_loao"]
    for threshold, fallback in [((cv["loao"] + cv["nested_loao"]) / 2, True), (cv["nested_loao"], False)]:
        monkeypatch.setattr(energy_model, "FALLBACK_MAX_REL_ERR", threshold)
        assert fit_profile(attempts, n_boot=0)["fallback"] is fallback


def test_term_choice_keeps_the_intercept_and_takes_the_lowest_cv_error():
    # docs/04 P0a: the intercept is always in; the token terms are the subset with the lowest LOAO error,
    # ties to fewer terms. Columns are bitmasks over TERMS; the intercept is bit 5 (32).
    err = np.ones((5, 2 ** len(TERMS)))
    err[:, 1] = 0.005  # output alone, no intercept: lowest overall, never a candidate
    err[:, 32 | 1] = 0.02  # output + intercept
    err[:, 32 | 1 | 2] = 0.01  # output + uncached + intercept: the lowest candidate
    assert energy_model._choose(err)[0] == 32 | 1 | 2
    err[:, 32 | 1] = 0.01  # a tie goes to fewer terms
    assert energy_model._choose(err)[0] == 32 | 1

    # The owner's probe (5% per-turn, 3% per-attempt noise): the old one-SE rule dropped the intercept.
    for seed in range(1000, 1005):
        rep = fitted(subrun(PLANT, seed=seed, noise=0.05, attempt_noise=0.03), target="raw", n_boot=0)
        assert "intercept" in rep["terms"], seed


def test_nested_cv_is_no_lower_than_plain_cv_with_spurious_terms():
    # Few attempts and only `output` planted: the plain LOAO error of the chosen term set is the minimum of
    # many noisy estimates, so it is optimistic; nested LOAO redoes the choice inside each outer fold.
    gaps = []
    for seed in range(4):
        rep = fitted(subrun({"output": 2.0}, seed=seed, n_tasks=4, noise=0.5, attempt_noise=0.2), n_boot=10)
        gaps.append(rep["cv"]["nested_loao"] - rep["cv"]["loao"])
    assert min(gaps) >= -1e-12 and sum(gaps) > 0  # equal (to rounding) when every fold picks the same terms


def test_coefficients_stay_nonnegative_where_least_squares_goes_negative():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(0, 1, (200, 3)), np.ones(200)])
    y = X @ [2.0, -1.0, 0.5, 3.0] + 0.01 * rng.standard_normal(200)
    assert np.linalg.lstsq(X, y, rcond=None)[0][1] < 0
    b = nnls(X, y)
    assert (b >= 0).all() and b[1] == 0
    grad = X.T @ (X @ b - y)  # KKT: zero on the support, non-negative off it
    assert np.allclose(grad[b > 0], 0, atol=1e-8) and (grad[b == 0] >= 0).all()

    recs = subrun({**PLANT, "context": -1.5e-3}, noise=0.01)  # active joules stay positive at 60K context
    by_profile, _ = fit_attempts(recs)
    X = np.vstack([a["X"] for a in by_profile["Q4_K_M"]])
    y = np.concatenate([a["idle_subtracted"] for a in by_profile["Q4_K_M"]])
    assert np.linalg.lstsq(X / np.abs(X).max(0), y, rcond=None)[0][TERMS.index("context")] < 0
    rep = fitted(recs)
    assert all(v >= 0 for v in rep["coef"].values()) and all(lo >= 0 for lo, _ in rep["coef_ci95"].values())


def test_pair_ordering():
    r = pair_ordering(["a", "a", "b", "b", "c"], ["a1", "a2", "b1", "b2", "c1"], measured=[10, 20, 30, 40, 5],
                      predicted=[11, 19, 45, 35, 5])
    assert (r["n_pairs"], r["n_correct"], r["share_correct"]) == (2, 1, 0.5)
    assert [p["correct"] for p in r["pairs"]] == [True, False]
    assert r["pairs"][0]["attempts"] == ["a1", "a2"]


def test_fitted_report_orders_well_separated_pairs():
    rep = fitted(subrun(PLANT, noise=0.01))
    assert rep["pairs"]["n_pairs"] == 15 and rep["pairs"]["share_correct"] >= 0.9


def chosen_fit(attempts, cols, target="raw"):
    return nnls(np.vstack([a["X"][:, cols] for a in attempts]), np.concatenate([a[target] for a in attempts]))


def test_pairs_use_held_out_predictions():
    attempts = fit_attempts(subrun(PLANT, noise=0.3, attempt_noise=0.4))[0]["Q4_K_M"]
    rep = fit_profile(attempts, target="raw", n_boot=0)
    cols = [TERMS.index(t) for t in rep["terms"]]
    loao = {a["attempt_id"]: a["X"][:, cols].sum(0) @ chosen_fit(attempts[:i] + attempts[i + 1:], cols)
            for i, a in enumerate(attempts)}
    insample = chosen_fit(attempts, cols)
    for p in rep["pairs"]["pairs"]:
        assert p["predicted_j"] == pytest.approx([loao[a] for a in p["attempts"]], rel=1e-6)
    gap = max(abs(a["X"][:, cols].sum(0) @ insample / loao[a["attempt_id"]] - 1) for a in attempts)
    assert gap > 1e-2  # in-sample predictions would fail the check above


def test_bootstrap_resamples_attempts_with_the_chosen_terms_fixed():
    attempts = fit_attempts(subrun(PLANT, noise=0.05))[0]["Q4_K_M"]
    rep = fit_profile(attempts, target="raw", n_boot=100, seed=7)
    assert len(rep["terms"]) < len(TERMS) and set(rep["coef_ci95"]) == set(rep["terms"])
    cols = [TERMS.index(t) for t in rep["terms"]]
    rng = np.random.default_rng(7)
    draws = [chosen_fit([attempts[i] for i in rng.integers(0, len(attempts), len(attempts))], cols)
             for _ in range(100)]
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    for j, t in enumerate(rep["terms"]):
        assert rep["coef_ci95"][t] == pytest.approx([lo[j], hi[j]], rel=1e-6), t


def test_estimate_attempt_joules_sums_predicted_turns():
    rec = {"attempt_id": "x", "per_turn": [{"output_tokens": 10, "input_uncached": 100, "input_cached": 0},
                                           {"output_tokens": 20, "input_uncached": 50, "input_cached": 110}]}
    model = {"coef": {"output": 2.0, "output_x_context": 0.01, "intercept": 5.0}}
    assert estimate_attempt_joules(rec, model) == pytest.approx((20 + 5) + (40 + 0.01 * 20 * 110 + 5))


def test_pick_agreement_and_candidate_rules():
    pool = [pool_rec("a1", "A", 100, 900), pool_rec("a2", "A", 200, 800), pool_rec("a3", "A", 300, 700),
            pool_rec("a4", "A", 10, 10, resolved=False),  # cheapest, but a failure is never a candidate
            pool_rec("b1", "B", 50, 500), pool_rec("b2", "B", 60, 400),
            pool_rec("c1", "C", 50, 500)]  # one candidate: skipped
    pool[4]["grading"] = {"test_files_touched": False}
    by_out = lambda r: estimate_attempt_joules(r, {"coef": {"output": 1.0}})
    by_unc = lambda r: estimate_attempt_joules(r, {"coef": {"uncached": 1.0}})
    same = pick_agreement(pool, by_out, by_out, min_candidates=2)
    assert same == {"min_candidates": 2, "n_tasks": 2, "share_same_pick": 1.0, "mean_spearman": 1.0,
                    "n_tasks_spearman": 2}
    opposite = pick_agreement(pool, by_out, by_unc, min_candidates=2)
    assert opposite["share_same_pick"] == 0.0 and opposite["mean_spearman"] == pytest.approx(-1.0)

    flagged = copy.deepcopy(pool)
    flagged[0]["grading"] = {"test_files_touched": True}  # a1 drops out, so A's cheapest by output is a2
    assert pick_agreement(flagged, by_out, by_unc, min_candidates=2)["share_same_pick"] == 0.0
    ties = [pool_rec("t1", "T", 5, 5, turns=3), pool_rec("t2", "T", 5, 5, turns=2)]
    flat = lambda r: 1.0
    assert pick_agreement(ties, flat, by_out, min_candidates=2) == {
        "min_candidates": 2, "n_tasks": 1, "share_same_pick": 1.0, "mean_spearman": None, "n_tasks_spearman": 0}


def test_pick_agreement_uses_the_arms_candidates():
    # Only BF16 successes are candidates, and a task needs min_candidates of them (docs/03 Method step 1: 4).
    by_out = lambda r: estimate_attempt_joules(r, {"coef": {"output": 1.0}})
    by_unc = lambda r: estimate_attempt_joules(r, {"coef": {"uncached": 1.0}})
    mixed = [pool_rec("bf-1", "M", 100, 900), pool_rec("bf-2", "M", 200, 100),
             pool_rec("q4-1", "M", 50, 50, profile="Q4_K_M")]  # cheapest by both, but not a BF16 run
    assert pick_agreement(mixed, by_out, by_unc, min_candidates=2)["share_same_pick"] == 0.0
    four = [pool_rec(f"f{i}", "F", 10 * (i + 1), 100 * (4 - i)) for i in range(4)]
    r = pick_agreement(mixed + four, by_out, by_unc)
    assert (r["min_candidates"], r["n_tasks"], r["share_same_pick"]) == (4, 1, 0.0)
    with pytest.raises(ValueError, match="min_candidates"):
        pick_agreement(mixed, by_out, by_unc, min_candidates=1)


def test_batched_fit_recovers_coefficients_and_feeds_agreement():
    rng = np.random.default_rng(5)
    windows = []
    for i in range(600):
        p, d = int(rng.integers(0, 20000)), int(rng.integers(0, 800))
        windows.append({"t": float(i), "joules": 60 + 0.02 * p + 0.8 * d + rng.normal(0, 2), "prompt_tokens": p,
                        "predicted_tokens": d})
    fit = fit_batched(windows)
    assert fit["n_windows"] == 600
    assert fit["coef"]["intercept_per_window"] == pytest.approx(60, rel=0.02)
    assert fit["coef"]["prefill"] == pytest.approx(0.02, rel=0.02)
    assert fit["coef"]["decode"] == pytest.approx(0.8, rel=0.02)

    rec = pool_rec("x", "X", 10, 100)
    assert estimate_batched_joules(rec, fit) == pytest.approx(fit["coef"]["prefill"] * 100 + fit["coef"]["decode"] * 10)
    pool = [pool_rec("a1", "A", 100, 900), pool_rec("a2", "A", 200, 100), pool_rec("b1", "B", 50, 50),
            pool_rec("b2", "B", 60, 40)]
    c1 = lambda r: estimate_attempt_joules(r, {"coef": {"output": 1.0}})
    decode_only = {"coef": {"intercept_per_window": 60.0, "prefill": 0.0, "decode": 1.0}}
    prefill_heavy = {"coef": {"intercept_per_window": 60.0, "prefill": 1.0, "decode": 0.0}}
    agree = lambda est: pick_agreement(pool, c1, est, min_candidates=2)["share_same_pick"]
    assert agree(lambda r: estimate_batched_joules(r, decode_only)) == 1.0
    assert agree(lambda r: estimate_batched_joules(r, prefill_heavy)) == 0.0

    fit_neg = fit_batched([{**w, "joules": 60 - 1e-4 * w["prompt_tokens"] + 0.8 * w["predicted_tokens"]}
                           for w in windows])
    assert min(fit_neg["coef"].values()) >= 0


def test_fit_data_keeps_only_attributable_concurrency_one_attempts():
    recs = subrun(PLANT, n_tasks=3)
    recs[0]["concurrency"] = 16
    recs[1]["energy"]["attributable"] = False
    recs[2]["end_reason"] = "error"
    recs[3]["per_turn"] = [{**t, "gen_energy_j": None} for t in recs[3]["per_turn"]]
    by_profile, excluded = fit_attempts(recs)
    assert [a["attempt_id"] for a in by_profile["Q4_K_M"]] == [recs[4]["attempt_id"], recs[5]["attempt_id"]]
    assert excluded == {"concurrency_not_1": 1, "not_attributable": 1, "error": 1, "no_measured_turns": 1}


def test_malformed_inputs_fail_loudly():
    base = subrun(PLANT, n_tasks=2)

    def broken(edit):
        recs = copy.deepcopy(base)
        edit(recs[0])
        return recs

    cases = [
        (lambda r: r["per_turn"][0].pop("output_tokens"), "output_tokens"),
        (lambda r: r["per_turn"][1].update(input_uncached=-5), "input_uncached"),
        (lambda r: r["per_turn"][1].pop("gen_energy_j"), "gen_energy_j"),
        (lambda r: r["per_turn"][1].update(gen_energy_j=float("nan")), "gen_energy_j"),
        (lambda r: r["energy"].pop("idle_w_ref"), "idle_w_ref"),
        (lambda r: r.pop("concurrency"), "concurrency"),
        (lambda r: r.pop("end_reason"), "end_reason"),
        (lambda r: r.update(per_turn=[]), "per_turn"),
    ]
    for edit, field in cases:
        with pytest.raises(ValueError, match=field):
            fit_attempts(broken(edit))
    with pytest.raises(ValueError, match="duplicate"):
        fit_attempts(base + base[:1])
    with pytest.raises(ValueError, match="at least 3"):
        fit_profile(fit_attempts(base[:2])[0]["Q4_K_M"])
    hot = copy.deepcopy(base)
    hot[0]["energy"]["idle_w_ref"] = 1e6  # idle reference above the measured draw
    with pytest.raises(ValueError, match="non-positive"):
        fit_profile(fit_attempts(hot)[0]["Q4_K_M"], target="idle_subtracted")

    with pytest.raises(ValueError, match="empty"):
        pool_records([])
    with pytest.raises(ValueError, match="held-out"):
        pool_records([pool_rec("p", "t", 1, 1, lst="heldout100")])
    with pytest.raises(ValueError, match="held-out"):
        pool_records([pool_rec("p", "t", 1, 1, repo="scrapy")])
    with pytest.raises(ValueError, match="duplicate"):
        pool_records([pool_rec("p", "t", 1, 1), pool_rec("p", "t", 1, 1)])
    with pytest.raises(ValueError, match="resolved"):
        pool_records([{k: v for k, v in pool_rec("p", "t", 1, 1).items() if k != "resolved"}])
    with pytest.raises(ValueError, match="resolved"):
        pool_records([pool_rec("p", "t", 1, 1, resolved="false")])
    with pytest.raises(ValueError, match="profile"):
        pool_records([{k: v for k, v in pool_rec("p", "t", 1, 1).items() if k != "profile"}])
    with pytest.raises(ValueError, match="unknown"):
        estimate_attempt_joules(pool_rec("p", "t", 1, 1), {"coef": {"prefill": 1.0}})

    with pytest.raises(ValueError, match="no server-metrics windows"):
        fit_batched([])
    with pytest.raises(ValueError, match="predicted_tokens"):
        fit_batched([{"t": 0.0, "joules": 1.0, "prompt_tokens": 1}])


def test_pool_records_drop_and_count_infrastructure_errors():
    kept, n_error = pool_records([pool_rec("a", "t", 1, 1), pool_rec("b", "t", 1, 1, end_reason="error")])
    assert [r["attempt_id"] for r in kept] == ["a"] and n_error == 1


def test_cli_writes_model_and_estimates(tmp_path):
    q4 = subrun(PLANT, seed=1, profile="Q4_K_M")
    bf16 = subrun({**PLANT, "output": 3.0}, seed=2, profile="BF16")
    q4[0]["tokens"]["server_reported_cached"] -= 100  # one attempt's server count disagrees with the ideal split
    concurrent = copy.deepcopy(q4[1])
    concurrent.update(attempt_id="conc", concurrency=16)
    concurrent["energy"] = {"gpu_j": None, "attributable": False, "idle_w_ref": None}
    sub = tmp_path / "sub.jsonl"
    sub.write_text("".join(json.dumps(r) + "\n" for r in q4 + bf16 + [concurrent]))

    rng = np.random.default_rng(9)
    pool = [make_attempt(rng, f"p{t}-{a}", f"p{t}", PLANT, profile="BF16") for t in range(20) for a in range(4)]
    pool[0]["resolved"] = False
    pool[1]["end_reason"] = "error"
    pool_path = tmp_path / "pool.jsonl"
    pool_path.write_text("".join(json.dumps(r) + "\n" for r in pool))
    windows = [{"t": float(i), "joules": 60 + 0.1 * p + 2.5 * d, "prompt_tokens": p, "predicted_tokens": d}
               for i, (p, d) in enumerate(rng.integers(0, 2000, (200, 2)).tolist())]
    win_path = tmp_path / "windows.jsonl"
    win_path.write_text("".join(json.dumps(w) + "\n" for w in windows))

    out = tmp_path / "p0a"
    script = os.path.join(ROOT, "scripts", "p0a_energy_model.py")
    args = ["--subrun", str(sub), "--pool", str(pool_path), "--windows", str(win_path), "--out", str(out),
            "--n-boot", "30"]
    res = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    em = json.loads((out / "energy_model.json").read_text())
    assert em["fewest_joules_fallback"] is False and set(em["profiles"]) == {"Q4_K_M", "BF16"}
    assert em["excluded_subrun_attempts"] == {"concurrency_not_1": 1}
    assert em["profiles"]["Q4_K_M"]["idle_subtracted"]["coef"]["output"] == pytest.approx(2.0, rel=0.05)
    assert em["profiles"]["BF16"]["idle_subtracted"]["coef"]["output"] == pytest.approx(3.0, rel=0.05)
    assert em["profiles"]["Q4_K_M"]["server_cache_mismatch"] == {"n_attempts": 30, "n_differ": 1, "n_missing": 0,
                                                                 "share_differ": pytest.approx(1 / 30)}
    assert em["pool"] == {"n_attempts": 79, "n_dropped_error": 1}
    # p0 keeps 2 candidates (one failure, one error), below min_candidates 4.
    assert em["bf16_vs_q4"]["n_tasks"] == 19 and em["bf16_vs_q4"]["min_candidates"] == 4
    assert em["bf16_vs_q4"]["share_same_pick"] is not None
    b = em["batched"]
    assert b["coef"]["decode"] == pytest.approx(2.5, rel=0.01)
    assert b["below_85pct"] == (b["agreement_with_c1"]["share_same_pick"] < 0.85)

    rows = [json.loads(line) for line in (out / "est_joules.jsonl").read_text().splitlines()]
    assert len(rows) == 79 and set(rows[0]) == {"attempt_id", "est_joules_q4"}
    model = em["profiles"]["Q4_K_M"]["raw"]
    by_id = {r["attempt_id"]: r for r in pool}
    assert all(r["est_joules_q4"] == pytest.approx(estimate_attempt_joules(by_id[r["attempt_id"]], model))
               for r in rows)

    bf16_only = tmp_path / "bf16.jsonl"
    bf16_only.write_text("".join(json.dumps(r) + "\n" for r in bf16))
    res = subprocess.run([sys.executable, script, "--subrun", str(bf16_only), "--pool", str(pool_path), "--out",
                          str(tmp_path / "x")], capture_output=True, text=True, check=False)
    assert res.returncode != 0 and "Q4_K_M" in res.stderr
