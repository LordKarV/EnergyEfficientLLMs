"""Tests for the P0 gate statistics on synthetic attempt records. Run:  python -m pytest tests/test_p0_analysis.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.p0_analysis import p0a_primary, p0a_sensitivity, p0b_report, p0c_breakage, processed_tokens


def rec(task, profile, resolved, out=100, unc=900, cached=4000, lst="heldout100", runtime="llama.cpp", repo="scrapy"):
    return {"task_id": task, "profile": profile, "resolved": resolved, "list": lst, "runtime": runtime, "repo": repo,
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached}}


def test_processed_tokens_counts_output_and_all_input():
    assert processed_tokens(rec("t", "BF16", True, out=10, unc=20, cached=30)) == 60


def p0a_data(n_tasks=60, inflate=1.0, q_resolve_every=1):
    rows = []
    for i in range(n_tasks):
        for a in range(3):
            ok = (i + a) % 2 == 0
            rows.append(rec(f"t{i}", "BF16", ok))
            ok_q = ok and (i % q_resolve_every == 0)
            rows.append(rec(f"t{i}", "Q4_K_M", ok_q, out=int(100 * inflate), unc=int(900 * inflate),
                            cached=int(4000 * inflate)))
    return rows


def test_p0a_identical_profiles_fail_with_ratio_one():
    res = p0a_primary(p0a_data(inflate=1.0), n_boot=500, seed=0)
    assert abs(res["R"] - 1.0) < 1e-9
    assert res["ci95"][1] < 1.15 and res["verdict"] == "fail"


def test_p0a_large_inflation_passes():
    res = p0a_primary(p0a_data(inflate=1.4), n_boot=500, seed=0)
    assert abs(res["R"] - 1.4) < 0.02 and res["verdict"] == "pass"


def test_p0a_charges_failures_and_handles_zero_resolve_tasks():
    rows = p0a_data(inflate=1.0, q_resolve_every=2)  # Q4 resolves half as often at the same token cost
    res = p0a_primary(rows, n_boot=200, seed=1)
    assert abs(res["R"] - 1.5) < 1e-9  # same tokens, fewer resolves -> more tokens per resolved attempt
    sens = p0a_sensitivity(rows)
    assert sens["resolve_rate_ratio"] < 1.0 and sens["n_tasks_resolved_both"] > 0


def test_p0a_bootstrap_is_seeded():
    rows = p0a_data(inflate=1.2)
    assert p0a_primary(rows, n_boot=300, seed=3)["ci95"] == p0a_primary(rows, n_boot=300, seed=3)["ci95"]


def test_p0b_report_criteria():
    rows = []
    for i in range(10):  # pool tasks: successes out of 16 = 0, 2, 5, 8, 16, 4, 4, 1, 0, 12
        k = [0, 2, 5, 8, 16, 4, 4, 1, 0, 12][i]
        rows += [rec(f"p{i}", "BF16", a < k, lst="pool200", runtime="vllm", repo="pandas") for a in range(16)]
    for i in range(4):  # held-out mean pass@1 = 0.25
        rows += [rec(f"h{i}", "BF16", a < 4, lst="heldout100", runtime="vllm") for a in range(16)]
    r = p0b_report(rows, inrepo_size=300, pool_total=3887)
    assert r["pool"]["n_tasks"] == 10 and r["pool"]["n_ge4"] == 6
    assert abs(r["criteria"]["share_ge4_among_solvable"] - 6 / 8) < 1e-9
    assert abs(r["projected_pool"] - 0.6 * (3887 - 300)) < 1e-6
    assert abs(r["heldout_mean_pass1"] - 0.25) < 1e-9
    assert abs(r["rl_share_mixed"] - 7 / 10) < 1e-9  # successes in 1..15: 2,5,8,4,4,1,12
    assert r["pass"] == {"c1": True, "c2": True, "c3": True}


def test_p0c_breakage_by_bin():
    vllm_counts = {"a": 0, "b": 2, "c": 6, "d": 12}
    rows = []
    for t, (b, q) in {"a": (0, 0), "b": (2, 1), "c": (4, 1), "d": (4, 2)}.items():  # passes out of 4
        rows += [rec(t, "BF16", i < b, lst="pool200") for i in range(4)]
        rows += [rec(t, "Q4_K_M", i < q, lst="pool200") for i in range(4)]
    r = p0c_breakage(rows, vllm_counts, profile="Q4_K_M", n_boot=200, seed=0)
    assert r["L_by_bin"] == {"0": 0.0, "1-3": 0.25, "4-8": 0.75, "9-16": 0.5}
    assert abs(r["L_total"] - 1.5) < 1e-9 and abs(r["share_4plus"] - 1.25 / 1.5) < 1e-9
