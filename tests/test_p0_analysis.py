"""Tests for the P0 gate statistics on synthetic attempt records. Run:  python -m pytest tests/test_p0_analysis.py -q"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace.p0_analysis import (clean, design_mismatches, p0a_factors, p0a_primary, p0a_verdict, p0b_report,
                                p0c_breakage, processed_tokens)

_ids = iter(range(10**9))


def rec(task, profile, resolved, out=100, unc=900, cached=4000, lst="heldout100", runtime="llama.cpp", repo="scrapy",
        end_reason=None, run_id="run-a"):
    end_reason = end_reason or ("submitted" if resolved else "turn_limit")
    return {"attempt_id": f"a{next(_ids)}", "run_id": run_id, "task_id": task, "profile": profile,
            "resolved": resolved, "list": lst, "runtime": runtime, "repo": repo, "end_reason": end_reason,
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached}}


def varied(n_tasks=40, inflate=1.0, q_rate=1.0, seed=0):
    """Tasks differ in difficulty and length, so the bootstrap is not degenerate."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_tasks):
        p, size = rng.uniform(0.2, 0.9), rng.uniform(0.5, 2.0)
        for _ in range(3):
            rows.append(rec(f"t{i}", "BF16", rng.random() < p, out=int(100 * size), unc=int(900 * size),
                            cached=int(4000 * size)))
            rows.append(rec(f"t{i}", "Q4_K_M", rng.random() < p * q_rate, out=int(100 * size * inflate),
                            unc=int(900 * size * inflate), cached=int(4000 * size * inflate)))
    return rows


def test_processed_tokens_counts_output_and_all_input():
    assert processed_tokens(rec("t", "BF16", True, out=10, unc=20, cached=30)) == 60


def test_verdict_boundaries():
    assert p0a_verdict(1.15, 1.0001, 2.0) == "pass"
    assert p0a_verdict(1.15, 1.0, 2.0) == "inconclusive"  # lower bound must exceed 1
    assert p0a_verdict(1.10, 0.9, 1.1499) == "fail"
    assert p0a_verdict(1.10, 0.9, 1.15) == "inconclusive"  # upper bound must be below 1.15
    assert p0a_verdict(None, None, None) == "inconclusive"


def test_identical_profiles_give_r_one_in_every_draw():
    # BF16 and Q4 have the same per-task data; if pairing broke under resampling, draws would vary.
    rows = []
    for r in varied(seed=1):
        if r["profile"] == "BF16":
            rows += [r, {**r, "profile": "Q4_K_M", "attempt_id": r["attempt_id"] + "q"}]
    res = p0a_primary(rows, n_boot=300, seed=0)
    assert abs(res["R"] - 1) < 1e-12 and abs(res["ci95"][0] - 1) < 1e-12 and abs(res["ci95"][1] - 1) < 1e-12
    assert res["verdict"] == "fail"


def test_bootstrap_has_spread_and_depends_on_seed():
    rows = varied(inflate=1.3, seed=2)
    a, b = p0a_primary(rows, n_boot=400, seed=3), p0a_primary(rows, n_boot=400, seed=4)
    assert a["ci95"][0] < a["R"] < a["ci95"][1]
    assert a["ci95"] != b["ci95"] and a == p0a_primary(rows, n_boot=400, seed=3)


def test_clear_inflation_passes_and_factors_compose():
    rows = varied(inflate=1.6, seed=5)
    res = p0a_primary(rows, n_boot=500, seed=0)
    assert res["verdict"] == "pass"
    f = p0a_factors(rows)
    assert abs(f["length_factor"] * f["completion_factor"] - res["R"]) < 1e-9


def test_fewer_resolves_at_equal_tokens_is_completion_not_length():
    rows = []
    for i in range(10):
        rows += [rec(f"t{i}", "BF16", True) for _ in range(3)]
        rows += [rec(f"t{i}", "Q4_K_M", a == 0) for a in range(3)]  # 1 of 3 resolved
    f = p0a_factors(rows)
    assert abs(f["length_factor"] - 1.0) < 1e-12 and abs(f["completion_factor"] - 3.0) < 1e-12
    assert abs(p0a_primary(rows, n_boot=50, seed=0)["R"] - 3.0) < 1e-12


def test_zero_resolve_cases_are_explicit_and_json_safe():
    only_q = [rec("t0", "BF16", False), rec("t0", "Q4_K_M", True)]
    assert p0a_primary(only_q, n_boot=50, seed=0)["R"] == 0.0
    only_b = [rec("t0", "BF16", True), rec("t0", "Q4_K_M", False)]
    res = p0a_primary(only_b, n_boot=50, seed=0)
    assert res["R"] == "inf"
    neither = [rec("t0", "BF16", False), rec("t0", "Q4_K_M", False)]
    assert p0a_primary(neither, n_boot=50, seed=0)["R"] is None
    json.dumps(res, allow_nan=False)


def test_errors_are_dropped_and_counted():
    rows = varied(seed=6)
    extra = [rec("t0", "Q4_K_M", False, end_reason="error", out=50) for _ in range(10)]
    kept, dropped = clean(rows + extra + [rows[0]])
    assert dropped == {"error": 10, "duplicate": 1} and len(kept) == len(rows)


def test_design_mismatches_flag_wrong_attempt_counts():
    rows = [rec("t0", "BF16", True) for _ in range(3)] + [rec("t1", "BF16", True) for _ in range(4)]
    assert design_mismatches(rows, 3) == [["t1", "BF16", 4]]


def test_p0b_report_criteria_and_per_task_counts():
    rows = []
    for i, k in enumerate([0, 2, 5, 8, 16, 4, 4, 1, 0, 12]):
        rows += [rec(f"p{i}", "BF16", a < k, lst="pool200", runtime="vllm", repo="pandas") for a in range(16)]
    for i in range(4):
        rows += [rec(f"h{i}", "BF16", a < 4, lst="heldout100", runtime="vllm") for a in range(16)]
    r = p0b_report(rows, inrepo_size=300)
    assert r["pool"]["n_ge4"] == 6 and r["per_task"]["p4"]["successes"] == 16
    assert abs(r["projected_pool"] - 0.6 * (3887 - 300)) < 1e-6
    assert abs(r["rl_share_mixed"] - 7 / 10) < 1e-12  # 1..15 of 16: 2, 5, 8, 4, 4, 1, 12
    assert r["pass"] == {"c1": True, "c2": True, "c3": True} and r["tasks_below_design_attempts"] == []
    no_held = p0b_report([x for x in rows if x["list"] == "pool200"][:-1], inrepo_size=300)
    assert no_held["pass"]["c3"] == "not computed" and no_held["tasks_below_design_attempts"] == ["p9"]


def test_p0c_bins_ci_and_verdict():
    counts, rows = {}, []
    import random
    rng = random.Random(7)
    for i in range(60):
        counts[f"c{i}"] = rng.choice([0, 2, 6, 12])
        base_p = 0.8 if counts[f"c{i}"] >= 4 else 0.3
        rows += [rec(f"c{i}", "BF16", rng.random() < base_p, lst="pool200") for _ in range(4)]
        rows += [rec(f"c{i}", "Q4_K_M", rng.random() < base_p * 0.6, lst="pool200") for _ in range(4)]
    rows += [rec("unbinned", "BF16", True, lst="pool200"), rec("unbinned", "Q4_K_M", False, lst="pool200")]
    r = p0c_breakage(rows, counts, "Q4_K_M", n_boot=400, seed=0)
    assert r["n_tasks"] == 60 and r["tasks_excluded"] == ["unbinned"]
    assert r["L_total_lo95"] > 0 and r["verdict"] == "pass" and r["share_4plus"] >= 0.4
    assert r["share_4plus_ci95"][0] <= r["share_4plus"] <= r["share_4plus_ci95"][1]


def test_cli_selects_runs_by_manifest(tmp_path):
    rows = []
    for i in range(20):
        rows += [rec(f"p{i}", "BF16", a < 6, lst="pool200", runtime="vllm", repo="pandas", run_id="p0b") for a in range(16)]
        rows += [rec(f"p{i}", "BF16", True, lst="pool200", runtime="vllm", repo="pandas", run_id="pilot") for _ in range(4)]
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "m.json").write_text(json.dumps({"P0b": ["p0b"]}))
    out = tmp_path / "g.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "p0_analyze.py"), str(path), "--manifest",
                    str(tmp_path / "m.json"), "--inrepo-size", "300", "--n-boot", "50", "--out", str(out)],
                   check=True, capture_output=True)
    g = json.loads(out.read_text())["P0b"]
    assert all(v["attempts"] == 16 and v["successes"] == 6 for v in g["per_task"].values())  # pilot excluded
    assert g["design_mismatches"] == []


def test_p0b_counts_only_unflagged_successes():
    rows = []
    for a in range(16):  # 6 resolved, but 3 of them touched test files
        r = rec("p0", "BF16", a < 6, lst="pool200", runtime="vllm", repo="pandas")
        r["grading"] = {"test_files_touched": a < 3}
        rows.append(r)
    rep = p0b_report(rows, inrepo_size=300)
    assert rep["per_task"]["p0"]["successes"] == 3 and rep["pool"]["n_ge4"] == 0
    assert rep["pool"]["flagged_successes_excluded"] == 3
