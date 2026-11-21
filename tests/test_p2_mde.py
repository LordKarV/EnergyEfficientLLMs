"""Tests for the P2 minimum detectable effect simulation and the guard's resolve-rate SE.

Run:  python -m pytest tests/test_p2_mde.py -q
"""

import json
import math
import os
import subprocess
import sys
from statistics import NormalDist

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qtrace.p2_analysis import HELDOUT691_BY_REPO, guard_se, mde

Z = NormalDist().inv_cdf


def held(s, k=16, per_repo=4):
    return [{"repo": r, "successes": s, "attempts": k} for r in HELDOUT691_BY_REPO for _ in range(per_repo)]


def test_guard_se_matches_closed_form():
    tasks = [{"repo": "a", "successes": 4, "attempts": 16}, {"repo": "b", "successes": 8, "attempts": 16},
             {"repo": "b", "successes": 0, "attempts": 16}, {"repo": "c", "successes": 2, "attempts": 3}]
    # v_t = s (k - s) / (k (k - 1)): a 0.2; b mean(64/240, 0) = 2/15; c 2/6. Weights 100, 300, 200 of 600.
    v = (100 * 0.2 + 300 * (2 / 15) + 200 * (1 / 3)) / 600
    se = guard_se(tasks, {"a": 100, "b": 300, "c": 200}, seeds=3)
    assert se["mean_binomial_variance"] == pytest.approx(v)
    assert se["se_rr"] == pytest.approx(math.sqrt(v / 1800))
    assert se["se_rr_diff"] == pytest.approx(math.sqrt(2 * v / 1800))


def test_power_known_answer():
    # Every attempt resolves, so T is the log ratio of mean costs; costs 1 or 3 (CV^2 = 1/4) per attempt.
    # Normal approximation: sd(T) = sqrt(2 CV^2 / (S n)); power = Phi(-log r / sd - z_.95), size alpha at r = 1.
    n, S = sum(HELDOUT691_BY_REPO.values()), 3
    sd = math.sqrt(2 * 0.25 / (S * n))
    r80 = math.exp(-sd * (Z(0.95) + Z(0.8)))
    out = mde(held(16), [1.0, 3.0], [1.0], ratios=(r80, 1.0), n_sim=400, n_boot=1000, seed=1)
    pw = {p["ratio"]: p["power"] for p in out["power"]}
    assert pw[r80] == pytest.approx(0.8, abs=0.07)
    assert 0.01 <= pw[1.0] <= 0.10
    assert out["expected_resolves_per_arm_seed"] == n and out["mean_pass1"] == 1.0


def test_mde_picks_largest_passing_ratio_and_is_deterministic():
    kw = {"ratios": (0.5, 0.95, 1.0), "n_sim": 40, "n_boot": 200, "seed": 3}
    a = mde(held(16), [1.0, 3.0], [1.0], **kw)
    assert a == mde(held(16), [1.0, 3.0], [1.0], **kw)
    assert [p["power"] >= 0.8 for p in a["power"]] == [True, True, False]
    assert a["mde_ratio_80"] == 0.95


def test_guard_in_simulation_matches_se():
    # p = 4/16 everywhere and constant cost: the simulated RR difference has sd sqrt(2 p (1 - p) / (S n)) (the
    # plug-in rate), and with margin m sd the guard passes with probability about Phi(m - z_.95).
    n, S, p = sum(HELDOUT691_BY_REPO.values()), 3, 0.25
    sd = math.sqrt(2 * p * (1 - p) / (S * n))
    margin = sd * (Z(0.95) + Z(0.8))
    out = mde(held(4), [1.0], [1.0], ratios=(0.5,), n_sim=400, n_boot=1000, seed=2, guard_margin=margin)
    assert out["guard_se"]["simulated_sd_rr_diff"] == pytest.approx(sd, rel=0.12)
    assert out["power"][0]["power"] == 1.0
    assert out["power"][0]["power_with_guard"] == pytest.approx(0.8, abs=0.07)


@pytest.mark.parametrize("tasks,cr,msg", [
    (held(4)[4:], [1.0], "held-out repos"),
    (held(4, k=1), [1.0], "k >= 2"),
    (held(4), [], "cost_resolved"),
    (held(4), [0.0], "cost_resolved"),
])
def test_mde_rejects_bad_inputs(tasks, cr, msg):
    with pytest.raises(ValueError, match=msg):
        mde(tasks, cr, [1.0], n_sim=1, n_boot=100)


def test_cli_on_attempt_rows(tmp_path):
    rows = []
    for i, repo in enumerate(HELDOUT691_BY_REPO):
        for a in range(4):
            rows.append({"attempt_id": f"h{i}-{a}", "list": "heldout100", "profile": "BF16", "repo": repo,
                         "task_id": f"t{i}", "resolved": a < 2, "end_reason": "submitted"})
    rows.append({"attempt_id": "h-err", "list": "heldout100", "profile": "BF16", "repo": "scrapy", "task_id": "t3",
                 "resolved": False, "end_reason": "error"})
    for a in range(6):
        rows.append({"attempt_id": f"p{a}", "list": "pool200", "profile": "BF16", "repo": "pandas", "task_id": "u",
                     "turns": 3, "resolved": a % 2 == 0, "end_reason": "submitted",
                     "tokens": {"output": 10 * (a + 1), "input_uncached": 5, "input_cached": 5}})
    att, est = tmp_path / "p0b.jsonl", tmp_path / "est.jsonl"
    att.write_text("".join(json.dumps(r) + "\n" for r in rows))
    est.write_text("".join(json.dumps({"attempt_id": f"p{a}", "est_joules_q4": 100.0 + a}) + "\n" for a in range(6)))
    script = os.path.join(ROOT, "scripts", "p2_mde.py")
    common = ["--n-sim", "5", "--n-boot", "100", "--ratios", "0.9", "1.0"]
    for extra, source, mean_resolved in (([], "processed_tokens", 40.0), (["--est-joules", str(est)],
                                                                          "est_joules_q4", 102.0)):
        out_path = tmp_path / f"{source}.json"
        subprocess.run([sys.executable, script, str(att), *common, *extra, "--out", str(out_path)], check=True,
                       capture_output=True)
        out = json.loads(out_path.read_text())
        assert out["cost"]["source"] == source and out["cost"]["mean_resolved"] == mean_resolved
        assert out["n_measured_tasks"] == 4 and out["mean_pass1"] == 0.5  # the error row is left out
        assert [p["ratio"] for p in out["power"]] == [0.9, 1.0]
