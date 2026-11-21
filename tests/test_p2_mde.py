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

import numpy as np

from qtrace import p2_analysis
from qtrace.p2_analysis import HELDOUT691_BY_REPO, guard_se, mde
from qtrace.p2_data import strat_weights

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


def test_mde_with_guard_reports_the_guard_capped_mde():
    # p = 4/16, constant costs: ratio 0.5 is always significant, but a margin of z_.95 sd passes the guard in about
    # half the studies at every ratio, so no ratio reaches 80% power with the guard.
    n, S, p = sum(HELDOUT691_BY_REPO.values()), 3, 0.25
    margin = math.sqrt(2 * p * (1 - p) / (S * n)) * Z(0.95)
    out = mde(held(4), [1.0], [1.0], ratios=(0.5, 1.0), n_sim=200, n_boot=300, seed=4, guard_margin=margin)
    assert out["power"][0]["power"] == 1.0 and out["mde_ratio_80"] == 0.5
    assert out["mde_ratio_80_with_guard"] is None
    assert out["guard_pass_rate"] == out["power"][0]["power_with_guard"]
    assert out["guard_pass_rate"] == pytest.approx(0.5, abs=0.1)
    plain = mde(held(4), [1.0], [1.0], ratios=(0.5,), n_sim=2, n_boot=100)
    assert "mde_ratio_80_with_guard" not in plain and "guard_pass_rate" not in plain


def test_mde_resamples_and_weights_within_each_repo(monkeypatch):
    # Only coveragepy's tasks resolve (8/16), and it holds 300 of the 500 simulated tasks, against 4 of the 16
    # measured ones: the c3 count, the RR-difference spread and the bootstrap strata all depend on the repo.
    tasks = [{"repo": r, "successes": 8 if r == "coveragepy" else 0, "attempts": 16}
             for r in HELDOUT691_BY_REPO for _ in range(4)]
    n_by_repo = {"coveragepy": 300, "datalad": 100, "pyramid": 50, "scrapy": 50}
    seen = []

    def spy(repos, n_boot, rng):
        seen.append(strat_weights(repos, n_boot, rng))
        return seen[-1]

    monkeypatch.setattr(p2_analysis, "strat_weights", spy)
    out = mde(tasks, [1.0], [1.0], ratios=(0.5,), n_by_repo=n_by_repo, n_sim=400, n_boot=100, seed=5)
    assert out["expected_resolves_per_arm_seed"] == 150.0 and out["mean_pass1"] == 0.3
    # Var of the paired RR difference: 2 p (1 - p) / S per coveragepy task, none from the others, over n^2.
    assert out["guard_se"]["simulated_sd_rr_diff"] == pytest.approx(math.sqrt(2 * 0.25 * 300 / 3) / 500, rel=0.12)
    repos = np.repeat(sorted(n_by_repo), [n_by_repo[r] for r in sorted(n_by_repo)])
    (W,) = seen
    for r, k in n_by_repo.items():
        assert (W[:, repos == r].sum(1) == k).all()


@pytest.mark.parametrize("tasks,cr,msg", [
    (held(4)[4:], [1.0], "held-out repos"),
    (held(4, k=1), [1.0], "k >= 2"),
    (held(4), [], "cost_resolved"),
    (held(4), [0.0], "cost_resolved"),
])
def test_mde_rejects_bad_inputs(tasks, cr, msg):
    with pytest.raises(ValueError, match=msg):
        mde(tasks, cr, [1.0], n_sim=1, n_boot=100)


def cli_rows(profile="BF16", runtime="vllm", resolved_of=lambda a: a < 2, out_tokens=10):
    rows = []
    for i, repo in enumerate(HELDOUT691_BY_REPO):
        for a in range(4):
            rows.append({"attempt_id": f"h{profile}{i}-{a}", "list": "heldout100", "profile": profile, "repo": repo,
                         "runtime": runtime, "task_id": f"t{i}", "resolved": resolved_of(a), "end_reason": "submitted"})
    for a in range(6):
        rows.append({"attempt_id": f"p{profile}{a}", "list": "pool200", "profile": profile, "repo": "pandas",
                     "runtime": runtime, "task_id": "u", "turns": 3, "resolved": a % 2 == 0, "end_reason": "submitted",
                     "tokens": {"output": out_tokens * (a + 1), "input_uncached": 5, "input_cached": 5}})
    return rows


def run_cli(tmp_path, rows, *extra, name="att"):
    att = tmp_path / f"{name}.jsonl"
    att.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out_path = tmp_path / f"{name}.json"
    script = os.path.join(ROOT, "scripts", "p2_mde.py")
    proc = subprocess.run([sys.executable, script, str(att), "--n-sim", "5", "--n-boot", "100", "--ratios", "0.9",
                           "1.0", *extra, "--out", str(out_path)], check=False, capture_output=True, text=True)
    return proc, (json.loads(out_path.read_text()) if proc.returncode == 0 else None)


def test_cli_on_attempt_rows(tmp_path):
    rows = cli_rows()
    rows.append({"attempt_id": "h-err", "list": "heldout100", "profile": "BF16", "repo": "scrapy", "task_id": "t3",
                 "runtime": "vllm", "resolved": False, "end_reason": "error"})
    est = tmp_path / "est.jsonl"
    est.write_text("".join(json.dumps({"attempt_id": f"pBF16{a}", "est_joules_q4": 100.0 + a}) + "\n"
                           for a in range(6)))
    for extra, source, mean_resolved in (([], "processed_tokens", 40.0), (["--est-joules", str(est)],
                                                                          "est_joules_q4", 102.0)):
        proc, out = run_cli(tmp_path, rows, *extra, name=source)
        assert proc.returncode == 0, proc.stderr
        assert out["cost"]["source"] == source and out["cost"]["mean_resolved"] == mean_resolved
        assert out["n_measured_tasks"] == 4 and out["mean_pass1"] == 0.5  # the error row is left out
        assert [p["ratio"] for p in out["power"]] == [0.9, 1.0]


def test_cli_counts_only_submitted_resolves_and_reads_each_profile(tmp_path):
    rows = cli_rows() + cli_rows("Q4_K_M", "llama.cpp", lambda a: True, out_tokens=1000)
    # A resolved attempt that hit the turn limit is a failure (P2 counts only submitted resolves).
    rows += [{"attempt_id": f"hl{i}", "list": "heldout100", "profile": "BF16", "repo": repo, "runtime": "vllm",
              "task_id": f"t{i}", "resolved": True, "end_reason": "turn_limit"}
             for i, repo in enumerate(HELDOUT691_BY_REPO)]
    proc, bf16 = run_cli(tmp_path, rows, name="bf16")
    assert proc.returncode == 0, proc.stderr
    assert bf16["mean_pass1"] == 0.4 and bf16["cost"]["mean_resolved"] == 40.0
    proc, q4 = run_cli(tmp_path, rows, "--profile", "Q4_K_M", "--pool-profile", "Q4_K_M", name="q4")
    assert proc.returncode == 0, proc.stderr
    assert q4["mean_pass1"] == 1.0 and q4["cost"]["mean_resolved"] == 3010.0


def test_cli_fails_on_a_pool_attempt_without_est_joules(tmp_path):
    est = tmp_path / "est.jsonl"
    est.write_text("".join(json.dumps({"attempt_id": f"pBF16{a}", "est_joules_q4": 1.0}) + "\n" for a in range(5)))
    proc, _ = run_cli(tmp_path, cli_rows(), "--est-joules", str(est))
    assert proc.returncode != 0 and "1 pool attempts have no est_joules row" in proc.stderr and "pBF165" in proc.stderr


def test_cli_fails_on_heldout_rows_from_two_runtimes(tmp_path):
    # P0a also runs BF16 on heldout100, on llama.cpp; its attempts must not join P0b's vLLM counts.
    p0a = [dict(r, attempt_id="a" + r["attempt_id"], runtime="llama.cpp", resolved=True)
           for r in cli_rows() if r["list"] == "heldout100"]
    proc, _ = run_cli(tmp_path, cli_rows() + p0a)
    assert proc.returncode != 0 and "['llama.cpp', 'vllm']" in proc.stderr
