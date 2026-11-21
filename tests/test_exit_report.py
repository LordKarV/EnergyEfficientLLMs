"""Tests for the pilot exit report. Run:  python -m pytest tests/test_exit_report.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from qtrace.exit_report import attempt_stats, project_hours, render_markdown


def rec(runtime, end_reason, t0, t1, out=1000, unc=3000, cached=20000, server_cached=20000, malformed=None,
        tool_s=60.0, cpu_s=30.0, gen_s=120.0, profile="BF16", touched=False):
    return {"runtime": runtime, "end_reason": end_reason, "resolved": end_reason == "submitted",
            "grading": {"test_files_touched": touched},
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached,
                       "server_reported_cached": server_cached},
            "malformed": malformed or {"misnested_tag": 0, "wrong_name": 0, "text_after_call": 0,
                                       "truncated_call": 0, "no_call_no_answer": 0},
            "energy": {"t_start": t0, "t_end": t1}, "tool_seconds": tool_s, "container_cpu_seconds": cpu_s,
            "gen_seconds": gen_s,
            "turns": 10, "tool_calls": 10, "profile": profile, "concurrency": 14}


def test_attempt_stats_rates_and_shares():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "context_limit", 0, 900),
            rec("vllm", "turn_limit", 300, 1200), rec("vllm", "infra_timeout", 600, 3600,
                                                      malformed={"misnested_tag": 2, "wrong_name": 0,
                                                                 "text_after_call": 1, "truncated_call": 0,
                                                                 "no_call_no_answer": 0})]
    s = attempt_stats(rows)["vllm"]
    assert s["n_attempts"] == 4
    assert abs(s["attempts_per_hour"] - 4.0) < 1e-9  # 4 attempts in a 3,600 s span
    assert s["end_reason_share"]["context_limit"] == 0.25
    assert s["context_limit_check"] == {"share": 0.25, "over_10pct": True}
    assert s["malformed_per_turn"]["misnested_tag"] == 2 / 40
    assert s["prefix_cache_hit_rate_ideal"] == 20000 / 23000
    assert s["server_vs_ideal_cached"] == 1.0
    assert s["realized_uncached_per_attempt"] == 3000.0
    assert s["tokens_per_attempt"]["output"] == 1000


def test_project_hours_uses_token_rates_not_wave_time():
    stats = {"tokens_per_attempt": {"output": 1000, "input_uncached": 3000, "input_cached": 20000},
             "tool_seconds_per_attempt": 60.0, "realized_uncached_per_attempt": None}
    rates = {"decode_tok_s": 500.0, "prefill_tok_s": 5000.0}
    h = project_hours(stats, rates, n_attempts=1000, concurrency=14)
    # per attempt: 1000/500 + 3000/5000 = 2.6 s of server time; tool time overlaps across 14 slots
    assert abs(h["server_hours"] - 1000 * 2.6 / 3600) < 1e-9
    assert abs(h["tool_hours"] - 1000 * 60 / 14 / 3600) < 1e-9
    assert h["hours"] == h["server_hours"] + h["tool_hours"]  # 04: rate time plus tool time


def test_markdown_lists_every_required_section():
    rows = [rec("llama.cpp", "submitted", 0, 600)]
    md = render_markdown(attempt_stats(rows), probe=[{"length": 16384, "status": "ok", "sec_per_step": 9.0,
                                                      "mem_gib": 30.0, "mem_kind": "cuda_peak",
                                                      "supervised_frac": 0.15, "loss_path": "chunked"}],
                         extras={}, projections={})
    for heading in ("Attempts per hour", "Limits hit", "Malformed calls", "Tokens and prefix cache",
                    "Training probe", "Not yet provided"):
        assert heading in md
    for key in ("sampler_echo_ok", "per_turn_energy_ok", "concurrency"):
        assert f"`{key}`" in md  # listed as missing until the harness writes it


def test_realized_cache_misses_raise_the_projection():
    stats = {"tokens_per_attempt": {"output": 1000, "input_uncached": 3000, "input_cached": 20000},
             "tool_seconds_per_attempt": 0.0, "realized_uncached_per_attempt": 13000.0}
    rates = {"decode_tok_s": 500.0, "prefill_tok_s": 5000.0}
    real = project_hours(stats, rates, 100, 14)["hours"]
    ideal = project_hours(stats, rates, 100, 14, realized=False)["hours"]
    assert real > ideal and abs(real - 100 * (2 + 13000 / 5000) / 3600) < 1e-9


def test_projections_by_profile_with_fallback_and_sub_run():
    from qtrace.exit_report import projections

    rows = [rec("llama.cpp", "submitted", 0, 600), rec("vllm", "submitted", 0, 600)]
    stats = attempt_stats(rows)
    extras = {"rates": {"llama.cpp": {"BF16": {"decode_tok_s": 400, "prefill_tok_s": 4000},
                                      "Q4_K_M": {"decode_tok_s": 600, "prefill_tok_s": 5000}}}}
    p = projections(stats, extras)
    assert p["P0a"]["profiles"]["Q3_K_M"]["rate_from"] == "Q4_K_M"
    assert p["P0a"]["profiles"]["Q8_0"]["rate_from"] == "BF16"
    assert p["P0a"]["upper_hours"] > p["P0a"]["point_hours"]  # the 600-attempt top-up
    assert "P0b" not in p and any("vllm BF16" in x for x in p["not_included"])
    assert p["energy_sub_run"]["serial"] and abs(p["energy_sub_run"]["point_hours"] - 150 * (120 + 60 + 45) / 3600) < 1e-9


def test_null_server_cached_counts_as_not_reported():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "submitted", 0, 600, server_cached=None)]
    s = attempt_stats(rows)["vllm"]
    assert s["n_server_cached_missing"] == 1
    assert s["realized_uncached_per_attempt"] is None and s["server_vs_ideal_cached"] is None
    assert "| 1 |" in render_markdown({"vllm": s}, probe=[], extras={}, projections={})


def test_error_rows_leave_the_rate_denominators():
    rows = [rec("vllm", "submitted", 0, 600), rec("vllm", "context_limit", 0, 600),
            rec("vllm", "infra_timeout", 0, 600), rec("vllm", "turn_limit", 0, 600),
            rec("vllm", "error", 0, 600), rec("vllm", "error", 0, 600)]
    for r in rows[4:]:
        r["grading"] = None  # an infrastructure fault may never reach the grader
    s = attempt_stats(rows)["vllm"]
    assert s["n_error"] == 2 and s["n_attempts"] == 6
    assert s["resolve_rate"] == 0.25
    assert s["context_limit_check"]["share"] == 0.25 and s["wall_clock_check"]["share"] == 0.25


def test_probe_table_marks_slow_kernel_rows():
    base = {"length": 16384, "status": "ok", "sec_per_step": 9.0, "mem_gib": 30.0, "mem_kind": "cuda_peak",
            "supervised_frac": 0.15, "loss_path": "chunked"}
    ops = lambda *fast: {"in_use": {f"op{i}": {"fast": f, "impl": "x"} for i, f in enumerate(fast)}}
    probe = [{**base, "kernels": ops(True, True)}, {**base, "length": 8192, "kernels": ops(True, False)},
             {**base, "length": 4096}]  # a probe file written before the kernels field
    md = render_markdown({}, probe=probe, extras={}, projections={})
    assert "| loss | supervised | length | status | s/step | memory | kernels |" in md
    assert "| chunked | 0.15 | 16384 | ok | 9.0 | 30.0 GiB (cuda_peak) | fast |" in md
    assert "| chunked | 0.15 | 8192 | ok | 9.0 | 30.0 GiB (cuda_peak) | slow path |" in md
    assert "| chunked | 0.15 | 4096 | ok | 9.0 | 30.0 GiB (cuda_peak) | unknown |" in md


def test_flagged_share_by_profile_and_strict_field():
    rows = [rec("llama.cpp", "submitted", 0, 600, profile="BF16", touched=True),
            rec("llama.cpp", "submitted", 0, 600, profile="BF16"),
            rec("llama.cpp", "turn_limit", 0, 600, profile="Q4_K_M")]
    s = attempt_stats(rows)["llama.cpp"]
    assert s["flagged_share_by_profile"] == {"BF16": 0.5, "Q4_K_M": 0.0}
    assert abs(s["flagged_share"] - 1 / 3) < 1e-12
    assert "| llama.cpp | BF16 | 50.0% |" in render_markdown({"llama.cpp": s}, probe=[], extras={}, projections={})
    del rows[2]["grading"]["test_files_touched"]
    with pytest.raises(KeyError):
        attempt_stats(rows)
    for bad in (None, 0, 1):  # a 0 would print a 0.0% flagged share instead of stopping
        rows[2]["grading"]["test_files_touched"] = bad
        with pytest.raises(TypeError, match="must be a bool"):
            attempt_stats(rows)


@pytest.mark.parametrize("field,value", [("runtime", "vLLM"), ("profile", "Q4_K_M_gates")])
def test_unknown_runtime_or_profile_raises(field, value):
    r = rec("vllm", "submitted", 0, 600)
    r[field] = value
    with pytest.raises(ValueError, match="not in"):
        attempt_stats([r])
