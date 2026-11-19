"""Tests for the pilot exit report. Run:  python -m pytest tests/test_exit_report.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.exit_report import attempt_stats, project_hours, render_markdown


def rec(runtime, end_reason, t0, t1, out=1000, unc=3000, cached=20000, server_cached=20000, malformed=None,
        tool_s=60.0, cpu_s=30.0):
    return {"runtime": runtime, "end_reason": end_reason, "resolved": end_reason == "submitted",
            "tokens": {"output": out, "input_uncached": unc, "input_cached": cached,
                       "server_reported_cached": server_cached},
            "malformed": malformed or {"misnested_tag": 0, "wrong_name": 0, "text_after_call": 0,
                                       "truncated_call": 0, "no_call_no_answer": 0},
            "energy": {"t_start": t0, "t_end": t1}, "tool_seconds": tool_s, "container_cpu_seconds": cpu_s,
            "turns": 10, "tool_calls": 10, "profile": "BF16", "concurrency": 14}


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
    assert s["malformed_per_call"]["misnested_tag"] == 2 / 40
    assert s["prefix_cache_hit_rate_ideal"] == 20000 / 23000
    assert s["server_vs_ideal_cached"] == 1.0
    assert s["tokens_per_attempt"]["output"] == 1000


def test_project_hours_uses_token_rates_not_wave_time():
    stats = {"tokens_per_attempt": {"output": 1000, "input_uncached": 3000, "input_cached": 20000},
             "tool_seconds_per_attempt": 60.0}
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
