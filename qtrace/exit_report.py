"""Pilot exit report (docs/04, P0b pilot): the numbers the owner reads to set the P0 spending ceiling.

attempt_stats and the markdown come from the attempt records agreed in issue #1. Numbers the attempt
records can't give (aggregate token rates from synthetic load, peak GPU memory, energy-counter checks,
idle watts, the replay match rate) come from an optional extras JSON; anything missing is listed under
"Not yet provided" rather than silently left out.
"""

from collections import Counter, defaultdict

MALFORMED = ["misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer"]
# Attempts per P0 step and the runtime each runs on (docs/04, Compute estimate and the gate sections).
P0_STEPS = {"P0b": ("vllm", 4800), "P0a": ("llama.cpp", 3000), "P0c": ("llama.cpp", 2400), "P0d": ("llama.cpp", 800)}
HERO_PREFIX_HIT = 0.92
# Keys of the extras JSON, agreed with the harness in issue #1 / PR #7.
EXTRAS = ["rates", "peak_gpu_gib", "energy_counter", "replay_match_rate", "sampler_echo_ok",
          "per_turn_energy_ok", "concurrency"]


def attempt_stats(rows):
    by_runtime = defaultdict(list)
    for r in rows:
        by_runtime[r["runtime"]].append(r)
    return {rt: _stats(rs) for rt, rs in sorted(by_runtime.items())}


def _stats(rows):
    n = len(rows)
    span = max(r["energy"]["t_end"] for r in rows) - min(r["energy"]["t_start"] for r in rows)
    ends = Counter(r["end_reason"] for r in rows)
    calls = sum(r["tool_calls"] for r in rows)
    tok = {k: sum(r["tokens"][k] for r in rows) for k in ("output", "input_uncached", "input_cached")}
    server_cached = sum(r["tokens"].get("server_reported_cached", 0) for r in rows)
    ctx_share = ends.get("context_limit", 0) / n
    wall_share = ends.get("infra_timeout", 0) / n
    hit = tok["input_cached"] / (tok["input_cached"] + tok["input_uncached"]) if tok["input_cached"] else 0.0
    return {
        "n_attempts": n,
        "span_s": span,
        "attempts_per_hour": n / span * 3600 if span else None,
        "resolve_rate": sum(bool(r["resolved"]) for r in rows) / n,
        "end_reason_share": {k: v / n for k, v in sorted(ends.items())},
        "context_limit_check": {"share": ctx_share, "over_10pct": ctx_share > 0.10},
        "wall_clock_check": {"share": wall_share, "over_2pct": wall_share > 0.02},
        "malformed_per_call": {k: sum(r["malformed"][k] for r in rows) / calls if calls else 0.0 for k in MALFORMED},
        "tokens_per_attempt": {k: v / n for k, v in tok.items()},
        "prefix_cache_hit_rate_ideal": hit,
        "prefix_cache_flag": hit < HERO_PREFIX_HIT - 0.12,
        "server_vs_ideal_cached": server_cached / tok["input_cached"] if tok["input_cached"] else None,
        "tool_seconds_per_attempt": sum(r["tool_seconds"] for r in rows) / n,
        "container_vcpus_busy": sum(r["container_cpu_seconds"] for r in rows) / span if span else None,
        "concurrency": Counter(r["concurrency"] for r in rows).most_common(1)[0][0],
    }


def project_hours(stats, rates, n_attempts, concurrency):
    """Instance-hours for n attempts from aggregate token rates (not from one wave's wall clock).

    rates are aggregate tokens per second with all slots busy, so server time is already shared
    across slots; tool time runs in parallel across `concurrency` containers. Following docs/04
    ("tokens per attempt / aggregate token rate plus tool time"), the two are added, which is
    conservative: in practice they partly overlap.
    """
    t = stats["tokens_per_attempt"]
    per_attempt = t["output"] / rates["decode_tok_s"] + t["input_uncached"] / rates["prefill_tok_s"]
    server_h = n_attempts * per_attempt / 3600
    tool_h = n_attempts * stats["tool_seconds_per_attempt"] / concurrency / 3600
    return {"server_hours": server_h, "tool_hours": tool_h, "hours": server_h + tool_h}


def projections(stats, extras):
    out = {}
    for step, (runtime, n) in P0_STEPS.items():
        rate = extras.get("rates", {}).get(runtime)
        if rate and runtime in stats:
            out[step] = {"runtime": runtime, "attempts": n,
                         **project_hours(stats[runtime], rate, n, stats[runtime]["concurrency"])}
    if out:
        out["total_hours"] = sum(v["hours"] for v in out.values())
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(stats, probe, extras, projections):
    lines = ["# Pilot exit report", ""]
    lines += ["## Attempts per hour", "", "| runtime | attempts | per hour | resolve rate | concurrency |",
              "|---|---|---|---|---|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | {s['n_attempts']} | {s['attempts_per_hour']:.1f} | {_pct(s['resolve_rate'])} "
                     f"| {s['concurrency']} |")
    lines += ["", "## Limits hit", "", "| runtime | end reasons | 64K context (owner check at 10%) "
              "| wall clock (cap check at 2%) |", "|---|---|---|---|"]
    for rt, s in stats.items():
        ends = ", ".join(f"{k} {_pct(v)}" for k, v in s["end_reason_share"].items())
        ctx, wall = s["context_limit_check"], s["wall_clock_check"]
        lines.append(f"| {rt} | {ends} | {_pct(ctx['share'])}{' **over**' if ctx['over_10pct'] else ''} "
                     f"| {_pct(wall['share'])}{' **over**' if wall['over_2pct'] else ''} |")
    lines += ["", "## Malformed calls", "", "Per tool call, by category.", "",
              "| runtime | " + " | ".join(MALFORMED) + " |", "|---" * (len(MALFORMED) + 1) + "|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | " + " | ".join(_pct(s["malformed_per_call"][k]) for k in MALFORMED) + " |")
    lines += ["", "## Tokens and prefix cache", "",
              "| runtime | output | uncached input | cached input | ideal hit rate | server / ideal cached "
              "| tool s | vCPUs busy |", "|---|---|---|---|---|---|---|---|"]
    for rt, s in stats.items():
        t = s["tokens_per_attempt"]
        flag = " **below HERO's ~92%**" if s["prefix_cache_flag"] else ""
        ratio = "n/a" if s["server_vs_ideal_cached"] is None else f"{s['server_vs_ideal_cached']:.2f}"
        busy = "n/a" if s["container_vcpus_busy"] is None else f"{s['container_vcpus_busy']:.1f}"
        lines.append(f"| {rt} | {t['output']:.0f} | {t['input_uncached']:.0f} | {t['input_cached']:.0f} "
                     f"| {_pct(s['prefix_cache_hit_rate_ideal'])}{flag} | {ratio} "
                     f"| {s['tool_seconds_per_attempt']:.0f} | {busy} |")
    lines += ["", "## Training probe", ""]
    if probe:
        lines += ["| loss | supervised | length | status | s/step | memory |", "|---|---|---|---|---|---|"]
        for p in probe:
            sec = f"{p['sec_per_step']:.1f}" if p["status"] == "ok" else "-"
            mem = "n/a" if p.get("mem_gib") is None else f"{p['mem_gib']:.1f} GiB ({p['mem_kind']})"
            lines.append(f"| {p['loss_path']} | {p['supervised_frac']:.2f} | {p['length']} | {p['status']} "
                         f"| {sec} | {mem} |")
    else:
        lines.append("No probe results.")
    if projections:
        lines += ["", "## Projected instance-hours", "", "| step | runtime | attempts | hours |", "|---|---|---|---|"]
        for step, v in projections.items():
            if step != "total_hours":
                lines.append(f"| {step} | {v['runtime']} | {v['attempts']} | {v['hours']:.1f} |")
        lines.append(f"| total | | | {projections['total_hours']:.1f} |")
    missing = [k for k in EXTRAS if k not in extras]
    lines += ["", "## Not yet provided", ""]
    lines += [f"- `{k}`" for k in missing] or ["Nothing: every section is filled."]
    for k in EXTRAS[1:]:
        if k in extras:
            lines += ["", f"`{k}`: `{extras[k]}`"]
    return "\n".join(lines) + "\n"
