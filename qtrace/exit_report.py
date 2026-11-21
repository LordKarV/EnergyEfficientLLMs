"""Pilot exit report (docs/04, P0b pilot): the numbers the owner reads to set the P0 spending ceiling.

attempt_stats and the markdown come from the attempt records agreed in issue #1. Numbers the attempt
records can't give (aggregate token rates from synthetic load, peak GPU memory, energy-counter checks,
idle watts, the replay match rate) come from an optional extras JSON; anything missing is listed under
"Not yet provided" rather than silently left out.
"""

from collections import Counter, defaultdict

# Allowed record values. Runtimes from docs/04 (Serving); profile names from docs/04 (Profiles), with
# Q4_K_M_gate8 naming the Q4_K_M file whose ssm_alpha/ssm_beta are overridden to Q8_0. Issue #1 does not
# pin these strings yet, so an unknown value raises here instead of moving a step under "Not included".
RUNTIMES = ("vllm", "llama.cpp")
PROFILES = ("BF16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8")
MALFORMED = ["misnested_tag", "wrong_name", "text_after_call", "truncated_call", "no_call_no_answer"]
# Concurrent P0 blocks (docs/04): runtime, attempts per profile at the point and the upper count.
# P0a has a 600-attempt top-up (BF16 and Q4_K_M) if inconclusive; P0d can grow to about 1,600.
P0_STEPS = {
    "P0b": ("vllm", {"BF16": (4800, 4800)}),
    "P0a": ("llama.cpp", {"BF16": (600, 900), "Q8_0": (600, 600), "Q4_K_M": (600, 900), "Q3_K_M": (600, 600),
                          "Q4_K_M_gate8": (600, 600)}),
    "P0c": ("llama.cpp", {"BF16": (800, 800), "Q4_K_M": (800, 800), "Q3_K_M": (800, 800)}),
    "P0d": ("llama.cpp", {"BF16": (400, 800), "Q3_K_M": (400, 800)}),
}
# Profiles without their own measured rate borrow one: conservative for Q3_K_M (no faster than Q4_K_M).
RATE_FALLBACK = {"Q8_0": "BF16", "Q3_K_M": "Q4_K_M", "Q4_K_M_gate8": "Q4_K_M"}
SUB_RUN = {"attempts": 150, "settle_s": 45}  # energy sub-run: 5 profiles x 15 tasks x 2, concurrency 1
NOT_INCLUDED = ["KL scoring of the P0b pool (BF16 and Q3_K_M forward passes, on its own machine)",
                "per-block overhead: server restarts, GGUF loads and image pulls between serial blocks"]
HERO_PREFIX_HIT = 0.92
# Keys of the extras JSON, agreed with the harness in issue #1 / PR #7.
EXTRAS = ["rates", "peak_gpu_gib", "energy_counter", "replay_match_rate", "sampler_echo_ok",
          "per_turn_energy_ok", "concurrency"]


def attempt_stats(rows):
    by_runtime = defaultdict(list)
    for r in rows:
        if r["runtime"] not in RUNTIMES or r["profile"] not in PROFILES:
            raise ValueError(f"attempt {r.get('attempt_id')}: runtime {r['runtime']!r} / profile {r['profile']!r} "
                             f"not in {RUNTIMES} / {PROFILES}")
        by_runtime[r["runtime"]].append(r)
    return {rt: _stats(rs) for rt, rs in sorted(by_runtime.items())}


def _stats(rows):
    n = len(rows)
    span = max(r["energy"]["t_end"] for r in rows) - min(r["energy"]["t_start"] for r in rows)
    ends = Counter(r["end_reason"] for r in rows)
    # end_reason error is an infrastructure fault, rerun as a fresh attempt (04), so those rows stay out of the
    # resolve, limit and flagged rates (as #6, #13 and #14 drop them); their count is reported.
    scored = [r for r in rows if r["end_reason"] != "error"]
    m = len(scored)
    by_profile = defaultdict(list)
    for r in scored:
        by_profile[r["profile"]].append(r["grading"]["test_files_touched"])  # required bool: missing raises
    turns = sum(r["turns"] for r in rows)
    tok = {k: sum(r["tokens"][k] for r in rows) for k in ("output", "input_uncached", "input_cached")}
    # A missing or null count means the server did not report it (as in #13); the realized split then needs all.
    n_server_missing = sum(r["tokens"].get("server_reported_cached") is None for r in rows)
    has_server = n_server_missing == 0
    server_cached = sum(r["tokens"]["server_reported_cached"] for r in rows) if has_server else 0
    ctx_share = ends.get("context_limit", 0) / m if m else None
    wall_share = ends.get("infra_timeout", 0) / m if m else None
    hit = tok["input_cached"] / (tok["input_cached"] + tok["input_uncached"]) if tok["input_cached"] else 0.0
    return {
        "n_attempts": n,
        "span_s": span,
        "attempts_per_hour": n / span * 3600 if span else None,
        "n_error": n - m,
        "resolve_rate": sum(bool(r["resolved"]) for r in scored) / m if m else None,
        "flagged_share": sum(sum(v) for v in by_profile.values()) / m if m else None,
        "flagged_share_by_profile": {p: sum(v) / len(v) for p, v in sorted(by_profile.items())},
        "end_reason_share": {k: v / n for k, v in sorted(ends.items())},
        "context_limit_check": {"share": ctx_share, "over_10pct": ctx_share is not None and ctx_share > 0.10},
        "wall_clock_check": {"share": wall_share, "over_2pct": wall_share is not None and wall_share > 0.02},
        # Per turn: a malformed turn executes no call, so a per-call rate can exceed 1.
        "malformed_per_turn": {k: sum(r["malformed"][k] for r in rows) / turns if turns else 0.0 for k in MALFORMED},
        "tokens_per_attempt": {k: v / n for k, v in tok.items()},
        # Prefill the server actually did: all input minus what it reported as cached.
        "realized_uncached_per_attempt": ((tok["input_cached"] + tok["input_uncached"] - server_cached) / n
                                          if has_server else None),
        "gen_seconds_per_attempt": sum(r.get("gen_seconds", 0.0) for r in rows) / n,
        "prefix_cache_hit_rate_ideal": hit,
        "prefix_cache_flag": hit < HERO_PREFIX_HIT - 0.12,
        "n_server_cached_missing": n_server_missing,
        "server_vs_ideal_cached": server_cached / tok["input_cached"] if tok["input_cached"] and has_server else None,
        "tool_seconds_per_attempt": sum(r["tool_seconds"] for r in rows) / n,
        "container_vcpus_busy": sum(r["container_cpu_seconds"] for r in rows) / span if span else None,
        "concurrency": Counter(r["concurrency"] for r in rows).most_common(1)[0][0],
    }


def project_hours(stats, rates, n_attempts, concurrency, realized=True):
    """Instance-hours for n attempts from aggregate token rates (not from one wave's wall clock).

    realized=True charges prefill for what the server actually recomputed (when it reported its cache
    counts); realized=False uses the ideal split under full prefix reuse, a lower bound.

    rates are aggregate tokens per second with all slots busy, so server time is already shared
    across slots; tool time runs in parallel across `concurrency` containers. Following docs/04
    ("tokens per attempt / aggregate token rate plus tool time"), the two are added, which is
    conservative: in practice they partly overlap.
    """
    t = stats["tokens_per_attempt"]
    uncached = stats["realized_uncached_per_attempt"] if realized and stats["realized_uncached_per_attempt"] \
        is not None else t["input_uncached"]
    per_attempt = t["output"] / rates["decode_tok_s"] + uncached / rates["prefill_tok_s"]
    server_h = n_attempts * per_attempt / 3600
    tool_h = n_attempts * stats["tool_seconds_per_attempt"] / concurrency / 3600
    return {"server_hours": server_h, "tool_hours": tool_h, "hours": server_h + tool_h}


def _rate(extras, runtime, profile):
    rates = extras.get("rates", {}).get(runtime, {})
    return rates.get(profile) or rates.get(RATE_FALLBACK.get(profile, profile))


def projections(stats, extras):
    """Projected instance-hours per P0 step at the point and upper attempt counts, plus the energy sub-run.

    Rates are keyed by (runtime, profile): extras["rates"][runtime][profile] = {decode_tok_s, prefill_tok_s}.
    A step is projected only when every profile in it has a rate (its own or the fallback in RATE_FALLBACK).
    """
    out, missing = {}, []
    for step, (runtime, profiles) in P0_STEPS.items():
        if runtime not in stats:
            missing.append(f"{step}: no {runtime} attempts in the pilot")
            continue
        s = stats[runtime]
        rows = {}
        for profile, (point, upper) in profiles.items():
            rate = _rate(extras, runtime, profile)
            if rate is None:
                missing.append(f"{step}: no rate for {runtime} {profile}")
                break
            rows[profile] = {
                "rate_from": profile if extras["rates"][runtime].get(profile) else RATE_FALLBACK[profile],
                "point_hours": project_hours(s, rate, point, s["concurrency"])["hours"],
                "upper_hours": project_hours(s, rate, upper, s["concurrency"])["hours"],
                "ideal_cache_hours": project_hours(s, rate, point, s["concurrency"], realized=False)["hours"],
            }
        else:
            out[step] = {"runtime": runtime, "profiles": rows,
                         "point_hours": sum(v["point_hours"] for v in rows.values()),
                         "upper_hours": sum(v["upper_hours"] for v in rows.values()),
                         "ideal_cache_hours": sum(v["ideal_cache_hours"] for v in rows.values())}
    if "llama.cpp" in stats:
        s = stats["llama.cpp"]
        per = s["gen_seconds_per_attempt"] + s["tool_seconds_per_attempt"] + SUB_RUN["settle_s"]
        out["energy_sub_run"] = {"runtime": "llama.cpp", "serial": True, "attempts": SUB_RUN["attempts"],
                                 "point_hours": SUB_RUN["attempts"] * per / 3600,
                                 "upper_hours": SUB_RUN["attempts"] * per / 3600,
                                 "ideal_cache_hours": SUB_RUN["attempts"] * per / 3600}
    if out:
        out["total"] = {k: sum(v[k] for v in out.values()) for k in ("point_hours", "upper_hours", "ideal_cache_hours")}
    out["not_included"] = NOT_INCLUDED + missing
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(stats, probe, extras, projections):
    lines = ["# Pilot exit report", ""]
    lines += ["## Attempts per hour", "", ("Resolve, limit and flagged shares leave out end_reason error rows "
              "(rerun as fresh attempts)."), "",
              "| runtime | attempts | errors | per hour | resolve rate | flagged | concurrency |",
              "|---|---|---|---|---|---|---|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | {s['n_attempts']} | {s['n_error']} | {s['attempts_per_hour']:.1f} "
                     f"| {_pct(s['resolve_rate'])} | {_pct(s['flagged_share'])} | {s['concurrency']} |")
    lines += ["", "## Test-file tripwire", "", "Share of attempts with grading.test_files_touched, by profile.", "",
              "| runtime | profile | flagged |", "|---|---|---|"]
    for rt, s in stats.items():
        lines += [f"| {rt} | {p} | {_pct(v)} |" for p, v in s["flagged_share_by_profile"].items()]
    lines += ["", "## Limits hit", "", ("| runtime | end reasons | 64K context (owner check at 10%) "
              "| wall clock (cap check at 2%) |"), "|---|---|---|---|"]
    for rt, s in stats.items():
        ends = ", ".join(f"{k} {_pct(v)}" for k, v in s["end_reason_share"].items())
        ctx, wall = s["context_limit_check"], s["wall_clock_check"]
        lines.append(f"| {rt} | {ends} | {_pct(ctx['share'])}{' **over**' if ctx['over_10pct'] else ''} "
                     f"| {_pct(wall['share'])}{' **over**' if wall['over_2pct'] else ''} |")
    lines += ["", "## Malformed calls", "", "Per turn, by category (a malformed turn executes no call).", "",
              "| runtime | " + " | ".join(MALFORMED) + " |", "|---" * (len(MALFORMED) + 1) + "|"]
    for rt, s in stats.items():
        lines.append(f"| {rt} | " + " | ".join(_pct(s["malformed_per_turn"][k]) for k in MALFORMED) + " |")
    lines += ["", "## Tokens and prefix cache", "",
              ("| runtime | output | uncached input | cached input | ideal hit rate (flag below 80%) "
              "| server / ideal cached | server count missing "
              "| tool s | vCPUs busy |"), "|---|---|---|---|---|---|---|---|---|"]
    for rt, s in stats.items():
        t = s["tokens_per_attempt"]
        flag = " **below HERO's ~92%**" if s["prefix_cache_flag"] else ""
        ratio = "n/a" if s["server_vs_ideal_cached"] is None else f"{s['server_vs_ideal_cached']:.2f}"
        busy = "n/a" if s["container_vcpus_busy"] is None else f"{s['container_vcpus_busy']:.1f}"
        lines.append(f"| {rt} | {t['output']:.0f} | {t['input_uncached']:.0f} | {t['input_cached']:.0f} "
                     f"| {_pct(s['prefix_cache_hit_rate_ideal'])}{flag} | {ratio} | {s['n_server_cached_missing']} "
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
        lines += ["", "## Projected instance-hours", "",
                  ("Point and upper attempt counts from 04, prefill at the realized cache split; the ideal-cache "
                  "column is a lower bound."), "",
                  "| step | runtime | point h | upper h | ideal-cache h |", "|---|---|---|---|---|"]
        for step, v in projections.items():
            if step in ("total", "not_included"):
                continue
            lines.append(f"| {step} | {v['runtime']} | {v['point_hours']:.1f} | {v['upper_hours']:.1f} "
                         f"| {v['ideal_cache_hours']:.1f} |")
        if "total" in projections:
            t = projections["total"]
            lines.append(f"| total | | {t['point_hours']:.1f} | {t['upper_hours']:.1f} | {t['ideal_cache_hours']:.1f} |")
        lines += ["", "Not included in the total:", ""] + [f"- {x}" for x in projections["not_included"]]
    missing = [k for k in EXTRAS if k not in extras]
    lines += ["", "## Not yet provided", ""]
    lines += [f"- `{k}`" for k in missing] or ["Nothing: every section is filled."]
    for k in EXTRAS[1:]:
        if k in extras:
            lines += ["", f"`{k}`: `{extras[k]}`"]
    return "\n".join(lines) + "\n"
