"""Fit the P0a per-turn energy model (docs/04) and estimate Q4_K_M joules for every pool attempt.

    python scripts/p0a_energy_model.py --subrun outputs/attempts/sub-*.jsonl \
        --pool outputs/attempts/p0b-*.jsonl [--windows outputs/p0a/q4_windows.jsonl] --out outputs/p0a

--subrun: attempt files holding the energy sub-run; only concurrency-1 attributable attempts are fitted,
          per profile, and Q4_K_M must be among them.
--pool:   candidate pool attempt files (P0b); held-out lists or repos fail. end_reason "error" is dropped.
          Every attempt gets an estimate; the pick-agreement checks use the arms' candidates (resolved,
          unflagged BF16 runs) on tasks with at least --min-candidates of them (4; the P0b fallback is 2).
--windows: the Q4_K_M block's 1 s llama-server metrics windows {t, joules, prompt_tokens, predicted_tokens}.

Writes energy_model.json ({"profiles": {...}, "fewest_joules_fallback": bool, ...}) and
est_joules.jsonl (one {attempt_id, est_joules_q4} per pool attempt, written even under the fallback;
selection obeys the flag). The fallback reads the nested LOAO error of the raw Q4_K_M fit.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl, write_jsonl
from qtrace.energy_model import (
    AGREEMENT_MIN,
    FALLBACK_MAX_REL_ERR,
    MIN_CANDIDATES,
    RANK_PROFILE,
    RANK_TARGET,
    TARGETS,
    TERMS,
    estimate_attempt_joules,
    estimate_batched_joules,
    fit_attempts,
    fit_batched,
    pick_agreement,
    pool_records,
    profile_report,
)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subrun", nargs="+", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--windows")
    ap.add_argument("--out", default="outputs/p0a")
    ap.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()

    by_profile, excluded = fit_attempts([r for p in args.subrun for r in read_jsonl(p)])
    if RANK_PROFILE not in by_profile:
        raise SystemExit(f"no usable {RANK_PROFILE} sub-run attempts; runs are ranked with its coefficients")
    profiles = {p: profile_report(a, n_boot=args.n_boot, seed=args.seed) for p, a in sorted(by_profile.items())}
    model = profiles[RANK_PROFILE][RANK_TARGET]
    pool, n_error = pool_records([r for p in args.pool for r in read_jsonl(p)])
    q4 = {r["attempt_id"]: estimate_attempt_joules(r, model) for r in pool}
    by_q4 = lambda r: q4[r["attempt_id"]]

    bf16 = None
    if "BF16" in profiles:
        bf16 = pick_agreement(pool, lambda r: estimate_attempt_joules(r, profiles["BF16"][RANK_TARGET]), by_q4,
                              args.min_candidates)
    batched = None
    if args.windows:
        fit = fit_batched(read_jsonl(args.windows))
        agree = pick_agreement(pool, lambda r: estimate_batched_joules(r, fit), by_q4, args.min_candidates)
        share = agree["share_same_pick"]
        batched = {**fit, "agreement_with_c1": agree, "below_85pct": None if share is None else share < AGREEMENT_MIN}

    report = {
        "ranking": {"profile": RANK_PROFILE, "target": RANK_TARGET},
        "fewest_joules_fallback": model["fallback"],
        "fallback_rule": f"nested LOAO median absolute relative error on attempt joules > {FALLBACK_MAX_REL_ERR} "
                         f"({RANK_PROFILE}, {RANK_TARGET}); fewest_joules then falls back to fewest_turns",
        "fallback_error": model["cv"]["nested_loao"],
        "terms_available": list(TERMS),
        "profiles": profiles,
        "excluded_subrun_attempts": excluded,
        "pool": {"n_attempts": len(pool), "n_dropped_error": n_error},
        "bf16_vs_q4": bf16,
        "batched": batched,
        "n_boot": args.n_boot, "seed": args.seed,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "energy_model.json"), "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    write_jsonl(os.path.join(args.out, "est_joules.jsonl"),
                [{"attempt_id": aid, "est_joules_q4": j} for aid, j in q4.items()])

    print(f"{'profile':8} {'target':16} {'nested':>7} {'loao':>7} {'pairs':>6}  terms")
    for p, rep in profiles.items():
        for t in TARGETS:
            r = rep[t]
            pairs = f"{r['pairs']['n_correct']}/{r['pairs']['n_pairs']}"
            print(f"{p:8} {t:16} {pct(r['cv']['nested_loao']):>7} {pct(r['cv']['loao']):>7} {pairs:>6}  "
                  + "+".join(r["terms"]))
    print(f"fewest_joules_fallback: {model['fallback']} ({RANK_PROFILE} {RANK_TARGET} nested LOAO "
          f"{pct(model['cv']['nested_loao'])} vs {pct(FALLBACK_MAX_REL_ERR)})")
    if bf16:
        print(f"BF16 vs {RANK_PROFILE} same cheapest run: {pct(bf16['share_same_pick'])} of {bf16['n_tasks']} tasks, "
              f"mean Spearman {'n/a' if bf16['mean_spearman'] is None else round(bf16['mean_spearman'], 3)}")
    if batched:
        a = batched["agreement_with_c1"]
        print(f"batched vs concurrency-1 same cheapest run: {pct(a['share_same_pick'])} of {a['n_tasks']} tasks"
              f"{' (below 85%: name the regime)' if batched['below_85pct'] else ''}")
    print("wrote", os.path.join(args.out, "energy_model.json"), "and est_joules.jsonl,", len(q4), "rows")


if __name__ == "__main__":
    main()
