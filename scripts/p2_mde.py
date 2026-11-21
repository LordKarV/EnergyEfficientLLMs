"""Simulate the P2 minimum detectable effect for P0b criterion 3 and the guard's resolve-rate SE (docs/04).

    python scripts/p2_mde.py outputs/attempts/<P0b run>.jsonl ... [--est-joules outputs/p0a/est_joules.jsonl] \
        [--profile BF16] [--ratios 0.8 0.9 ...] [--n-sim 500] [--n-boot 1000] [--seed 0] [--guard-margin M] \
        [--out outputs/p0b/mde.json]

Per-task pass rates come from the heldout100 rows at --profile (P0b's BF16 by default; Q4_K_M for P0a's runs,
whose file is then passed too). Those rows must all come from one runtime: P0a also runs BF16 on heldout100, on
llama.cpp, so passing P0a's file with --profile BF16 fails rather than pooling its attempts into P0b's vLLM
counts (docs/04 P0b criterion 3 reads P0b's 16 attempts per task). Costs come from the pool200 rows at --pool-profile, split by outcome: each
attempt's est_joules_q4 from est_joules.jsonl when --est-joules is given, else its processed tokens (output plus
every input token), which the output names. An attempt is resolved when resolved is true and end_reason is
"submitted", as P2 counts it; end_reason "error" rows are left out.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.energy_model import pool_records
from qtrace.p2_analysis import MDE_POWER, MDE_RATIOS, mde


def resolved(rec):
    if type(rec.get("resolved")) is not bool:
        raise ValueError(f"attempt {rec.get('attempt_id')!r}: resolved must be true or false")
    return rec["resolved"] and rec["end_reason"] == "submitted"


def heldout_tasks(rows, profile, lst="heldout100"):
    acc = {}
    rows = [r for r in rows if r.get("list") == lst and r.get("profile") == profile and r["end_reason"] != "error"]
    runtimes = sorted({r["runtime"] for r in rows})
    if len(runtimes) > 1:
        raise ValueError(f"{lst} attempts at {profile} come from runtimes {runtimes}; pass the files of one runtime "
                         "(P0b's vLLM run for BF16, P0a's llama.cpp runs for Q4_K_M)")
    for r in rows:
        a = acc.setdefault(r["task_id"], {"repo": r["repo"], "successes": 0, "attempts": 0})
        a["successes"] += int(resolved(r))
        a["attempts"] += 1
    if not acc:
        raise ValueError(f"no {lst} attempts at {profile}")
    return [acc[t] for t in sorted(acc)]


def pool_costs(rows, profile, est_joules=None):
    kept, _, _ = pool_records(rows)
    kept = [r for r in kept if r["profile"] == profile]
    if est_joules is not None:
        missing = [r["attempt_id"] for r in kept if r["attempt_id"] not in est_joules]
        if missing:
            raise ValueError(f"{len(missing)} pool attempts have no est_joules row, e.g. {missing[:3]}")
    cost = {True: [], False: []}
    for r in kept:
        cost[resolved(r)].append(est_joules[r["attempt_id"]] if est_joules is not None
                                 else sum(r["tokens"][k] for k in ("output", "input_uncached", "input_cached")))
    return cost[True], cost[False]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--est-joules", help="est_joules.jsonl from scripts/p0a_energy_model.py")
    ap.add_argument("--profile", default="BF16", help="profile of the heldout100 pass rates")
    ap.add_argument("--pool-profile", default="BF16", help="profile of the pool200 cost rows")
    ap.add_argument("--ratios", type=float, nargs="+", default=list(MDE_RATIOS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-sim", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-margin", type=float)
    ap.add_argument("--out", default="outputs/p0b/mde.json")
    args = ap.parse_args(argv)

    rows = [r for p in args.attempts for r in read_jsonl(p)]
    est = None if args.est_joules is None else {e["attempt_id"]: e["est_joules_q4"] for e in read_jsonl(args.est_joules)}
    try:
        tasks = heldout_tasks(rows, args.profile)
        cr, cf = pool_costs(rows, args.pool_profile, est)
        out = mde(tasks, cr, cf, ratios=args.ratios, seeds=args.seeds, n_sim=args.n_sim, n_boot=args.n_boot,
                  alpha=args.alpha, seed=args.seed, guard_margin=args.guard_margin)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"p2_mde: {e!r}") from e
    out["cost"]["source"] = "est_joules_q4" if est is not None else "processed_tokens"
    out["inputs"] = {"attempts": args.attempts, "est_joules": args.est_joules, "profile": args.profile,
                     "pool_profile": args.pool_profile}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    se = out["guard_se"]
    print(f"{args.profile} mean pass@1 {out['mean_pass1']:.3f}, expected resolves per arm-seed run "
          f"{out['expected_resolves_per_arm_seed']:.1f} on {out['n_tasks']} tasks")
    for p in out["power"]:
        print(f"  true JPR ratio {p['ratio']:.3f}: power {p['power']:.3f}"
              + (f", with guard {p['power_with_guard']:.3f}" if "power_with_guard" in p else ""))
    print(f"MDE at 80% power: ratio {out['mde_ratio_80']} (cost: {out['cost']['source']}); se_rr {se['se_rr']:.4f}, "
          f"se_rr_diff {se['se_rr_diff']:.4f} ({se['formula']})")
    print(f"guard margin = k x se_rr_diff = k x {se['margin_se']:.4f} (the guard bounds the paired difference; "
          "k is the owner's decision)")
    if args.guard_margin is not None:
        print(f"MDE at 80% power with the guard at margin {args.guard_margin}: ratio {out['mde_ratio_80_with_guard']}")
        if out["guard_pass_rate"] < MDE_POWER:
            print(f"  the guard alone passes in {out['guard_pass_rate']:.3f} of simulated studies, so power with the "
                  "guard stays below 0.8 at every ratio")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
