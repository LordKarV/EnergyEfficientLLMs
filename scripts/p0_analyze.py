"""Compute the P0 gates (docs/04) from attempt records and write outputs/p0/gates.json.

    python scripts/p0_analyze.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json --inrepo-size N

The manifest maps each gate to the run ids that belong to it, so a gate never mixes in another
run's attempts (the pilot, the energy sub-run, P0d continuations):
    {"P0b": ["run-..."], "P0a": ["run-..."], "P0c": ["run-..."]}
Records with end_reason "error" are dropped (04 reruns them as fresh attempts) and duplicate
attempt ids are dropped; both are counted per gate. Attempt counts per task and profile are
checked against the design (P0b 16, P0a 3 on heldout100 and verified100, P0c 4). Every gate also
reports its result excluding flagged attempts (test-file tripwire), the flagged share per profile, and
a sensitivity analysis excluding infra_timeout hits (04). P0a.secondary holds 04's secondaries: every
profile against BF16 on both lists, turns per resolved attempt, and a paired permutation test.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0_analysis import (
    clean,
    design_mismatches,
    flagged,
    flagged_share,
    p0a_factors,
    p0a_primary,
    p0a_secondary,
    p0b_report,
    p0c_breakage,
)

DESIGN = {"P0b": 16, "P0a": 3, "P0c": 4}


def gate_rows(rows, run_ids, lst):
    picked, dropped = clean([r for r in rows if r.get("run_id") in set(run_ids)])
    return [r for r in picked if lst is None or r["list"] == lst], dropped


def unflagged(rows):
    return [r for r in rows if not flagged(r)]


def no_timeouts(rows):
    return [r for r in rows if r["end_reason"] != "infra_timeout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inrepo-size", type=int, required=True, help="set by the owner (04, Owner decisions)")
    ap.add_argument("--min-candidates", type=int, default=4, help="P0b fallback 1 sets 2 (04, P0b)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default="outputs/p0/gates.json")
    args = ap.parse_args()

    rows = [r for path in args.attempts for r in read_jsonl(path)]
    with open(args.manifest) as f:
        manifest = json.load(f)
    gates = {}

    if "P0b" in manifest:
        sel, dropped = gate_rows(rows, manifest["P0b"], None)
        kw = {"inrepo_size": args.inrepo_size, "min_candidates": args.min_candidates, "n_sub": args.n_boot,
              "seed": args.seed}
        sens = p0b_report(no_timeouts(sel), **kw)
        gates["P0b"] = {**p0b_report(sel, **kw),
                        "flagged_share": flagged_share(sel),
                        "sensitivity_excluding_infra_timeout": {
                            k: v for k, v in sens.items()
                            if k not in ("per_task", "heldout_per_task", "tasks_below_design_attempts")},
                        "dropped": dropped, "design_mismatches": design_mismatches(sel, DESIGN["P0b"])}
    if "P0a" in manifest:
        both, dropped = gate_rows(rows, manifest["P0a"], None)
        sel = [r for r in both if r["list"] == "heldout100"]
        gates["P0a"] = {"primary": p0a_primary(sel, n_boot=args.n_boot, seed=args.seed),
                        "primary_excluding_flagged": p0a_primary(unflagged(sel), n_boot=args.n_boot, seed=args.seed),
                        "flagged_share": flagged_share(sel),
                        "factors": p0a_factors(sel),
                        "factors_excluding_flagged": p0a_factors(unflagged(sel)),
                        "sensitivity_excluding_infra_timeout": p0a_primary(no_timeouts(sel), n_boot=args.n_boot,
                                                                           seed=args.seed),
                        "secondary": p0a_secondary(both, n_boot=args.n_boot, seed=args.seed),
                        "dropped": dropped, "design_mismatches": design_mismatches(both, DESIGN["P0a"])}
    if "P0c" in manifest:
        if "P0b" not in gates:
            raise SystemExit("P0c bins tasks by P0b's success counts; add P0b's run ids to the manifest")
        counts = {t: v["successes"] for t, v in gates["P0b"]["per_task"].items()}
        sel, dropped = gate_rows(rows, manifest["P0c"], "pool200")
        kw = {"n_boot": args.n_boot, "seed": args.seed}
        gates["P0c"] = {p: {**p0c_breakage(sel, counts, p, **kw),
                            "excluding_flagged": p0c_breakage(unflagged(sel), counts, p, **kw),
                            "sensitivity_excluding_infra_timeout": p0c_breakage(no_timeouts(sel), counts, p, **kw)}
                        for p in ("Q4_K_M", "Q3_K_M") if any(r["profile"] == p for r in sel)}
        gates["P0c"]["flagged_share"] = flagged_share(sel)
        gates["P0c"]["dropped"] = dropped
        gates["P0c"]["design_mismatches"] = design_mismatches(sel, DESIGN["P0c"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(gates, f, indent=1, allow_nan=False)
    for name, g in gates.items():
        verdicts = g.get("pass") or g.get("primary", {}).get("verdict") or \
            {k: v["verdict"] for k, v in g.items() if isinstance(v, dict) and "verdict" in v}
        print(name, json.dumps(verdicts))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
