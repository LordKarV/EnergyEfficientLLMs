"""Compute the P0 gates (docs/04) from attempt records and write outputs/p0/gates.json.

    python scripts/p0_analyze.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json --inrepo-size N

The manifest maps each gate to the run ids that belong to it, so a gate never mixes in another
run's attempts (the pilot, the energy sub-run, P0d continuations):
    {"P0b": ["run-..."], "P0a": ["run-..."], "P0c": ["run-..."]}
Records with end_reason "error" are dropped (04 reruns them as fresh attempts) and duplicate
attempt ids are dropped; both are counted per gate. Attempt counts per task and profile are
checked against the design (P0b 16, P0a 3, P0c 4).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0_analysis import clean, design_mismatches, p0a_factors, p0a_primary, p0b_report, p0c_breakage

DESIGN = {"P0b": 16, "P0a": 3, "P0c": 4}


def gate_rows(rows, run_ids, lst):
    picked, dropped = clean([r for r in rows if r.get("run_id") in set(run_ids)])
    return [r for r in picked if lst is None or r["list"] == lst], dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inrepo-size", type=int, required=True, help="set by the owner (04, Owner decisions)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default="outputs/p0/gates.json")
    args = ap.parse_args()

    rows = [r for path in args.attempts for r in read_jsonl(path)]
    manifest = json.load(open(args.manifest))
    gates = {}

    if "P0b" in manifest:
        sel, dropped = gate_rows(rows, manifest["P0b"], None)
        gates["P0b"] = {**p0b_report(sel, inrepo_size=args.inrepo_size), "dropped": dropped,
                        "design_mismatches": design_mismatches(sel, DESIGN["P0b"])}
    if "P0a" in manifest:
        sel, dropped = gate_rows(rows, manifest["P0a"], "heldout100")
        no_timeouts = [r for r in sel if r.get("end_reason") != "infra_timeout"]
        gates["P0a"] = {"primary": p0a_primary(sel, n_boot=args.n_boot, seed=args.seed),
                        "factors": p0a_factors(sel),
                        "sensitivity_excluding_infra_timeout": p0a_primary(no_timeouts, n_boot=args.n_boot,
                                                                           seed=args.seed),
                        "dropped": dropped, "design_mismatches": design_mismatches(sel, DESIGN["P0a"])}
    if "P0c" in manifest:
        if "P0b" not in gates:
            raise SystemExit("P0c bins tasks by P0b's success counts; add P0b's run ids to the manifest")
        counts = {t: v["successes"] for t, v in gates["P0b"]["per_task"].items()}
        sel, dropped = gate_rows(rows, manifest["P0c"], "pool200")
        gates["P0c"] = {p: p0c_breakage(sel, counts, p, n_boot=args.n_boot, seed=args.seed)
                        for p in ("Q4_K_M", "Q3_K_M") if any(r["profile"] == p for r in sel)}
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
