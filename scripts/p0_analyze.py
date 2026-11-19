"""Compute the P0 gates (docs/04) from attempt records and write outputs/p0/gates.json.

    python scripts/p0_analyze.py outputs/attempts/*.jsonl [--inrepo-size 300] [--n-boot 5000]

Each gate is computed only when its data is present: P0b needs vLLM BF16 attempts on pool200 and
heldout100; P0a needs llama.cpp BF16 and Q4_K_M on heldout100; P0c needs llama.cpp BF16 and the
quantized profile on pool200, plus P0b's counts to bin tasks.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0_analysis import p0a_primary, p0a_sensitivity, p0b_report, p0c_breakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--inrepo-size", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default="outputs/p0/gates.json")
    args = ap.parse_args()

    rows = [r for path in args.attempts for r in read_jsonl(path)]
    have = Counter((r["runtime"], r["profile"], r["list"]) for r in rows)
    print("attempts by (runtime, profile, list):", dict(have))
    gates = {}

    if have[("vllm", "BF16", "pool200")]:
        gates["P0b"] = p0b_report(rows, inrepo_size=args.inrepo_size)
    if have[("llama.cpp", "BF16", "heldout100")] and have[("llama.cpp", "Q4_K_M", "heldout100")]:
        gates["P0a"] = {"primary": p0a_primary(rows, n_boot=args.n_boot, seed=args.seed),
                        "sensitivity": p0a_sensitivity(rows)}
    if "P0b" in gates:
        counts = Counter(r["task_id"] for r in rows
                         if r["runtime"] == "vllm" and r["list"] == "pool200" and r["resolved"])
        counts = {t: counts.get(t, 0) for t in {r["task_id"] for r in rows if r["list"] == "pool200"}}
        for profile in ("Q4_K_M", "Q3_K_M"):
            if have[("llama.cpp", "BF16", "pool200")] and have[("llama.cpp", profile, "pool200")]:
                gates.setdefault("P0c", {})[profile] = p0c_breakage(rows, counts, profile,
                                                                     n_boot=args.n_boot, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(gates, f, indent=1)
    for name, g in gates.items():
        print(name, json.dumps(g.get("pass") or g.get("primary", {}).get("verdict") or
                               {k: v["verdict"] for k, v in g.items()}))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
