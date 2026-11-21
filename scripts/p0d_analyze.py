"""P0d steps 2-3: choose the stability score, pick the replay pairs and, given continuation results, run
the DiD test (docs/04, P0d).

    python scripts/p0d_analyze.py outputs/attempts/<P0b run>.jsonl ... --scores outputs/p0d/scores.jsonl \
        [--est-joules outputs/p0a/est_joules.jsonl] [--power-rates BF16 Q3] [--continuations FILE]

Writes outputs/p0d/score_choice.json: the chosen score (or null), every candidate's numbers, the
replay pairs under the chosen score and, with --power-rates, the simulated power of the DiD test at
4 and 8 seeds per prefix. --power-rates are the expected BF16 and Q3_K_M continuation success rates
(from the pilot and P0c). With --continuations (rows {task_id, prefix, profile, successes, attempts,
replay_valid}) it also writes outputs/p0d/did.json. Without --est-joules the joules check does not
run and the rule stays undecided unless a candidate fails another check first.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.p0d import CANDIDATES, CONFOUNDS, POOL_LISTS, choose_score, did_test, join_runs, power, select_pairs

POWER_EFFECTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
POWER_SEEDS = (4, 8)  # docs/04: 4 seeded attempts per prefix, extra budget goes to 8


def fmt(x):
    return "n/a" if x is None else f"{x:+.2f}"


def print_table(choice):
    cols = ("turns", "tokens", "joules", "entropy", "split-half", "agreement")
    print(f"{'candidate':<12}" + "".join(f"{c:>11}" for c in cols) + "  result")
    for name in CANDIDATES:
        c = choice["candidates"][name]
        vals = [c["against"][k].get("mean") for k in CONFOUNDS]
        vals += [c["split_half"].get("spearman_brown_mean"), c["probe_agreement"].get("mean")]
        result = "pass" if c["pass"] else f"fails {', '.join(c['failed'])}" if c["failed"] else \
            f"not run: {', '.join(c['not_run'])}"
        print(f"{name:<12}" + "".join(f"{fmt(v):>11}" for v in vals) + f"  {result}")
    print(choice["verdict"])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--scores", default="outputs/p0d/scores.jsonl")
    ap.add_argument("--lists", nargs="+", default=list(POOL_LISTS), help="the pool lists p0d_score_pool.py scored")
    ap.add_argument("--est-joules", help="outputs/p0a/est_joules.jsonl; omit until P0a's energy model exists")
    ap.add_argument("--continuations", help="continuation results, one row per (task, prefix, profile)")
    ap.add_argument("--power-rates", nargs=2, type=float, metavar=("BF16", "Q3"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out-dir", default="outputs/p0d")
    args = ap.parse_args(argv)

    attempts = [r for path in args.attempts for r in read_jsonl(path)]
    joules = read_jsonl(args.est_joules) if args.est_joules else None
    runs, info = join_runs(attempts, read_jsonl(args.scores), joules, lists=args.lists)
    choice = choose_score(runs, have_joules=joules is not None, missing_q3=info["n_eligible_without_q3"])
    pairs = select_pairs(runs, choice["chosen"]) if choice["chosen"] else None
    out = {**choice, "inputs": {**info, "scores": args.scores, "est_joules": args.est_joules}, "pairs": pairs}
    if args.power_rates and pairs and pairs["n_tasks"] >= 2:
        out["power"] = [power(*args.power_rates, n_tasks=pairs["n_tasks"], seeds=s, effects=POWER_EFFECTS,
                              n_boot=args.n_boot, seed=args.seed) for s in POWER_SEEDS]
    write_json(os.path.join(args.out_dir, "score_choice.json"), out)
    print_table(choice)
    if pairs:
        print(f"pairs: {pairs['n_tasks']} tasks, spread {pairs['spread']}")
    for p in out.get("power", []):
        print(f"power at {p['seeds']} seeds/prefix, {p['n_tasks']} tasks:",
              ", ".join(f"{e['effect']:.2f}: {e['power']:.2f}" for e in p["power"]), f"(MDE80 {p['mde_80']})")

    if args.continuations:
        if not pairs:
            raise SystemExit("no chosen score, so there are no replay pairs to test")
        cov = {p["task_id"]: p["diff"] for p in pairs["tasks"]}
        rows = read_jsonl(args.continuations)
        unknown = sorted({r["task_id"] for r in rows} - set(cov))
        if unknown:
            raise SystemExit(f"continuation tasks that are not replay pairs: {unknown[:5]}")
        did = did_test(rows, cov, n_boot=args.n_boot, seed=args.seed)
        did["pair_tasks_without_continuations"] = sorted(set(cov) - {r["task_id"] for r in rows})
        did["score"] = choice["chosen"]
        write_json(os.path.join(args.out_dir, "did.json"), did)
        adj = did["adjusted"]["estimate"] if did["adjusted"] else None
        print(f"DiD {did['did_mean']:+.3f} (p one-sided {did['p_one_sided']:.4f}, "
              f"upper95 {did['upper95_one_sided']:+.3f}), adjusted {fmt(adj)}, "
              f"pair drop rate {did['pair_drop_rate']:.2f}: {did['decision']}")


if __name__ == "__main__":
    main()
