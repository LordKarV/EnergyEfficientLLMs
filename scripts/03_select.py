"""Step 3: build one training set per arm from the scored pool.

Run:  python scripts/03_select.py --config configs/base.yaml
Prints a table so you can see at a glance how the arms differ in length and stability.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl, write_jsonl
from qtrace.select import ARMS, build_arm, group_candidates


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sel = cfg["select"]

    rows = read_jsonl(cfg["paths"]["scored"])
    by_problem = group_candidates(rows, sel["min_candidates"])
    print(f"{len(by_problem)} problems have at least {sel['min_candidates']} distinct candidate solutions")

    print(f"\n{'arm':16s} {'examples':>8s} {'tokens':>8s} {'kl_mean':>9s} {'entropy':>9s}")
    for arm in ARMS:
        chosen = build_arm(by_problem, arm, sel["alpha"], sel["seed"])
        write_jsonl(os.path.join(cfg["paths"]["arms_dir"], arm + ".jsonl"), chosen)
        print(f"{arm:16s} {len(chosen):8d} "
              f"{mean([c['n_tokens'] for c in chosen]):8.1f} "
              f"{mean([c['kl_mean'] for c in chosen]):9.4f} "
              f"{mean([c['entropy_fp'] for c in chosen]):9.4f}")


if __name__ == "__main__":
    main()
