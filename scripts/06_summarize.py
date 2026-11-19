"""Step 6: collect every evaluation result into one table, with confidence intervals.

Accuracy differences of 1-2 points on 1,319 test problems are within noise, so
every accuracy gets a bootstrap 95% interval, and --compare runs a paired
bootstrap between two models on the same test problems.

Run:  python scripts/06_summarize.py --config configs/base.yaml
      python scripts/06_summarize.py --config configs/base.yaml --compare qtrace_seed0 shortest_seed0 --precision rtn3
"""

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import load_config, read_jsonl


def bootstrap_interval(values, n_resamples=2000, seed=0):
    """95% interval for the mean: resample the list with replacement many times."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def paired_bootstrap(a, b, n_resamples=2000, seed=0):
    """Compare two models on the same problems.

    a and b are 0/1 correctness lists in the same problem order. Returns the mean
    accuracy difference (a - b) and its 95% interval. If the interval excludes 0,
    the difference is unlikely to be test-set noise.
    """
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    return sum(diffs) / n, means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()
    eval_dir = load_config(args.config)["paths"]["eval_dir"]

    if args.compare:
        tag_a, tag_b = args.compare
        rows_a = {r["qid"]: r for r in read_jsonl(os.path.join(eval_dir, f"{tag_a}__{args.precision}.jsonl"))}
        rows_b = {r["qid"]: r for r in read_jsonl(os.path.join(eval_dir, f"{tag_b}__{args.precision}.jsonl"))}
        qids = sorted(set(rows_a) & set(rows_b))
        diff, low, high = paired_bootstrap(
            [int(rows_a[q]["correct"]) for q in qids],
            [int(rows_b[q]["correct"]) for q in qids],
        )
        print(f"{tag_a} - {tag_b} at {args.precision}: {100 * diff:+.2f} points "
              f"(95% interval {100 * low:+.2f} to {100 * high:+.2f}, n={len(qids)})")
        return

    print("| model | precision | accuracy (95% CI) | mean tokens | unfinished | J/problem | J/correct | real efficiency |")
    print("|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        with open(path) as f:
            s = json.load(f)
        correct = [int(r["correct"]) for r in read_jsonl(path[:-len(".json")] + ".jsonl")]
        low, high = bootstrap_interval(correct)

        def joules(key):
            return f"{s[key]:.2f}" if s.get(key) is not None and s["efficiency_is_real"] else "n/a"

        print(f"| {s['tag']} | {s['precision']} | {100 * s['accuracy']:.1f} ({100 * low:.1f}-{100 * high:.1f}) "
              f"| {s['mean_tokens']:.0f} | {100 * s['unfinished_rate']:.1f}% "
              f"| {joules('joules_per_problem')} | {joules('joules_per_correct')} "
              f"| {'yes' if s['efficiency_is_real'] else 'no'} |")


if __name__ == "__main__":
    main()
