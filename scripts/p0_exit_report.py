"""Write the pilot exit report (docs/04, P0b pilot) to outputs/p0/exit_report.{md,json}.

    python scripts/p0_exit_report.py outputs/attempts/*.jsonl --probe outputs/probe/probe.jsonl \
        --extras outputs/p0/pilot_extras.json

The extras JSON holds what attempt records can't: {"rates": {runtime: {"decode_tok_s", "prefill_tok_s"}},
"peak_gpu_gib": {runtime: GiB}, "energy_counter": {"readable", "monotonic", "update_period_s",
"idle_w": {...}}, "replay_match_rate": float, "sampler_echo_ok": {runtime: bool},
"per_turn_energy_ok": bool, "concurrency": {"containers", "vcpus_per_container"}}.
Missing keys are listed in the report.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import read_jsonl
from qtrace.exit_report import attempt_stats, projections, render_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--extras", default=None)
    ap.add_argument("--out-dir", default="outputs/p0")
    args = ap.parse_args()

    rows = [r for path in args.attempts for r in read_jsonl(path)]
    probe = read_jsonl(args.probe) if args.probe and os.path.exists(args.probe) else []
    extras = {}
    if args.extras and os.path.exists(args.extras):
        with open(args.extras) as f:
            extras = json.load(f)
    stats = attempt_stats(rows)
    proj = projections(stats, extras)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "exit_report.json"), "w") as f:
        json.dump({"stats": stats, "projections": proj, "extras": extras, "probe": probe}, f, indent=1)
    md = render_markdown(stats, probe, extras, proj)
    with open(os.path.join(args.out_dir, "exit_report.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
