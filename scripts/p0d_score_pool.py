"""P0d step 1: score the pool's successful BF16 runs against the Q3_K_M and Q4_K_M probes (docs/04, P0d).

    python scripts/p0d_score_pool.py outputs/attempts/*.jsonl --manifest outputs/p0/manifest.json \
        --kl-score bin/kl_score --gguf-p bf16.gguf --gguf-q3 q3_k_m.gguf --gguf-q4 q4_k_m.gguf \
        [--lists pool200] [--batch 25]

Every resolved BF16 attempt on --lists whose run_id is one of the --manifest's P0b runs is scored
(the manifest of scripts/p0_analyze.py); P0c's BF16 runs on pool200 are a different draw. An attempt
that ended in end_reason error is an infrastructure fault, rerun as a fresh attempt (docs/04), so it
is skipped and counted, as qtrace.p1_select.candidates drops it. A run flagged by the test-file tripwire is skipped and counted, because it can never be a selection
candidate (docs/04, "Test-file tripwire"). Held-out lists are refused.

Each (attempt, probe) becomes one row of --out: attempt_id, task_id, probe, gguf_p, gguf_q and the
keys of qtrace.kl.score_stats. One helper call scores --batch episodes, so each load of the two GGUF
files is shared by many runs, and the rows are appended after every call. A restart skips the pairs
already written. A torn last line left by a crash is cut off and that run is scored again.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.common import append_jsonl, read_jsonl
from qtrace.kl_gguf import score_trajectories_gguf
from qtrace.p0d import HELD_OUT_LISTS, flagged

EPISODE_KEYS = ("input_ids", "model_mask", "tool_call_positions", "command_spans", "turn_spans")


def pool_runs(attempt_paths, lists, run_ids):
    """(record, absolute episode path) for each run to score, the flagged count and the error count excluded."""
    runs, seen, n_flagged, n_error = [], set(), 0, 0
    for path in attempt_paths:
        base = os.path.dirname(os.path.abspath(path))
        for r in read_jsonl(path):
            missing = [k for k in ("attempt_id", "run_id", "task_id", "list", "profile", "end_reason", "resolved")
                       if k not in r]
            if missing:
                raise ValueError(f"{path}: attempt {r.get('attempt_id')!r} lacks {missing}")
            if r["run_id"] not in run_ids or r["profile"] != "BF16" or r["list"] not in lists:
                continue
            if r["end_reason"] == "error":
                n_error += 1
                continue
            if not r["resolved"]:
                continue
            if flagged(r):
                n_flagged += 1
                continue
            if r["attempt_id"] in seen:
                raise ValueError(f"duplicate attempt_id {r['attempt_id']!r} (is an attempts file passed twice?)")
            seen.add(r["attempt_id"])
            if "episode_path" not in r:
                raise ValueError(f"{path}: attempt {r['attempt_id']!r} has no episode_path")
            runs.append((r, os.path.join(base, r["episode_path"])))  # join keeps an absolute path as is
    return runs, n_flagged, n_error


def load_episode(path):
    """An episode .npz as the episode dict of score_trajectories_gguf, checked against the agreed format."""
    with np.load(path) as z:
        missing = [k for k in EPISODE_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        ids, mask = z["input_ids"], z["model_mask"]
        if ids.ndim != 1 or mask.shape != ids.shape or mask.dtype != np.bool_:
            raise ValueError(f"{path}: need input_ids [n] and a bool model_mask [n]")
        spans = {}
        for k in ("command_spans", "turn_spans"):
            a = z[k].reshape(0, 2) if z[k].size == 0 else z[k]  # an empty list may be saved with shape (0,)
            if a.ndim != 2 or a.shape[1] != 2 or (a[:, 0] < 0).any() or (a[:, 0] > a[:, 1]).any() or \
                    (a[:, 1] > len(ids)).any():
                raise ValueError(f"{path}: {k} must be [m, 2] (start, end) spans inside the episode")
            spans[k] = [tuple(s) for s in a.tolist()]
        calls = z["tool_call_positions"].reshape(-1)
        if ((calls < 0) | (calls >= len(ids))).any():
            raise ValueError(f"{path}: tool_call_positions outside the episode")
        return {"input_ids": torch.from_numpy(ids.astype(np.int64)).reshape(1, -1),
                "model_mask": torch.from_numpy(mask.copy()), "tool_call_positions": calls.tolist(), **spans}


def resume(out, gguf_p, probes):
    """(attempt_id, probe) pairs already in out.

    Every row is checked before anything is written, so a wrong --out is refused, not cut. A row
    without probe or the GGUF names means out is not a scores file. A row scored with a GGUF of
    another file name raises: one probe's ranking must come from one file. Names, not full paths, are
    compared, so a run can move between machines. A last line without its newline is then kept if it
    parses (the crash fell between a row and its newline) and cut if not, so that run is scored again.
    """
    if not os.path.exists(out):
        return set()
    with open(out, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n") + 1
    rows = [json.loads(line) for line in data[:cut].splitlines() if line.strip()]
    try:
        tail = json.loads(data[cut:]) if data[cut:].strip() else None
    except ValueError:
        tail = None
    rows += [tail] if tail is not None else []
    done = set()
    for r in rows:
        missing = [k for k in ("attempt_id", "probe", "gguf_p", "gguf_q") if not isinstance(r, dict) or k not in r]
        if missing:
            raise ValueError(f"{out} is not a P0d scores file (a row lacks {missing}); use another --out")
        want = probes.get(r["probe"])
        if want is None or os.path.basename(r["gguf_q"]) != os.path.basename(want) or \
                os.path.basename(r["gguf_p"]) != os.path.basename(gguf_p):
            raise ValueError(f"{out}: {r['attempt_id']} {r['probe']} was scored with a different GGUF "
                             f"({r['gguf_p']}, {r['gguf_q']}); use another --out")
        done.add((r["attempt_id"], r["probe"]))
    if cut < len(data):
        with open(out, "rb+") as f:
            if tail is None:
                f.truncate(cut)
                print(f"cut a torn last line ({len(data) - cut} bytes) from {out}")
            else:
                f.seek(0, os.SEEK_END)
                f.write(b"\n")  # or the next appended row would join this line
    return done


def score_pool(attempt_paths, binary, gguf_p, probes, lists, out, batch=25, n_batch=512, *, run_ids):
    """Score every pool run of the P0b run_ids under each probe ({probe name: GGUF path}), appending to out."""
    bad = sorted(set(lists) & set(HELD_OUT_LISTS))
    if bad:
        raise ValueError(f"{bad} are held-out lists and never enter a pool")
    runs, n_flagged, n_error = pool_runs(attempt_paths, lists, run_ids)
    done = resume(out, gguf_p, probes)
    print(f"{len(runs)} pool runs, {n_flagged} flagged by the test-file tripwire excluded, "
          f"{n_error} infrastructure errors excluded, {len(done)} rows already in {out}")
    scored = 0
    for probe, gguf_q in probes.items():
        todo = [(r, ep) for r, ep in runs if (r["attempt_id"], probe) not in done]
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            stats = score_trajectories_gguf(binary, gguf_p, gguf_q, [load_episode(ep) for _, ep in chunk],
                                            n_batch=n_batch)
            append_jsonl(out, [{"attempt_id": r["attempt_id"], "task_id": r["task_id"], "probe": probe,
                                "gguf_p": gguf_p, "gguf_q": gguf_q, **st} for (r, _), st in zip(chunk, stats)])
            scored += len(chunk)
            print(f"{probe}: {s + len(chunk)}/{len(todo)}")
    return {"kept": len(runs), "flagged_excluded": n_flagged, "infra_error_excluded": n_error,
            "already_scored": len(done), "scored": scored}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--manifest", required=True, help="scripts/p0_analyze.py's manifest; its P0b run ids are scored")
    ap.add_argument("--kl-score", required=True, help="the kl_score helper binary (qtrace/kl_gguf.py)")
    ap.add_argument("--gguf-p", required=True, help="BF16 GGUF")
    ap.add_argument("--gguf-q3", required=True)
    ap.add_argument("--gguf-q4", required=True)
    ap.add_argument("--lists", nargs="+", default=["pool200"])
    ap.add_argument("--batch", type=int, default=25, help="episodes per helper call (one model load each)")
    ap.add_argument("--n-batch", type=int, default=512, help="llama.cpp batch size")
    ap.add_argument("--out", default="outputs/p0d/scores.jsonl")
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        p0b = json.load(f).get("P0b")
    if not p0b:
        ap.error(f"{args.manifest} names no P0b run ids")
    report = score_pool(args.attempts, args.kl_score, args.gguf_p, {"Q3_K_M": args.gguf_q3, "Q4_K_M": args.gguf_q4},
                        args.lists, args.out, batch=args.batch, n_batch=args.n_batch,
                        run_ids=set(p0b))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
