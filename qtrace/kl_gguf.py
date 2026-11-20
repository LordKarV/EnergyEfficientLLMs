"""The KL scorer for GGUF models, run through llama.cpp.

P0d's probes are the `Q3_K_M` and `Q4_K_M` GGUF files that llama.cpp serves (docs/04), so the score is
computed in that runtime against the `BF16` GGUF, not against a PyTorch stand-in for the quantizer.
qtrace/kl_score.cpp does the model work and writes the per-position fields of kl.compare(); scoring
from there is kl.score_stats(), shared with the PyTorch path.

Build the helper once against a llama.cpp build tree (the same command works on Linux):

    LLAMA=path/to/llama.cpp
    c++ -std=c++17 -O3 qtrace/kl_score.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
        -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o kl_score
"""

import os
import subprocess
import tempfile

import numpy as np
import torch

from qtrace.kl import logit_index, score_stats

FIELDS = ("kl", "flip", "entropy_p", "nll_p", "nll_q")


def compare_gguf(binary, gguf_p, gguf_q, runs, n_batch=512):
    """Per-position stats with the keys of kl.compare(), one dict per (input_ids, idx) in runs.

    Both models load once for all runs. idx are logit positions from kl.logit_index().
    """
    with tempfile.TemporaryDirectory() as d:
        jobs = []
        for i, (input_ids, idx) in enumerate(runs):
            tok, pos, out = (os.path.join(d, f"{i}.{ext}") for ext in ("tok", "idx", "out"))
            np.asarray(input_ids, dtype=np.int32).reshape(-1).tofile(tok)
            np.asarray(idx, dtype=np.int32).tofile(pos)
            jobs.append((tok, pos, out))
        jobs_file = os.path.join(d, "jobs.txt")
        with open(jobs_file, "w") as f:
            f.writelines(" ".join(j) + "\n" for j in jobs)
        # stderr is kept for failures only: llama.cpp warns on every load about tensors it skips.
        r = subprocess.run([binary, gguf_p, gguf_q, jobs_file, str(n_batch)], capture_output=True, text=True,
                           check=False)
        if r.returncode:
            raise RuntimeError(f"kl_score exited with {r.returncode}:\n{r.stderr[-4000:]}")
        stats = []
        for (_, _, out), (_, idx) in zip(jobs, runs):
            rows = np.fromfile(out, dtype=np.float32).reshape(len(idx), len(FIELDS))
            stats.append({k: torch.from_numpy(rows[:, i].copy()) for i, k in enumerate(FIELDS)})
    return stats


def score_trajectories_gguf(binary, gguf_p, gguf_q, episodes, n_batch=512, entropy_threshold=None):
    """kl.score_trajectory() for each episode, with GGUF models.

    Each episode is a dict of score_trajectory's episode arguments: input_ids and model_mask, plus
    optionally tool_call_positions, command_spans and turn_spans.
    """
    idxs = [logit_index(e["model_mask"]) for e in episodes]
    stats = compare_gguf(binary, gguf_p, gguf_q, [(e["input_ids"], i) for e, i in zip(episodes, idxs)], n_batch)
    return [score_stats(s, i, e.get("tool_call_positions", ()), e.get("command_spans", ()), entropy_threshold,
                        e.get("turn_spans", ()))
            for s, i, e in zip(stats, idxs, episodes)]
