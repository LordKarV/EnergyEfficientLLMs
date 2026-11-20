"""Tests for the llama.cpp KL path. They need the built helper and two GGUF files of one model, so they
skip unless these are set:

    QTRACE_KL_SCORE=path/to/kl_score QTRACE_GGUF_P=bf16.gguf QTRACE_GGUF_Q=q3_k_m.gguf \
        python -m pytest tests/test_kl_gguf.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.kl import logit_index, score_stats
from qtrace.kl_gguf import compare_gguf, score_trajectories_gguf

BIN, P, Q = (os.environ.get(k) for k in ("QTRACE_KL_SCORE", "QTRACE_GGUF_P", "QTRACE_GGUF_Q"))
pytestmark = pytest.mark.skipif(not (BIN and P and Q), reason="QTRACE_KL_SCORE, QTRACE_GGUF_P and QTRACE_GGUF_Q not set")


def episode(n=700, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 150_000, (1, n), generator=g)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[50:300] = True
    mask[420:690] = True
    return {"input_ids": ids, "model_mask": mask, "command_spans": [(100, 140), (500, 520)],
            "tool_call_positions": [99, 499], "turn_spans": [(50, 300), (420, 690)]}


def test_same_model_gives_zero_kl():
    e = episode()
    s, = compare_gguf(BIN, P, P, [(e["input_ids"], logit_index(e["model_mask"]))])
    assert s["kl"].abs().max() < 1e-6
    assert s["flip"].sum() == 0
    assert torch.equal(s["nll_p"], s["nll_q"])


def test_batch_size_and_subsets_do_not_change_stats():
    e = episode()
    idx = logit_index(e["model_mask"])
    small, sub = compare_gguf(BIN, P, Q, [(e["input_ids"], idx), (e["input_ids"], idx[::5])], n_batch=64)
    big, = compare_gguf(BIN, P, Q, [(e["input_ids"], idx)], n_batch=512)
    assert small["kl"].mean() > 0
    # Equal to float tolerance, not bitwise: llama.cpp picks matmul kernels by batch size and by the
    # number of output rows. Repeat runs with the same settings are bitwise identical.
    for k in small:
        assert torch.allclose(small[k], big[k], rtol=2e-3, atol=2e-3), k
        assert torch.allclose(small[k][::5], sub[k], rtol=2e-3, atol=2e-3), k


def test_episode_scores_match_score_stats():
    e = episode()
    idx = logit_index(e["model_mask"])
    out, = score_trajectories_gguf(BIN, P, Q, [e])
    s, = compare_gguf(BIN, P, Q, [(e["input_ids"], idx)])
    assert out == score_stats(s, idx, e["tool_call_positions"], e["command_spans"], None, e["turn_spans"])
    assert out["n_scored"] == len(idx) and out["n_command"] == 60 and set(out["halves"]) == {"even", "odd"}


def test_bad_positions_fail_loudly():
    e = episode()
    with pytest.raises(RuntimeError, match="idx must be sorted"):
        compare_gguf(BIN, P, Q, [(e["input_ids"], torch.tensor([5, 3]))])
