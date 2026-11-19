"""Unit tests for the parts that must be exactly right. Run:  python -m pytest tests -q"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.answers import extract_answer, is_correct
from qtrace.quantize import quantize_model_rtn, quantize_weight_rtn
from qtrace.select import ARMS, build_arm, group_candidates, pick, ranks


def test_extract_answer_with_marker():
    assert extract_answer("3 + 4 = 7\n#### 7") == (7.0, True)
    assert extract_answer("so the total is\n#### $1,234.50") == (1234.5, True)
    assert extract_answer("first #### 5 then fixed\n#### -2") == (-2.0, True)
    assert extract_answer("The answer is #### 18.") == (18.0, True)


def test_extract_answer_fallback_and_missing():
    assert extract_answer("She has 3 and then 10 apples") == (10.0, False)
    assert extract_answer("no numbers here") == (None, False)
    assert extract_answer("#### none") == (None, True)


def test_is_correct():
    assert is_correct(18.0, 18)
    assert is_correct(0.33, 1 / 3) is False
    assert is_correct(None, 5) is False
    assert is_correct(1000000.0, 1000000.5) is True   # relative tolerance on big numbers
    assert is_correct(7.0, 8.0) is False


def test_rtn_uses_at_most_2_pow_bits_values_per_group():
    torch.manual_seed(0)
    w = torch.randn(4, 256)
    for bits in (2, 3, 4):
        q = quantize_weight_rtn(w, bits, group_size=128)
        assert q.shape == w.shape
        for row in q.reshape(4, 2, 128).reshape(-1, 128):
            assert row.unique().numel() <= 2**bits


def test_rtn_error_shrinks_with_more_bits_and_keeps_range():
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    errors = [(quantize_weight_rtn(w, b) - w).abs().mean().item() for b in (2, 3, 4, 8, 16)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-4
    q = quantize_weight_rtn(w, 3)
    assert torch.allclose(q.amin(dim=1), w.amin(dim=1)) and torch.allclose(q.amax(dim=1), w.amax(dim=1))


def test_rtn_handles_row_length_not_divisible_by_group():
    w = torch.randn(3, 100)
    assert quantize_weight_rtn(w, 4, group_size=128).shape == (3, 100)


def test_quantize_model_skips_lm_head():
    model = torch.nn.ModuleDict({"block": torch.nn.Linear(128, 8), "lm_head": torch.nn.Linear(128, 8)})
    head_before = model["lm_head"].weight.clone()
    block_before = model["block"].weight.clone()
    assert quantize_model_rtn(model, 3) == 1
    assert torch.equal(model["lm_head"].weight, head_before)
    assert not torch.equal(model["block"].weight, block_before)


def cand(n_tokens, kl, entropy=0.0, text=None, qid="q"):
    return {"qid": qid, "n_tokens": n_tokens, "kl_mean": kl, "entropy_fp": entropy,
            "text": text or f"{n_tokens}-{kl}"}


def test_ranks():
    cands = [cand(30, 0.1), cand(10, 0.3), cand(20, 0.2)]
    assert ranks(cands, "n_tokens") == [1.0, 0.0, 0.5]


def test_pick_rules():
    import random
    cands = [cand(10, 0.9, 0.5), cand(12, 0.1, 0.9), cand(40, 0.05, 0.1), cand(50, 0.8, 0.7)]
    rng = random.Random(0)
    assert pick(cands, "shortest", 0.5, rng)["n_tokens"] == 10
    assert pick(cands, "longest", 0.5, rng)["n_tokens"] == 50
    assert pick(cands, "stable", 0.5, rng)["kl_mean"] == 0.05
    assert pick(cands, "low_entropy", 0.5, rng)["entropy_fp"] == 0.1
    assert pick(cands, "qtrace", 0.5, rng)["n_tokens"] == 12        # short and stable
    assert pick(cands, "qtrace", 1.0, rng)["n_tokens"] == 10        # alpha=1 is "shortest"
    assert pick(cands, "qtrace", 0.0, rng)["kl_mean"] == 0.05       # alpha=0 is "stable"
    assert pick(cands, "short_stable", 0.5, rng)["n_tokens"] == 12
    assert pick(cands, "short_unstable", 0.5, rng)["n_tokens"] == 10
    assert pick(cands, "long_stable", 0.5, rng)["n_tokens"] == 40
    assert pick(cands, "long_unstable", 0.5, rng)["n_tokens"] == 50


def test_group_and_build_arm_same_problems_for_every_arm():
    rows = [cand(10 + i, 0.1 * i, qid="a") for i in range(5)]
    rows += [cand(10, 0.1, qid="a", text="10-0.0")]             # duplicate text is dropped
    rows += [cand(10 + i, 0.1 * i, qid="b") for i in range(2)]  # too few candidates
    grouped = group_candidates(rows, min_candidates=4)
    assert list(grouped) == ["a"] and len(grouped["a"]) == 5
    for arm in ARMS:
        chosen = build_arm(grouped, arm)
        assert [c["qid"] for c in chosen] == ["a"]
