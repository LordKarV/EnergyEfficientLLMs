"""Tests for the training probe on a tiny Qwen3.5. Run:  python -m pytest tests/test_probe.py -q"""

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest
import torch

from qtrace.probe import (
    DELTANET_TARGETS,
    GATE_TARGETS,
    attach_lora,
    chunked_ce_loss,
    kernel_paths,
    kernel_status,
    probe_length,
    supervised_mask,
)


def tiny_qwen35(seed=0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(cfg)


def test_lora_covers_deltanet_mixers_and_attention():
    model = attach_lora(tiny_qwen35(), include_gates=False)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for target in DELTANET_TARGETS + ["q_proj", "o_proj", "down_proj"]:
        assert any(f"{target}.lora_A" in n for n in names), target
    assert not any(f"{g}.lora_A" in n for g in GATE_TARGETS for n in names)


def test_lora_gates_are_optional():
    model = attach_lora(tiny_qwen35(), include_gates=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(any(f"{g}.lora_A" in n for n in names) for g in GATE_TARGETS)


def test_probe_length_reports_time_memory_and_trains():
    model = attach_lora(tiny_qwen35(), include_gates=False)
    for path in ("chunked", "full_logits"):
        rec = probe_length(model, length=96, steps=2, device="cpu", vocab_size=300, seed=0,
                           supervised_frac=0.5, loss_path=path, chunk=16)
        assert rec["status"] == "ok" and rec["length"] == 96 and rec["loss_path"] == path
        assert rec["sec_per_step"] > 0 and rec["loss_finite"] and rec["lora_grad_norm"] > 0
        assert rec["n_supervised"] > 0


def test_supervised_mask_hits_the_fraction_in_contiguous_spans():
    m = supervised_mask(8192, 0.15, period=2048)
    assert not m[0]
    assert abs(m.float().mean().item() - 0.15) < 0.01
    assert (m[1:] != m[:-1]).sum().item() <= 2 * 4  # at most one span per period


def test_chunked_loss_matches_full_logits_loss_and_gradients():
    torch.manual_seed(0)
    model = attach_lora(tiny_qwen35().float(), include_gates=False, grad_ckpt=False)
    ids = torch.randint(0, 300, (1, 64))
    mask = supervised_mask(64, 0.5, period=16)
    labels = torch.where(mask, ids[0], torch.full_like(ids[0], -100)).unsqueeze(0)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # LoRA B starts at zero; make the comparison non-trivial
        p.data.normal_(0, 0.02)

    full = model(input_ids=ids, labels=labels).loss
    g_full = torch.autograd.grad(full, params)
    chunked = chunked_ce_loss(model, ids, mask, chunk=7)
    g_chunk = torch.autograd.grad(chunked, params)
    assert torch.allclose(full, chunked, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_full, g_chunk))


def test_kernel_status_names_both_kernels():
    status = kernel_status()
    assert set(status) == {"fla", "causal_conv1d"}


def test_kernel_paths_report_what_transformers_resolved():
    paths = kernel_paths()
    assert set(paths) == {"chunk_gated_delta_rule", "causal_conv1d_fn"}
    status = kernel_status()
    if not status["fla"]:
        assert not paths["chunk_gated_delta_rule"]["fast"]
        assert paths["chunk_gated_delta_rule"]["impl"].endswith("modeling_qwen3_5.torch_chunk_gated_delta_rule")
    if not status["causal_conv1d"]:
        assert not paths["causal_conv1d_fn"]["fast"]


def load_script():
    path = os.path.join(ROOT, "scripts", "p0_training_probe.py")
    spec = importlib.util.spec_from_file_location("p0_training_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SLOW_IN_USE = {"chunk_gated_delta_rule": {"fast": False, "impl": "torch"},
               "causal_conv1d_fn": {"fast": True, "impl": "causal_conv1d"}}


def test_script_refuses_importable_kernels_that_are_not_in_use(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "kernel_status", lambda: {"fla": True, "causal_conv1d": True})
    monkeypatch.setattr(script, "kernel_paths", lambda: SLOW_IN_USE)
    out = tmp_path / "probe.jsonl"
    with pytest.raises(SystemExit, match="allow-slow-kernels"):
        script.main(["--model", str(tmp_path / "no_model"), "--out", str(out)])
    assert not out.exists()


def test_script_records_kernels_in_use_in_every_row(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    tiny_qwen35().save_pretrained(ckpt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    out = tmp_path / "probe.jsonl"
    load_script().main(["--model", str(ckpt), "--lengths", "48", "64", "--steps", "1", "--chunk", "16",
                        "--supervised-frac", "0.5", "--allow-slow-kernels", "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
    assert all(r["kernels"] == {"in_use": kernel_paths(), "importable": kernel_status()} for r in rows)
