"""Tests for the training probe on a tiny Qwen3.5. Run:  python -m pytest tests/test_probe.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from qtrace.probe import DELTANET_TARGETS, GATE_TARGETS, attach_lora, kernel_status, probe_length


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
    rec = probe_length(model, length=96, steps=2, device="cpu", vocab_size=300, seed=0)
    assert rec["status"] == "ok" and rec["length"] == 96
    assert rec["sec_per_step"] > 0 and rec["loss_finite"]
    assert rec["lora_grad_norm"] > 0


def test_kernel_status_names_both_kernels():
    status = kernel_status()
    assert set(status) == {"fla", "causal_conv1d"}
