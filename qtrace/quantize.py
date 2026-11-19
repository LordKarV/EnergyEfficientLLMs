"""Round-to-nearest (RTN) weight quantization, written out by hand.

This is "simulated" (also called "fake") quantization: each weight is snapped to
one of 2^bits allowed values, but the result is stored back as a normal float
tensor. That reproduces the ACCURACY effect of b-bit weights exactly. It does
NOT make the model smaller or faster, so never report latency, memory or energy
from a model quantized this way. Use a real low-bit runtime for those
(see docs/02_research_plan.md, section "Energy").

Scheme: weight-only, asymmetric, uniform, per group of `group_size` input
weights. Group size 128 is the convention in the quantized-reasoning papers we
compare against.
"""

import torch


def quantize_weight_rtn(weight, bits, group_size=128):
    """Quantize one 2-D weight matrix and return the de-quantized float copy.

    weight: tensor of shape (out_features, in_features)
    bits:   number of bits per weight, e.g. 8, 4 or 3
    """
    out_features, in_features = weight.shape
    # If the row length is not a multiple of group_size, treat each whole row as one group.
    group = group_size if in_features % group_size == 0 else in_features

    # Shape (out_features, number_of_groups, group): every group gets its own range.
    w = weight.detach().float().reshape(out_features, in_features // group, group)

    w_min = w.amin(dim=-1, keepdim=True)
    w_max = w.amax(dim=-1, keepdim=True)

    levels = 2**bits - 1                               # e.g. 4 bits -> integers 0..15
    scale = (w_max - w_min).clamp(min=1e-8) / levels   # width of one quantization step

    integers = torch.round((w - w_min) / scale).clamp(0, levels)  # what a real runtime would store
    dequantized = integers * scale + w_min                        # what the layer actually computes with

    return dequantized.reshape(out_features, in_features).to(weight.dtype)


def quantize_model_rtn(model, bits, group_size=128, skip=("lm_head",)):
    """Quantize every nn.Linear weight in the model in place. Returns the number of layers changed.

    We skip the output head because in Qwen3 it shares its weight tensor with the
    input embedding table, and the papers we compare against quantize only the
    linear layers inside the transformer blocks.
    """
    changed = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(part in name for part in skip):
            continue
        module.weight.data = quantize_weight_rtn(module.weight.data, bits, group_size)
        changed += 1
    return changed
