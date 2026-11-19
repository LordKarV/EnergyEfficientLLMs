"""Load a tokenizer and a model at a chosen numeric precision.

Precision names used everywhere in this repo:
  bf16   full precision reference (float32 on machines without a CUDA GPU)
  rtnN   our own simulated N-bit round-to-nearest weights, e.g. rtn8, rtn4, rtn3
         (accuracy is faithful; speed, memory and energy are NOT)
  bnb8   bitsandbytes 8-bit  (real low-bit storage; needs a CUDA GPU)
  bnb4   bitsandbytes 4-bit NF4 (real low-bit storage; needs a CUDA GPU)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qtrace.quantize import quantize_model_rtn


def pick_device():
    """Use the NVIDIA GPU if there is one, then Apple Silicon, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Decoder-only models continue from the END of the prompt, so when several
    # prompts of different length share a batch the padding must go on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name, precision="bf16"):
    """Return a model in eval mode on the best available device."""
    device = pick_device()

    if precision in ("bnb8", "bnb4"):
        from transformers import BitsAndBytesConfig

        if precision == "bnb8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=quant_config, device_map={"": 0}
        )
        return model.eval()

    # Everything else starts from the full-precision weights.
    model = AutoModelForCausalLM.from_pretrained(name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)

    if precision.startswith("rtn"):
        bits = int(precision[len("rtn"):])
        quantize_model_rtn(model, bits)
    elif precision != "bf16":
        raise ValueError(f"Unknown precision: {precision}")

    return model.eval()


def efficiency_is_real(precision):
    """True if latency, memory and energy measured at this precision mean something."""
    return not precision.startswith("rtn")
