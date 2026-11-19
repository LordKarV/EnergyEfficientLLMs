"""Build a tiny random Qwen3-style model and tokenizer for offline smoke tests.

The model is random, so its answers are nonsense. The point is only to check that
every script runs end to end without downloading anything or needing a GPU.

Run:  python tests/make_tiny_model.py   (writes tests/tiny_model/)
"""

import os

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_model")

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)

# Vocabulary: special tokens, then printable ASCII characters (a character-level tokenizer).
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<unk>"]
vocab = {tok: i for i, tok in enumerate(specials)}
for code in list(range(32, 127)) + [10]:
    vocab[chr(code)] = len(vocab)

tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")  # split into single characters
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    eos_token="<|im_end|>",
    pad_token="<|endoftext|>",
    unk_token="<unk>",
    additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
)
tokenizer.chat_template = CHAT_TEMPLATE

config = Qwen3Config(
    vocab_size=len(vocab), hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=2048,
    eos_token_id=vocab["<|im_end|>"], pad_token_id=vocab["<|endoftext|>"], bos_token_id=None,
)
model = Qwen3ForCausalLM(config)

model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("wrote", OUT)
