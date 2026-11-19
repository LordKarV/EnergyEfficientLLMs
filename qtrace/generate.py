"""One batched text-generation function, used for both sampling (script 01) and evaluation (script 05)."""

import torch


def stop_token_ids(model, tokenizer):
    """All token ids that mean "the answer is over": the model's EOS ids plus the padding id."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    ids.add(tokenizer.pad_token_id)
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens, samples_per_prompt=1,
             temperature=None, top_p=None, top_k=None):
    """Generate continuations for a list of prompt strings.

    temperature=None means greedy decoding (deterministic, one output per prompt).
    Otherwise we sample `samples_per_prompt` continuations per prompt.

    Returns a flat list of dicts in the order
    [prompt0-sample0, prompt0-sample1, ..., prompt1-sample0, ...], each with:
      text      the generated text
      n_tokens  number of generated tokens before the stop token
      finished  False if generation was cut off by max_new_tokens
    """
    # The chat template already inserted special tokens, so do not add them a second time.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    inputs = inputs.to(model.device)

    options = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    if temperature is None:
        options["do_sample"] = False
    else:
        options.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                       num_return_sequences=samples_per_prompt)

    output = model.generate(**inputs, **options)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]  # drop the prompt part

    stops = stop_token_ids(model, tokenizer)
    results = []
    for row in new_tokens.tolist():
        finished = False
        for position, token_id in enumerate(row):
            if token_id in stops:
                row = row[:position]
                finished = True
                break
        results.append({
            "text": tokenizer.decode(row, skip_special_tokens=True).strip(),
            "n_tokens": len(row),
            "finished": finished,
        })
    return results
