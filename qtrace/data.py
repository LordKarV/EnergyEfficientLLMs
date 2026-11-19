"""Load math problems and build the exact text the model sees."""

from qtrace.answers import extract_answer
from qtrace.common import read_jsonl

INSTRUCTION = (
    "Solve the following math problem. Show your reasoning step by step, "
    "then write the final answer on the last line in the form '#### <number>'."
)


def load_problems(source, limit=None):
    """Return a list of {qid, question, gold}.

    source is either "gsm8k:train" / "gsm8k:test" (downloaded from the HF hub)
    or a path to a local .jsonl file with "question" and "answer" fields, where
    "answer" ends in "#### <number>" like GSM8K.
    """
    if source.startswith("gsm8k:"):
        from datasets import load_dataset

        split = source.split(":")[1]
        raw = list(load_dataset("openai/gsm8k", "main", split=split))
        prefix = "gsm8k-" + split
    else:
        raw = read_jsonl(source)
        prefix = "local"

    problems = []
    for index, row in enumerate(raw):
        gold, _ = extract_answer(row["answer"])
        problems.append({"qid": f"{prefix}-{index}", "question": row["question"], "gold": gold})
    if limit is not None:
        problems = problems[:limit]
    return problems


def build_prompt(tokenizer, question, enable_thinking=False):
    """Format one problem with the model's chat template, ending where the model should start writing.

    The same function is used for sampling, scoring, training and evaluation, so the
    model always sees an identical prompt format.
    """
    messages = [{"role": "user", "content": INSTRUCTION + "\n\nProblem: " + question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,     # append the "assistant starts here" tokens
        enable_thinking=enable_thinking,  # Qwen3 template switch; ignored by templates without it
    )
