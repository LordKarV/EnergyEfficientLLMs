"""Small helpers shared by every script: config loading, JSONL files, seeding."""

import json
import os
import random

import yaml


def load_config(path):
    """Read a YAML config file into a plain nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Write a list of dicts to a .jsonl file, creating the folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path, rows):
    """Append rows to a .jsonl file. Used so long jobs can be resumed after a crash."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def set_seed(seed):
    """Seed Python and PyTorch so sampling and training are repeatable."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
