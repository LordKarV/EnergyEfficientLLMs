"""Pull the final numeric answer out of a solution and check it against the gold answer.

GSM8K gold answers end with a line like "#### 18". We ask the model to use the same
marker, so grading is a string-parsing problem, not a judgment call.
"""

import re

# Matches numbers such as 18, -3, 1,234, 2.50, .5
NUMBER = re.compile(r"-?(?:\d[\d,]*\.?\d*|\.\d+)")


def to_number(text):
    """Turn '1,234.50' or '$18' into a float. Returns None if there is no number."""
    match = NUMBER.search(text.replace("$", ""))
    if match is None:
        return None
    cleaned = match.group(0).replace(",", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer(text):
    """Return (number, used_marker).

    If the text contains '####', we take the number right after the LAST marker.
    Otherwise we fall back to the last number anywhere in the text and report
    used_marker=False, so we can count how often the model ignores the format.
    """
    if "####" in text:
        after_marker = text.rsplit("####", 1)[1]
        return to_number(after_marker), True
    numbers = NUMBER.findall(text.replace("$", ""))
    if not numbers:
        return None, False
    return to_number(numbers[-1]), False


def is_correct(predicted, gold, tolerance=1e-4):
    """Numeric comparison with a small tolerance, so 18 and 18.0 count as equal."""
    if predicted is None or gold is None:
        return False
    return abs(predicted - gold) <= tolerance * max(1.0, abs(gold))
