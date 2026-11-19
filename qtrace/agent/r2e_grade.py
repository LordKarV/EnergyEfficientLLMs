"""R2E-Gym grading for the harness's verify() (docs/04: R2E-Gym runtime, graded by its own tests).

Mirrors R2E-Gym's reward (r2egym/agenthub/runtime/docker.py, `_calculate_reward_r2e`, commit 0d94c4e):
run the task's hidden `run_tests.sh`, parse pytest's short summary, and count the task resolved only if
the parsed test -> status map matches the row's `expected_output_json` exactly (same keys, same
statuses, keys compared up to the first " - ").

R2E-Gym's reward is all-or-nothing, and the dataset's old-commit results have no per-test names, so
the fail-to-pass / pass-to-pass split comes from a baseline run of the same tests on the untouched
repo (`baseline_statuses`). Tests are deterministic per image, so the harness can cache the baseline
per task.

Hidden files: R2E-Gym images ship `run_tests.sh` in /testbed and the tests in /r2e_tests. `hide_tests`
moves both to /root before the model starts (R2E-Gym's `setup_env` does the same). `verify` links the
tests back only for the test run and removes the link afterwards, so the model never sees them.
"""

import json
import re
from dataclasses import dataclass

TEST_TIMEOUT = 300


@dataclass(frozen=True)
class Grading:
    resolved: bool
    fail_to_pass_ok: int
    fail_to_pass_total: int
    pass_to_pass_ok: int
    pass_to_pass_total: int


def parse_pytest_log(log):
    """R2E-Gym's parse_log_pytest, verbatim in behavior: statuses from the short test summary."""
    status = {}
    if not log or "short test summary info" not in log:
        return status
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            status[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            status[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            status[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return status


def _normalize(status):
    decolored = {re.sub(r"\u001b\[\d+m", "", k): v for k, v in status.items()}
    return {k.split(" - ")[0]: decolored[k] for k in sorted(decolored)}


def r2e_resolved(parsed, expected):
    """R2E-Gym's rule: same number of tests and every parsed status equals the expected one."""
    parsed, expected = _normalize(parsed), _normalize(expected)
    if len(parsed) != len(expected):
        return False
    return all(not k or (k in expected and parsed[k] == expected[k]) for k in parsed)


def grading(parsed, expected, baseline):
    """Grading for the harness. F2P: expected to pass but not passing at baseline; P2P: passing at both."""
    parsed, expected, baseline = _normalize(parsed), _normalize(expected), _normalize(baseline)
    should_pass = [k for k, v in expected.items() if v == "PASSED"]
    f2p = [k for k in should_pass if baseline.get(k) != "PASSED"]
    p2p = [k for k in should_pass if baseline.get(k) == "PASSED"]
    return Grading(
        resolved=r2e_resolved(parsed, expected),
        fail_to_pass_ok=sum(parsed.get(k) == "PASSED" for k in f2p),
        fail_to_pass_total=len(f2p),
        pass_to_pass_ok=sum(parsed.get(k) == "PASSED" for k in p2p),
        pass_to_pass_total=len(p2p),
    )


def hide_tests(container):
    """Move run_tests.sh and the tests out of the model's view, as R2E-Gym's setup_env does."""
    out, code = container.exec("mv /testbed/run_tests.sh /root/run_tests.sh && mv /r2e_tests /root/r2e_tests",
                               workdir="/")
    if code != 0:
        raise RuntimeError(f"hide_tests failed (exit {code}): {out[-500:]!r}")


def run_tests(container, timeout=TEST_TIMEOUT):
    """Run the hidden tests with a temporary link, then remove it. Returns the parsed status map."""
    script = ("ln -sfn /root/r2e_tests /testbed/r2e_tests; bash /root/run_tests.sh; rc=$?; "
              "rm -f /testbed/r2e_tests; exit $rc")
    out, _ = container.exec(script, workdir="/testbed", timeout=timeout)
    log = re.sub(r"\x1b\[[0-9;]*m|\r", "", out.decode("utf-8", errors="replace"))
    return parse_pytest_log(log)


def baseline_statuses(container, timeout=TEST_TIMEOUT):
    """Statuses on the untouched repo. Call after hide_tests and before the model's first turn."""
    return run_tests(container, timeout)


def verify(container, expected_output_json, baseline, timeout=TEST_TIMEOUT):
    """Grade the container's current repo state against the task row's expected_output_json."""
    expected = json.loads(expected_output_json) if isinstance(expected_output_json, str) else expected_output_json
    return grading(run_tests(container, timeout), expected, baseline)
