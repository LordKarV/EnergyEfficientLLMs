"""Repo-state fingerprint after every tool call (docs/04, P0d replay rule).

sha256 over three parts, in a fixed order: `git status --porcelain` (untracked files included),
`git diff HEAD --binary` (staged and unstaged, binary-safe), and the sha256 of every untracked file's
content, so editing an untracked file changes the hash too. An untracked symlink contributes its
target path, not its content: some images ship dangling links (orange3's `datasets`). Runs in its own `docker exec`, outside
the model's context. The R2E-Gym virtualenv and test folder are excluded: the harness creates them
and they never differ between attempts.
"""

import hashlib

EXCLUDE = "-- . ':(exclude).venv' ':(exclude)r2e_tests'"
SCRIPT = f"""set -e
git status --porcelain=v1 -z --untracked-files=all {EXCLUDE}
printf '\\0--diff--\\0'
git -c core.quotepath=off diff HEAD --binary {EXCLUDE}
printf '\\0--untracked--\\0'
git ls-files --others --exclude-standard -z {EXCLUDE} | xargs -0 -r sh -c \
  'for f do if [ -L "$f" ]; then printf "link %s -> %s\\n" "$f" "$(readlink "$f")"; else sha256sum "$f"; fi; done' sh
"""


def fingerprint(container, repo="/testbed", timeout=60):
    """Hex sha256 of the repo state. Raises RuntimeError if git fails, so a bad read is never hashed."""
    out, code = container.exec(SCRIPT, workdir=repo, timeout=timeout)
    if code != 0:
        raise RuntimeError(f"fingerprint failed (exit {code}): {out[-500:]!r}")
    return hashlib.sha256(out).hexdigest()
