"""`key_is_clean` must actually FIND a key, not merely be wired up correctly.

The existing suite pins the guard's position -- that it scans the same pathspec
the commit carries, that it runs after `git add`, that a refusal unstages only
evidence. Nothing asserted that it detects anything. So the one backstop between
a live ARC API key and a push to GitHub had its core behaviour untested, which
is the same shape as the bug it was written for: that guard once ran one line
BEFORE `git add`, iterated an empty list, and returned clean every time.

The gzip case is the one most likely to matter in practice: evidence is stored
compressed (`spans.json.gz`, archived traces), so a guard that only greps plain
text would pass every real artifact while looking thorough.

The real function is extracted from the real file and run against fixture repos,
deliberately WITHOUT `cd`-ing into them. That is not an oversight: the file list
comes from `git -C "$REPO"` but the reads used cwd-relative paths, so the guard
was correct only because the daemon happens to `cd "$REPO"` before its main loop.
Running it from elsewhere is what distinguishes "correct" from "correct by
ambient coincidence" -- and before the fix, these two detection cases both
reported CLEAN with the key sitting in a staged file.
"""

from __future__ import annotations

import gzip
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "tools" / "preserve_evidence.sh").read_text(encoding="utf-8")

KEY = "sk-live-FIXTURE-NOT-A-REAL-KEY-000"


def _key_is_clean_fn() -> str:
    m = re.search(r"^key_is_clean\(\) \{.*?^\}$", SCRIPT, re.S | re.M)
    assert m, "preserve_evidence.sh no longer defines key_is_clean()"
    return m.group(0)


def _git_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GIT_", "ARC_API_KEY"))}
    env.update(
        HOME=str(home), GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig-none"),
        GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@example.invalid",
        LC_ALL="C",
    )
    return env


@pytest.fixture
def staged(tmp_path: Path):
    """A repo with an `evidence/` directory whose contents are staged."""
    home = tmp_path / "home"
    home.mkdir()
    env = _git_env(home)
    repo = tmp_path / "work"
    (repo / "evidence").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)],
                   env=env, check=True, capture_output=True)

    def stage():
        subprocess.run(["git", "-C", str(repo), "add", "evidence"],
                       env=env, check=True, capture_output=True)

    return repo, env, stage


def _run(repo: Path, env: dict, key: str | None) -> subprocess.CompletedProcess:
    script = (
        f'REPO={repo}\n'
        'log() { echo "LOG: $*"; }\n'
        + ("" if key is None else f'export ARC_API_KEY={key!r}\n')
        + _key_is_clean_fn()
        + '\nkey_is_clean; echo "rc=$?"\n'
    )
    return subprocess.run(["/bin/bash", "-c", script], env=env,
                          capture_output=True, text=True, timeout=60)


def test_a_plaintext_key_in_staged_evidence_is_refused(staged) -> None:
    repo, env, stage = staged
    (repo / "evidence" / "run.log").write_text(
        f"launching with ARC_API_KEY={KEY}\n", encoding="utf-8")
    stage()
    out = _run(repo, env, KEY)
    assert "rc=1" in out.stdout, f"the key was not detected: {out.stdout!r}"
    assert "REFUSING TO COMMIT" in out.stdout
    assert "run.log" in out.stdout, "the refusal does not name the offending file"


def test_a_gzipped_key_in_staged_evidence_is_refused(staged) -> None:
    """Evidence is stored compressed, so this is the case that matters.

    A guard that only greps plain text passes every real artifact in this repo
    while looking thorough.
    """
    repo, env, stage = staged
    blob = repo / "evidence" / "trace.jsonl.gz"
    blob.write_bytes(gzip.compress(
        f'{{"cmd": "export ARC_API_KEY={KEY}"}}\n'.encode()))
    stage()
    out = _run(repo, env, KEY)
    assert "rc=1" in out.stdout, (
        f"a key inside a gzipped evidence file was not detected: {out.stdout!r}"
    )
    assert "trace.jsonl.gz" in out.stdout


def test_clean_evidence_passes(staged) -> None:
    """The other direction, or `return 1` would satisfy every case above."""
    repo, env, stage = staged
    (repo / "evidence" / "run.log").write_text(
        "nothing secret here\n", encoding="utf-8")
    (repo / "evidence" / "t.gz").write_bytes(gzip.compress(b'{"ok": true}\n'))
    stage()
    out = _run(repo, env, KEY)
    assert "rc=0" in out.stdout, f"clean evidence was refused: {out.stdout!r}"
    assert "REFUSING" not in out.stdout


def test_an_unset_key_refuses_rather_than_passing(staged) -> None:
    """Fail closed: a check that cannot run must not report clean.

    This is the property the whole file turns on -- the guard's predecessor
    reported clean precisely because it had nothing to inspect.
    """
    repo, env, stage = staged
    (repo / "evidence" / "run.log").write_text("anything\n", encoding="utf-8")
    stage()
    out = _run(repo, env, None)
    assert "rc=1" in out.stdout, (
        f"an unset ARC_API_KEY was treated as clean: {out.stdout!r}"
    )
    assert "cannot run" in out.stdout


def test_an_unstaged_key_is_not_what_this_guard_reports_on(staged) -> None:
    """Scope check: the guard reads the INDEX, so an unstaged file is not a hit.

    Asserting this pins why staging must come first -- if the order regressed,
    the plaintext case above would silently become this one.
    """
    repo, env, stage = staged
    (repo / "evidence" / "clean.log").write_text("fine\n", encoding="utf-8")
    stage()
    # written AFTER staging, so it is not in the index
    (repo / "evidence" / "later.log").write_text(f"{KEY}\n", encoding="utf-8")
    out = _run(repo, env, KEY)
    assert "rc=0" in out.stdout, (
        "the guard reported on a file that was not staged; it must read the "
        f"index, which is what the commit will carry: {out.stdout!r}"
    )
