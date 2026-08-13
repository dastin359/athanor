"""One preserver per repository, and the lock must die with its holder.

`preserve_evidence.sh` takes an `flock` on fd 9 so that two preservers cannot
race the same git index -- the symptom being a `COMMIT FAILED` logged on a cycle
where nothing was actually wrong.

**The lock did not work the way the file believed it did.** Bash does not set
`O_CLOEXEC` on a descriptor opened by `exec`, so every child inherits fd 9 --
including the five `sleep` calls, one of which is `sleep "$TICK"`. An inherited
fd keeps the flock alive, so THE LOCK OUTLIVES ITS HOLDER. Reproduced here
2026-08-12 with the file's exact construct: SIGKILL the holder, start a
replacement, and it prints "another preserver already holds ... -- exiting rather
than racing its git index" and exits 0, while `fuser` shows a surviving `sleep`
still pinning the lock file.

The consequence is specific to a machine that is not replaced on a timer: after
any hard kill -- OOM, `kill -9`, unclean reboot -- NO preserver can start for up
to `TICK` seconds. `docs/ccarc3_desktop_migration.md` recommends
`CCARC3_PRESERVE_TICK=1800`, making that a thirty-minute blackout in which
`daemon_watchdog.sh` is refused too, and the log line reads exactly like correct
behaviour. A guard strictly worse than no guard.

This is the same defect `tools/supervisor.sh` shipped and reverted within the
hour (`51d5955`), and `tests/test_only_one_supervisor_runs.py` is the worked
example this file follows.

Two checks, and the order matters. The behavioural one is the evidence: it starts
a real preserver, lets it spawn its real children, kills it, and demands that a
fresh one start. The grep beside it is a cheap early warning for a spawn added
later without `9>&-`; it would not have caught the original bug on its own,
because the original bug was the absence of the whole convention.

**Everything runs against a throwaway repo, never this one.** The lock is keyed
on `$REPO`, which `preserve_evidence.sh` derives from its own location, so a copy
under `tmp_path` takes a different lock and cannot collide with a live preserver.
That isolation is also what keeps a unit test from committing to `evidence/` and
pushing it to a PUBLIC remote: the fixture repo has no remote at all.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PRESERVER = REPO / "tools" / "preserve_evidence.sh"

#: Long enough that the tick `sleep` is certainly alive when the holder is
#: killed -- that sleep is the child the whole bug is about.
TICK = "600"

REFUSAL = "another preserver already holds"


@pytest.fixture
def box(tmp_path):
    """A throwaway repo holding a copy of the preserver, plus a scratch dir."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy2(PRESERVER, repo / "tools" / PRESERVER.name)
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    for k, v in (("user.email", "t@example.invalid"), ("user.name", "t")):
        subprocess.run(["git", "config", k, v], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"],
                   cwd=repo, check=True)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            cwd=repo, capture_output=True, text=True,
                            check=True).stdout.strip()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return repo, scratch, branch


def _spawn(box, log: pathlib.Path):
    repo, scratch, branch = box
    env = dict(os.environ, CCARC3_SCRATCH=str(scratch),
               CCARC3_PRESERVE_TICK=TICK, CCARC3_BRANCH=branch)
    return subprocess.Popen(["bash", str(repo / "tools" / PRESERVER.name)],
                            cwd=str(repo), env=env,
                            stdout=log.open("w"), stderr=subprocess.STDOUT)


def _settle(proc, log: pathlib.Path, seconds: float = 4.0):
    """Let it take the lock and spawn its children, then prove it is still up."""
    time.sleep(seconds)
    assert proc.poll() is None, (
        "the preserver exited instead of holding the lock:\n"
        + log.read_text(encoding="utf-8"))


def _kill(proc):
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=20)


# ── the behavioural checks ───────────────────────────────────────────────────

def test_the_second_preserver_refuses_to_start(box, tmp_path):
    """The positive control: without this, the regression below could pass
    simply because the lock never worked at all."""
    first = _spawn(box, tmp_path / "a.log")
    second = None
    try:
        _settle(first, tmp_path / "a.log")
        second = _spawn(box, tmp_path / "b.log")
        second.wait(timeout=60)
        text = (tmp_path / "b.log").read_text(encoding="utf-8")
        assert REFUSAL in text, (
            "a second preserver started alongside the first; both would race the "
            "same git index:\n" + text)
    finally:
        _kill(second)
        _kill(first)


def test_the_lock_dies_with_its_holder(box, tmp_path):
    """The regression, and the reason this file exists.

    SIGKILL runs no cleanup -- the realistic case, since an OOM kill, an
    operator's `kill -9` and an unclean reboot all look like this. If any
    surviving child still holds fd 9 the flock outlives the process, and every
    future preserver is refused for as long as that child lives.
    """
    first = _spawn(box, tmp_path / "a.log")
    _settle(first, tmp_path / "a.log")
    first.kill()
    first.wait(timeout=20)

    second = _spawn(box, tmp_path / "b.log")
    try:
        _settle(second, tmp_path / "b.log")
        text = (tmp_path / "b.log").read_text(encoding="utf-8")
        assert REFUSAL not in text, (
            "the lock outlived its holder -- a child inherited fd 9 and is "
            "keeping it alive. Every spawn in preserve_evidence.sh that can "
            "outlive the statement starting it needs `9>&-`:\n" + text)
    finally:
        _kill(second)


# ── the cheap early warning ──────────────────────────────────────────────────

#: The property is narrow and mechanical: **a process that can outlive the
#: statement which started it must not hold fd 9.** That is a background spawn
#: (`&`) and `sleep`, which is synchronous but survives a kill of its parent for
#: as long as it was told to sleep -- plus `git push`, which talks to the network
#: and can sit there for minutes after its parent is gone.
#:
#: Deliberately NOT every line that starts a process. The `flock -n 9` line must
#: keep the descriptor it is locking, and the local, instant git plumbing cannot
#: cause the bug. A rule that fires on those is not stricter, it is wrong, and a
#: guard nobody can keep green is a guard that gets deleted.
_BACKGROUND = re.compile(r"&\s*$")
_SLEEP = re.compile(r"(?:^|\s|;)sleep\s+\S")
_PUSH = re.compile(r"(?:^|\s|;)git push\s")


def _spawn_lines() -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(PRESERVER.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("&&") or "flock -n 9" in stripped:
            continue
        if _BACKGROUND.search(stripped) or _SLEEP.search(stripped) or _PUSH.search(stripped):
            out.append((i, line))
    return out


def test_every_lasting_spawn_closes_the_lock_descriptor():
    offenders = [(i, l.strip()) for i, l in _spawn_lines() if "9>&-" not in l]
    assert not offenders, (
        "these start a process that can outlive this preserver, and it would "
        "inherit fd 9 and keep the flock alive after the preserver dies:\n  "
        + "\n  ".join(f"{i}: {l}" for i, l in offenders)
        + "\nAppend `9>&-` to each."
    )


def test_the_spawn_scan_sees_something():
    """A scan that matches nothing passes forever.

    The patterns are hand-written and the script they read changes; if a rename
    or a refactor made them match zero lines, the test above would go green by
    inspecting an empty list -- the same shape as every other guard in this repo
    that reported OK by not looking.
    """
    found = _spawn_lines()
    assert len(found) >= 5, (
        f"the spawn scan found only {len(found)} lines in preserve_evidence.sh; "
        f"it has stopped matching what it is meant to match")
    text = " ".join(l for _, l in found)
    assert 'sleep "$1"' in text, (
        "pause_preserver's long-lived sleep is no longer scanned; every tick "
        "is routed through that helper"
    )
    assert "git push" in text, "the network push is no longer scanned"
