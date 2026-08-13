"""A second supervisor cannot run, and the lock dies with its holder.

The driver has had a single-instance flock since it was written
(`clean_rollouts._only_driver()`). The supervisor that LAUNCHES the driver had
none, and the gap was covered by a manual runbook step: "check for duplicates,
keep the oldest by /proc mtime, kill the rest." Two supervisors really were
produced at 08:22 PDT 2026-08-12 by the watchdog racing a hand relaunch.

That step is itself the hazard. At 11:23 the same day an argv-exact scan
reported FOUR supervisors seconds after a restart; three were gone moments later
having never run a loop pass, and three hypotheses -- the scan self-matching,
setsid/nohup wrappers carrying the script path, a watchdog relaunch -- were each
tested and falsified, leaving the cause unknown. An operator following the
runbook against that snapshot would have killed live processes on evidence that
evaporated.

**The first attempt at this lock was reverted within the hour, and that failure
is what the second test below exists for.** Bash does not set O_CLOEXEC on a
descriptor opened by `exec`, so every child inherits fd 9 -- including the
driver, which runs for hours. An inherited fd keeps the flock alive, so the lock
survived its holder: a second supervisor was refused after the first was
SIGKILLed. Strictly worse than no lock. `_only_driver()` escapes this only
because PEP 446 makes Python's fds non-inheritable by default.

Two checks, and the order matters. The behavioural one is the evidence: it
starts a real supervisor, lets it spawn its real children, kills it, and demands
that a fresh one start. The grep is a cheap early warning for a spawn added
later without `9>&-` -- it would not have caught the original bug on its own,
because the original bug was the absence of the whole convention.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"


#: **The watchdog stays off for every supervisor this file starts.**
#:
#: These tests run the REAL `tools/supervisor.sh` against the REAL repo, which is
#: what makes them evidence. But `revive_watchdog()` launches the real
#: `daemon_watchdog.sh` on its first loop pass, and that watchdog relaunches
#: `supervisor.sh`, `heartbeat.sh` and `preserve_evidence.sh` -- detached, so they
#: OUTLIVE the test that started them. Measured 2026-08-12: one run of this file
#: left four daemons running against the working tree, among them a preserver
#: that commits `evidence/` and pushes it to a PUBLIC repository every 300s, and a
#: supervisor that can launch a paid sweep. A unit test must not be able to spend
#: money or publish.
#:
#: `supervisor.sh` documents this switch as being for "tests, and any run that
#: deliberately wants no watchdog"; it simply was not used here. Watchdog revival
#: has its own coverage in `test_the_supervisor_revives_the_watchdog.py`, which
#: does it properly -- against a fake repo holding a stub watchdog.
_NO_WATCHDOG = {"CCARC3_SUPERVISOR_WATCHES_WATCHDOG": "0"}


def _env(scratch: pathlib.Path, extra: dict | None = None) -> dict:
    env = dict(os.environ, CCARC3_SCRATCH=str(scratch), **_NO_WATCHDOG)
    env.update(extra or {})
    return env


def _spawn(scratch: pathlib.Path, log: pathlib.Path, env_extra: dict | None = None):
    return subprocess.Popen(["bash", str(SUPERVISOR)], cwd=str(REPO),
                            env=_env(scratch, env_extra),
                            stdout=log.open("w"), stderr=subprocess.STDOUT)


def _settle(proc, log: pathlib.Path, seconds: float = 4.0):
    """Give the supervisor time to take the lock and spawn its children."""
    time.sleep(seconds)
    assert proc.poll() is None, (
        "the supervisor exited instead of holding the lock:\n"
        + log.read_text(encoding="utf-8"))


@pytest.fixture
def scratch(tmp_path):
    # Brake engaged: the supervisor must not launch a real rollout from a test.
    (tmp_path / "concurrency").write_text("0\n", encoding="utf-8")
    return tmp_path


def _kill(proc):
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=20)


# ── the behavioural checks ───────────────────────────────────────────────────

def test_the_second_supervisor_refuses_to_start(scratch, tmp_path):
    first = _spawn(scratch, tmp_path / "a.log")
    try:
        _settle(first, tmp_path / "a.log")
        second = subprocess.run(
            ["bash", str(SUPERVISOR)], cwd=str(REPO),
            env=_env(scratch),
            capture_output=True, text=True, timeout=120)
        assert second.returncode == 0, second.stdout + second.stderr
        assert "another supervisor holds" in second.stdout, (
            "a second supervisor started alongside the first:\n"
            + second.stdout + second.stderr)
    finally:
        _kill(first)


def test_the_lock_dies_with_its_holder(scratch, tmp_path):
    """The regression. This is the test the first version of the lock failed.

    SIGKILL runs no cleanup -- which is the realistic case, since that is what a
    container replacement, an OOM kill, and an operator's `kill -9` all look
    like. If any surviving child still holds fd 9, the flock outlives the
    process and every future supervisor is refused.
    """
    first = _spawn(scratch, tmp_path / "a.log")
    _settle(first, tmp_path / "a.log")
    first.kill()
    first.wait(timeout=20)

    second = _spawn(scratch, tmp_path / "b.log")
    try:
        _settle(second, tmp_path / "b.log")
        text = (tmp_path / "b.log").read_text(encoding="utf-8")
        assert "another supervisor holds" not in text, (
            "the lock outlived its holder -- a child inherited fd 9 and is "
            "keeping it. Check that every spawn in supervisor.sh carries "
            "`9>&-`:\n" + text)
    finally:
        _kill(second)


def test_a_different_runner_gets_a_different_lock(scratch, tmp_path):
    """`supervisor.sh rerun_losses.py` is not a duplicate of the rollout arm.

    Keyed to the runner for the same reason `already_watching()` is keyed to the
    daemon set: refusing a supervisor that drives something else would make the
    second arm unstartable, and that exact mistake has been made in this repo
    before -- a guard so broad it made the behaviour untestable.
    """
    first = _spawn(scratch, tmp_path / "a.log")
    other = None
    try:
        _settle(first, tmp_path / "a.log")
        env = _env(scratch)
        other = subprocess.Popen(
            ["bash", str(SUPERVISOR), "tools/rerun_losses.py"], cwd=str(REPO),
            env=env, stdout=(tmp_path / "c.log").open("w"),
            stderr=subprocess.STDOUT)
        time.sleep(4)
        text = (tmp_path / "c.log").read_text(encoding="utf-8")
        assert "another supervisor holds" not in text, (
            "a supervisor for a different runner was refused as a duplicate:\n"
            + text)
    finally:
        _kill(other)
        _kill(first)


def test_the_switch_lets_a_test_run_two(scratch, tmp_path):
    first = _spawn(scratch, tmp_path / "a.log")
    second = None
    try:
        _settle(first, tmp_path / "a.log")
        second = _spawn(scratch, tmp_path / "b.log", {"CCARC3_SUPERVISOR_LOCK": "0"})
        _settle(second, tmp_path / "b.log")
    finally:
        _kill(second)
        _kill(first)


# ── the cheap early warning ──────────────────────────────────────────────────

#: **Narrowed, and the narrowing is the honest version rather than the
#: convenient one.** The first pattern here matched anything that starts a
#: process at all, on the reasoning that "a rule with a judgement call in it
#: cannot be checked". It flagged three lines that cannot cause the bug:
#:
#:   - `RUNNER_RE="...$(basename "$RUNNER" | sed ...)"` at script setup, which
#:     runs BEFORE the lock is taken and completes in microseconds;
#:   - the `tr | grep` pipeline inside `pids_of`, synchronous and instant;
#:   - `if ! flock -n 9` -- the lock acquisition itself, which obviously must
#:     keep the descriptor it is locking.
#:
#: A rule that fires on those is not stricter, it is wrong, and a guard nobody
#: can keep green is a guard that gets deleted. The property that actually
#: matters is narrower and still mechanical: **a process that can outlive the
#: statement which started it must not hold fd 9.** That is exactly two shapes
#: -- a background spawn (`&`), and `sleep`, which is synchronous but survives a
#: kill of its parent for as long as it was told to sleep. The `sleep 600` at
#: the bottom of the loop is the one that matters: ten minutes of a held lock is
#: long enough to break the recovery path.
_BACKGROUND = re.compile(r"&\s*$")
_SLEEP = re.compile(r"(?:^|\s|;)sleep\s+\S")


def _spawn_lines() -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(SUPERVISOR.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("&&") or "flock -n 9" in stripped:
            continue
        if _BACKGROUND.search(stripped) or _SLEEP.search(stripped):
            out.append((i, line))
    return out


def test_every_lasting_spawn_closes_the_lock_descriptor():
    offenders = [(i, l.strip()) for i, l in _spawn_lines() if "9>&-" not in l]
    assert not offenders, (
        "these start a process that can outlive this supervisor, and it would "
        "inherit fd 9 and keep the flock alive after the supervisor dies:\n  "
        + "\n  ".join(f"{i}: {l}" for i, l in offenders)
        + "\nAppend `9>&-` to each."
    )


def test_the_spawn_scan_sees_something():
    """A scan that matches nothing passes forever.

    The pattern is hand-written and the script it reads changes; if a rename or
    a refactor made it match zero lines, `test_every_spawn_closes...` would go
    green by looking at an empty list. Same shape as every other guard in this
    repo that reported OK by not looking.
    """
    found = _spawn_lines()
    assert len(found) >= 4, (
        f"the spawn scan found only {len(found)} lines in supervisor.sh; it has "
        f"stopped matching what it is meant to match")
    # It must see the two that matter by name, not just some count.
    text = " ".join(l for _, l in found)
    assert "sleep 600" in text, "the loop's long sleep is no longer scanned"
    assert "$RUNNER" in text, "the driver launch is no longer scanned"
